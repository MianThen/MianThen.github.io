---
title: 我给 A 股选股模型写了一个 Transformer：多任务架构、排序损失消融，以及一连串诚实的负结果
date: 2026-10-15 16:30:00
tags:
  - 量化
  - Transformer
  - 机器学习
  - 回测引擎
  - PyTorch
categories:
  - 量化系统设计
---

上一篇文章聊了回测引擎为什么要用定点数。这篇聊引擎上游的那个「大脑」：一个跑在 A 股日频截面上的时序 Transformer 选股模型。

这不是一篇「我的模型年化多少」的炫耀贴。恰恰相反，这篇文章里最有价值的部分，是五个精心设计的挑战者（ListMLE、LambdaLoss、Kendall 动态权重、PCGrad、GradNorm、对抗训练、对比预训练）**全部被预注册的实验门禁挡在门外**的过程。最终活下来的，是最朴素的那个基线。如果你只对「什么有效」感兴趣，这篇文章可能会让你失望；如果你对「怎么证明自己没被骗」感兴趣，希望这些记录有用。

模型代码在 `python/qbt_ml/`（Python 研究层），推理制品最终落到 C++ 回测引擎里跑。全文所有数字都来自冻结的实验报告，不是事后回忆。

<!-- more -->

## 零、任务定义：这不是股价预测

先把问题钉死。很多「Transformer 预测股价」的项目死在这一步——问题定义本身就不成立。

我的任务是**截面排序**：每天收盘后，给股票池里约 120 只股票打分，选 Top-20 持有，下一个交易日开盘成交（`NEXT_OPEN` 对齐）。模型不需要预测「明天涨多少」，只需要回答「明天开盘后第 5 根 K 线收盘时，哪 20 只的相对排名更靠前」。

这个定义直接决定了后面所有的设计：

- 每个训练 batch 是**一个完整 timestamp 的截面**（同一时刻的所有股票），而不是随机打散的样本——因为排序损失是截面内的相对比较；
- 标签从「下一根开盘买入、持有 5 根 K 线后卖出」的可成交路径构造，**成交时点被显式建模进标签里**，不存在「用收盘价预测收盘价」的偷价；
- 主指标是 NDCG@20、RankIC、top-bottom 收益差和换手率——没有一个指标是「预测精度」。

## 一、模型架构：一个刻意做小的因果 Transformer

`TemporalTransformerV1`（`python/qbt_ml/models/temporal_transformer.py`）的全部超参：

```python
@dataclass(frozen=True)
class TemporalTransformerConfig:
    feature_count: int          # 23 个特征
    lookback: int = 64          # 回看 64 根 K 线
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
```

d_model=64、4 头、3 层，放在今天属于「袖珍模型」。这是有意的：特征只有 23 个，回看窗口只有 64 根 K 线，样本量是截面级的（每天一个 batch），模型再大只会过拟合噪音。另外这个模型最终要导出 ONNX 塞进 C++ 引擎做逐分钟推理，小模型的延迟和内存开销都可控。

### 因果掩码与填充掩码

输入是 `[batch, 64, 23]` 的窗口，两个掩码叠加：

```python
# 因果掩码：t 时刻只能用 t 及之前的信息
causal_mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)

# 填充掩码：padding 位置不参与注意力
src_key_padding_mask = input.isnan().any(dim=-1)  # 实际更严格
```

值得说一句 padding：样本池里的股票上市时间不同，前几根 K 线往往是缺失的。我们用**前向填充 + 尾端对齐**构造窗口，缺失部分用 mask 挡住，不参与注意力计算。

### 最后一个有效位置池化

Transformer 编码器输出 `[batch, 64, 64]`，怎么聚合成一个向量？最直接的做法是取序列最后一个位置——但 padding 股票的最后几根是无效的。实现里先算出每只股票最后一个有效位置，只取那个位置的 hidden state 作为序列表示：

```python
# last_valid_idx: [batch]
last_hidden = out[torch.arange(batch), last_valid_idx]
```

这个池化同时承担了「对齐到当前时刻」的作用：模型永远只输出「以最后一根有效 K 线为当前时刻」的截面表示，不存在未来信息泄漏。

### 六个头

序列表示之后接六个平行的预测头：

| 头 | 输出 | 作用 |
| --- | --- | --- |
| expected_return | 连续值 | 主排序分数 |
| expected_volatility | 连续值 | 风险估计 |
| direction_probability | (0,1) | 上涨概率 |
| lower_quantile / upper_quantile | 连续值 | 区间估计（用 `torch.sort` 保证单调） |
| confidence | (0,1) | 排序置信度，用于交易闸门 |

六个头共享 encoder，但梯度通过多任务权重组合回传。后文会看到，**怎么组合这六个头的梯度，本身就是一整轮消融实验**。

### Scaler 以 buffer 形式注册进模型

特征标准化（z-score）的 mean / scale 不是训练脚本里的全局变量，而是 `register_buffer` 进模型，随模型一起导出：

```python
self.register_buffer("scaler_mean", ...)
self.register_buffer("scaler_scale", ...)
self.register_buffer("input_protected", ...)
```

其中 `input_protected` 是 4 个布尔特征（停牌、上市、ST、可交易）的标记，这 4 个特征**跳过 z-score**，直接以 0/1 进网络。这个细节在后文「推理与 C++ 对齐」里会再出现——标准化参数进 buffer，是从根上杜绝「训练和推理标准化不一致」的一步。

## 二、特征：BAR_V1 的 23 维

特征构建器叫 `BAR_V1`（`python/qbt_ml/features/bar_v1.py`），全部从 OHLCV + 复权因子 + 基础属性计算，一共 23 维：

**动量/收益类（6 个）**：过去 1/5/10/20 根 K 线的对数收益。注意用的是对数收益，不是百分比收益。

**波动类（4 个）**：过去 5/10/20/60 根 K 线的收益波动率。

**量价类（4 个）**：成交量的滚动 z-score（`_rolling_zscore`）、价格相对均线的偏离（`ma_deviation`）、价格在近期区间的相对位置（`price_position`，一个 0-1 区间内的值）、区间突破指标（`breakout`）。

**横截面类（1 个）**：当日收益的横截面排名（`cross_section_return_rank`）。这一维直接编码了「今天谁涨得比别人多」的截面信息。

**状态类（8 个）**：停牌标志、上市标志、ST 标志、可交易标志，以及它们的若干组合——这 8 个布尔特征里只有 4 个需要标准化，其余直接 0/1。

关键设计：**信号价格 = 原始价格 × 复权因子**（`signal_price = raw_price * adjustment_factor`）。所有收益计算都用复权价格，避免分红除权造成虚假的收益尖刺。特征有效性由 `valid_mask`（finite ∧ 已上市 ∧ 未停牌）保证，无效样本不进入标签构造。

## 三、标签：把「可成交」写进监督信号

`LabelSpecV2`（`python/qbt_ml/labels/spec_v2.py`）管标签。六个头对应六类标签，最值得展开的是排序类标签。

**收益标签**：下一根开盘买入、持有 5 根 K 线后的复权收益（`return_raw`）。训练对齐是 `NEXT_OPEN`，成交时点被显式建模——模型学的是「开盘买入持有 5 根」的真实可成交收益，不是收盘对收盘的幻觉。

**方向标签**：`sigmoid((r - 0) / 0.0025)` 的软标签，即用 0.25% 的尺度把收益压到 (0,1)，软化二值方向，让梯度更平滑。

**排序标签**：核心是 `rank_utility`。原始收益先做 1%/99% 的 winsorize（缩尾）再转成 [0,1] 的 relevance 分数，作为排序损失（NDCG、ListMLE、LambdaLoss）的监督目标。winsorize 这一步很关键：A 股日收益长尾极重，不复权 outlier 会在排序损失里主宰梯度。

所有标签的计算代码、参数、版本都被 SHA-256 进 manifest（见第十节），标签口径变更会直接改变制品哈希——这是可复现性的最后一道锁。

## 四、训练协议：Purged Walk-Forward

样本是时序的，就不能随便切 train/val/test。我们用的是带 purging 和 embargo 的 walk-forward：

```
train: 504 根  →  purge: 6 根  →  val: 126 根  →  purge: 6 根 + embargo: 5 根  →  test: 126 根
```

每个 fold 往前滚。purge 是因为相邻 K 线的标签窗口重叠（持有 5 根意味着 t 和 t+4 的标签共享收益），不 purge 就会把测试信息漏进训练集。embargo 是再留 5 根安全边际。

评估指标全部**按 timestamp 等权**聚合（`timestamp_walk_forward_splits` + `ranking_oos_metrics`）：NDCG@20、Precision@20、RankIC、top-bottom 收益差、换手率。之所以强调按 timestamp 等权，是因为截面样本数不均（有的日期 110 只、有的 60 只），按样本加权会被大截面日期带偏。

## 五、排序损失消融：基线赢了，而且赢得很干净

这是全项目最「反直觉」的一轮实验（`docs/phase1b_experiment_report.md`）。三个排序损失候选：

- **legacy**：`-Pearson(score, relevance)`，即最大化打分与 relevance 的截面相关系数；
- **ListMLE**：Plackett-Luce 模型的极大似然，直接把「排对序」的概率写进损失；
- **LambdaLoss**：按 ΔNDCG@20 给错误排序对加权，业界标准做法。

预注册假设是 ListMLE 或 LambdaLoss 至少在某几个指标上超过 legacy。结果（OOS，fold 聚合）：

| 损失 | NDCG@20 | RankIC | spread | turnover |
| --- | --- | --- | --- | --- |
| legacy | **0.6397** | **0.0548** | **+0.00284** | 0.1176 |
| ListMLE | 0.6309 | 0.0412 | +0.00179 | **0.1087** |
| LambdaLoss | 0.6131 | -0.0224 | -0.00216 | 0.1205 |

legacy 全面胜出；LambdaLoss 的 RankIC 直接转负。我们事后分析：在 64 根窗口、3 层的小模型上，ListMLE/LambdaLoss 这类「逐对/逐排列」损失信号太稀疏、方差太大，反而不如 Pearson 这种全局平滑的相关性损失稳定。**理论上更优雅的排序损失在小数据、小模型、长尾标签的截面任务上输给了朴素基线**——这个结论我们写了整整一页的 postmortem，包括为什么不能就此断言「ListMLE 没用」（也许是温度、采样、标签 winsorize 的交互问题），但按预注册纪律，本轮它就是输了。

## 六、不确定性加权：Kendall 权重的一次惨烈翻车

多任务损失权重是另一个经典的坑。我们试了**基于 Kendall 同方差假设的自动权重**（`KendallTaskWeights`）：让每个任务的可学习对数方差 `s_k` 参与梯度，理论上模型会自动学会给噪声大的任务降权。

翻车点在于**数值**：波动率任务的残差比收益任务小两个数量级，它的 `s_k` 在优化中一路滑向负方向，被 clamp 在 -6 之后，权重变成 `exp(-(-6)) ≈ 403`。也就是说，模型把 99% 的梯度预算花在了波动率头上，收益排序的监督基本被饿死。

OOS 对比（legacy + Kendall vs legacy 固定权重）：

| 配置 | NDCG@20 | RankIC | spread |
| --- | --- | --- | --- |
| legacy（固定权重） | 0.6397 | 0.0548 | +0.00284 |
| legacy + Kendall | 0.6384 | 0.0368 | +0.00134 |

NDCG 只掉了 0.21%，看起来「还好」；但 RankIC 掉了 32.9%，spread 掉了 52.8%。NDCG 是排序头指标所以被 gate 拦下来了，但这件事的教训是：**只看头指标会漏掉风险维度的退化**。Kendall 权重被废弃，rank 任务的权重固定为 0.1。

## 七、梯度手术：PCGrad 与 GradNorm

六头共享 encoder，梯度打架是真实存在的。我们试了两种多任务梯度处理：

- **PCGrad**：任务梯度两两投影、剔除冲突分量；
- **GradNorm-4**：按梯度范数动态平衡任务。

两组都提升了 RankIC（+0.0070 / +0.0071），但 NDCG、Precision、换手率全部退化（完整表格见 `docs/phase1e_experiment_report.md`）。PCGrad 的投影频率约 0.55——即超过一半的梯度对存在冲突，机制上确实在干活，但**把冲突强行剔除掉之后，模型反而失去了用冲突梯度学到的某种平衡**。Gate 全部未通过。

这一轮最诚实的注释是：我们**没有**把「RankIC 提升」写成「PCGrad 有效」。RankIC 提升但组合层面全面退化，说明 RankIC 单指标可能被梯度手术带偏了。预注册的门禁拦下的不是「有效」的假象，而是「局部指标有效」的假象。

## 八、对抗训练：三道门全关

最后的大招组合拳，全部以「鲁棒性」之名：

- **direction_apl**：方向头的对称交叉熵（NCE + RCE），应对标签噪音；
- **latent_fgm**：对 encoder 隐层做 FGM 扰动（`r = ε·g/‖g‖₂`）；
- **feature_pgd**：对特征输入做 PGD 对抗扰动（mask-aware 的 L∞ 约束）。

预注册门禁要求：对抗训练后**干净样本的 OOS 指标不得退化超过阈值**，且对抗样本上的鲁棒性确实提升。结果（`phase2b_oos_report.json`）：

- `clean_non_degraded = false`：干净 OOS 上 return_mae 退化 +0.0707，超过容许的退化窗口；
- 所有变体（none / direction_apl / latent_fgm / feature_pgd）在全部噪音率（0.05/0.1/0.2）下均未通过 gate。

也就是说，三种对抗训练都没有在「不牺牲干净性能」的前提下通过验证。direction_apl 属于那种「理论上你一定想试」的方法——它最终把方向头的鲁棒性提上去了，但代价是主收益头退化。多任务模型里，**一个头的鲁棒性提升以另一个头的性能为代价，这笔交易我们没有做成**。

## 九、对比预训练：InfoTS 的体面退场

还有一条并行线：用对比学习（InfoNCE）在无标签数据上预训练时序表示，再微调排序头。`phase5_status.md` 里这条线被正式 CLOSED——不是因为爆了，而是因为预训练收益不足以覆盖工程复杂度，在预算盘点时被砍掉。这是全项目唯一一个「主动放弃」而非「被打败」的实验。记录在案，防止后人重复造轮子。

## 十、制品契约：ONNX + Manifest + 黄金输入输出

模型要进 C++ 引擎，不能靠「把 PyTorch 模型拷过去」。制品契约（`docs/champion_artifact_registration.md`）：

- **ONNX**：opset 18，导出脚本 `python/qbt_ml/export/onnx_export.py`；
- **manifest.json**：记录 `model_sha256`、`feature_schema_sha256`、`label_spec_sha256`、`ranking_score_spec_sha256`、calendar/universe id、data cutoff——**特征、标签、打分口径任何一个变了，模型哈希就对不上**，引擎会拒载；
- **黄金 I/O**：一组固定输入 → 固定输出，C++ 推理结果必须逐位对齐。

这里有一个真实的 bug 值得讲：导出后我们发现 ONNX 推理在某些输入下出现 NaN——原因是序列尾部 padding 位置的特征在标准化后产生 `0 × inf`。修复是 `SafeExportWrapper`：`torch.where(isfinite(x), x, 0)`，在导出层把非有限值显式归零，而不是指望运行时兜底。修复后 C++（ONNX Runtime 1.19.2 arm64）与 PyTorch 在 1587 个截面、零错误率下完全一致，NaN 归零修复前后行为差异全部记录。

标准化参数（mean/scale/protected）作为 buffer 进模型，随 ONNX 一起导出——第一节埋的伏笔在这里兑现：**推理时的标准化就是训练时的标准化，没有第二套口径**。

## 十一、模拟盘：17 个月，以及三个还没解决的诚实问题

冠军模型 `champion-temporal-transformer-v1`（legacy 损失、fold-3、safe 导出）在 2025-04 到 2026-09 的模拟盘上跑了 17 个月：

- 总收益 **+15.80%（毛）**，扣显式成本后 **+14.45%**；
- 按 14bp 滑点再扣一遍，**+12.12%**；
- 换手率约 4.4 倍/年——注意回测里的换手率被 NDCG 的 spread 压制，说明收益主要来自 alpha 而非频繁交易。

模拟盘暴露了三个我们没有答案的问题，如实记录：

**1. 水平线错配**。训练标签是持有 5 根 K 线，但 OOS 诊断显示 RankIC 在 20 根水平上最高（0.0817, t=9.18），5 根上只有 0.0599（t=6.43）。模型学的其实是「更慢的 alpha」，而交易系统按 5 根持有（实际中位持有 6 天）。这条我们还没想清楚怎么改。

**2. 候选池变薄**。Python 研究层每个截面有约 108 个有效窗口，但 C++ 引擎里同一时刻只有 9-16 只股票给出有效预测。差了一个数量级。嫌疑在 C++ 侧的特征管线 `bar_v1_feature_pipeline.cpp`：`valid_` 的判定（finite ∧ tradable）可能与研究层语义不一致，导致候选池被过度裁剪。这条线还在排查，没定性。

**3. 置信度闸门把订单全挡了**。模拟盘配置了 `minimum_confidence = 0.55`，结果整整 17 个月 **0 笔成交**——`champion-artifact-backtest/backtest_summary.json` 里 orders=0。闸门设置太保守，或者 confidence 头的校准有问题（CQR 覆盖率的 OOS 是 0.7840/0.8998/0.8397，波动率区间的校准还行，但 confidence 头是另一套）。这是当前最优先的待办。

## 十二、这一趟下来，我们学到了什么

写到最后，把能沉淀的沉淀下来：

1. **预注册实验设计是防自欺的唯一武器**。这一整轮消融里，几乎每个「听起来更高级」的方法都在局部指标上给过我们希望（Kendall 的 NDCG 几乎没掉、PCGrad 的 RankIC 提升、对抗训练的方向鲁棒性），如果实验不是提前写死 gate、不是 fail-closed，我们大概率会挑一个「局部好看」的方法上生产，然后在真实盘里付学费。

2. **负结果要留证据链**。ListMLE 输给了 Pearson、LambdaLoss RankIC 转负、Kendall 权重翻车、PCGrad/GradNorm 组合退化、三种对抗训练全部未过门——每一个都被完整记录（实验配置、种子、fold 明细、postmortem），并带 SHA-256 契约哈希。这些不是「失败」，是防止后人重复烧钱的公共品。

3. **模型门禁在经济学门禁之前，且模型门禁独立于经济学门禁**。整个 2026 年，`promotion_eligible = false` 始终成立——模型可以因为 alpha 不够、也可以因为风险维度退化（如 Kendall 的 spread 腰斩）被拒，两者是独立的 gate，任何一个没过都不上生产。模拟盘的 0 成交就是这条纪律的直接后果：**宁可空仓 17 个月，也不把未达标的模型放出去**。

4. **小模型 + 干净的实验纪律，比大模型 + 混乱的实验流程产出更多可用的知识**。d_model=64 的模型，算力成本可以忽略不计，换来的是每一轮实验都能跑几十个 fold、每一条结论都能重复验证。

5. 最后一条，也是最有争议的一条：**在这个数据体量下，朴素基线没有被超越，可能不是因为我们不够努力，而是因为问题本身的信息量就那么多**。A 股截面、5 根持有、120 只股票、17 个月模拟盘——也许这个设定下，能榨出来的 alpha 就这么多。把这一点写下来，比假装我们找到了银弹，对下一个做这件事的人更有用。
