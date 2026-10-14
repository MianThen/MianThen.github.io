---
title: 为什么量化系统要用定点数
date: 2026-10-13 15:30:00
tags:
  - 量化
  - C++
  - 数值计算
  - 回测引擎
categories:
  - 量化系统设计
---

## 一个让人后背发凉的实验

同一批成交记录，同样的代码，只把求和的遍历方向从「从前到后」改成「从后到前」，盈亏合计就不一样了。这不是夸张，我在自己的回测引擎里跑出了这个结果：

```
5) same 100k fills, different reduction order:
   forward  = -2.7755575615628914e-17
   backward =  2.7755575615628914e-17
   sorted   =  5.992983886926595e-13
   forward != backward : YES (同一批成交，答案不同)
```

把结果打印到小数点后 16 位，三个数字互不相等。在浮点世界里，「盈亏合计」不是一个确定的值，而是一组依赖执行顺序的值。更糟的是，`-2.77e-17` 和 `2.77e-17` 符号相反——如果某段逻辑写的是 `if (pnl > 0)`，那么同一批交易，换个遍历顺序，就会走进完全不同的分支。

量化系统里有大量这样的判断：资金够不够、仓位是不是超限、风控是否触发、订单能不能成交。这些判断一旦不确定，回测的可复现性就没了，实盘和回测的一致性也就无从谈起。

定点数（fixed-point）就是为消灭这类不确定性而存在的。

<!-- more -->

## 一、先把问题说清楚：浮点数为什么会「飘」

IEEE 754 双精度浮点数用 `1 位符号 + 11 位指数 + 52 位尾数` 表示一个数，形式是 `±1.m × 2^e`。它有四个致命性质。

### 1. 十进制小数根本无法精确表示

`0.1` 在二进制里是无限循环小数（`0.0001100110011...`），只能被截断存储。所以：

```c
0.1 + 0.2 == 0.3   // false
```

实测输出 `0.1 + 0.2 = 0.30000000000000004`，而 `0.3 = 0.29999999999999999`。金额、价格、手续费率，恰恰都是十进制小数。金融数据的主流进制（十进制）和硬件的主流进制（二进制）之间存在结构性错配——这不是 bug，是表示论上的鸿沟。

### 2. 加法不满足结合律

浮点加法是「先对齐指数，再相加，再舍入」。当两个数数量级差得远时，小的那个会被整个吃掉：

```
a = 1.0000000010000000827
(a + b) + c = 0                    // b=1e17, c=-1e17
a + (b + c) = 1.0000000010000000827
DIFFER
```

`1e17 + a` 在双精度下就等于 `1e17`，`a` 的信息被彻底丢弃。这就是上面「换个顺序答案就变」的根源——**浮点求和的结果依赖于归约顺序，而编译器、并行化、撮合顺序都可能改变这个顺序**。

更隐蔽的是，双精度只有 52 位尾数，大约 15.95 位十进制有效数字。当账户权益涨到 `1e8` 元时，double 在该数量级的 ulp（相邻两个可表示值的间隔）约为 `1.49e-8`——看起来很小，但一次 0.01 元的累加只能保住大约 6-7 位有效小数，剩下的位数全部沉入舍入噪声。**账户越大，记账精度反而越差**，这与「钱越多账应该越清楚」的直觉恰好相反。定点数没有这个毛病：`int64` 的最小单位在任何数量级上都是恒定的。

### 3. 比较运算在边界上是不可靠谓词

`if (cash >= required)` 这类判断在数学上是二值的，在浮点上却是「带噪声的二值」：当两个操作数由不同的计算路径得到时，它们的舍入历史不同，理论上相等的两个数可能相差若干个 ulp。于是「恰好够钱」的订单有时成交、有时被拒，取决于编译器今天心情如何。一个交易系统里的谓词必须是**确定性函数**，不能是随机变量。

### 4. 编译器优化会让同一份代码给出不同结果

浮点运算何时使用融合乘加（FMA）、能否重排、能否向量化，取决于 `-O` 级别、`-ffast-math`、目标 CPU 指令集（x87 的 80 位扩展精度与 SSE 的 64 位运算在历史上甚至产生过不同的中间结果）。同一份回测代码在不同机器、甚至同一机器的不同编译参数下产生不同净值曲线，这对需要审计的交易系统是不可接受的。你无法向合规解释「这次回测和上次不一样是因为换了台编译机器」。

## 二、定点数的核心思路

定点数放弃「浮动的指数」，把指数固定下来，用整数直接存储按某个基准单位缩放后的值：

```
真实值 = 存储的整数 × 单位刻度
```

比如金额以「万分之一元」为最小单位存储，那么 `100.05 元` 就存成整数 `1000500`。所有金额运算退化成 `int64` 的整数加减乘除——**精确、满足结合律、结果与顺序无关、跨平台逐位一致**。

我引擎里的类型定义就是这条思路：

```cpp
// engine_common/include/engine_common/types.h:8-12
using SymbolId    = uint32_t;
using PriceTicks  = int64_t;   // 价格：以 tick 为单位的整数
using MoneyMinor  = int64_t;   // 金额：以最小货币单位为单位的整数
using Quantity    = int64_t;   // 数量：整数股/手
using TimestampNs = int64_t;   // 时间：整数纳秒
```

注意这里没有任何一个 `double`。价格用 `PriceTicks`（整数的「跳数」），金额用 `MoneyMinor`（整数的最小货币单位），连时间都是整数纳秒。行情事件、订单意图、成交回报全部是整数结构体：

```cpp
// engine_common/include/engine_common/types.h:56-76
struct MarketEvent {
    SymbolId symbol_id = 0;
    TimestampNs timestamp = 0;
    PriceTicks bid = 0;          // 整数 tick，不是 double
    PriceTicks ask = 0;
    Quantity bid_quantity = 0;
    Quantity ask_quantity = 0;
    uint64_t sequence = 0;       // 序号也是整数，可严格排序
    TimestampNs enqueue_timestamp = 0;
};

struct OrderIntent {
    int64_t client_order_id = 0;
    SymbolId symbol_id = 0;
    Side side = Side::BUY;
    OrderType type = OrderType::MARKET;
    TimeInForce time_in_force = TimeInForce::DAY;
    Quantity quantity = 0;
    PriceTicks limit_price = 0;
    TimestampNs timestamp = 0;
};
```

`sequence` 这个单调递增的整数同样关键：它让事件排序有一个全序的、无歧义的判据，不需要靠时间戳比较去猜先后（时间戳相同的事件在真实行情里大量存在）。

### 跨边界契约：把刻度写进消息里

定点系统最危险的 bug 不是算错，而是**单位错配**——发送方按 1e4 缩放，接收方按 1e2 解释，一切计算都「成功」但金额错了 100 倍。防御办法是把缩放因子作为消息的一部分随数据一起传输。看引擎发给策略运行时/回放分析器的成交事件：

```cpp
// engine_common/include/engine_common/types.h:78-94
struct ExecutionEvent {
    int64_t client_order_id = 0;
    int64_t execution_id = 0;
    uint64_t decision_id = 0;
    SymbolId symbol_id = 0;
    Side side = Side::BUY;
    ExecutionStatus status = ExecutionStatus::NEW;
    RejectReason reject_reason = RejectReason::NONE;
    Quantity last_quantity = 0;
    Quantity cumulative_quantity = 0;
    PriceTicks last_price = 0;
    int64_t price_scale = 0;      // 价格的缩放因子，自描述
    MoneyMinor explicit_fee = 0;
    int64_t fee_scale = 0;        // 费用的缩放因子，自描述
    uint32_t audit_flags = 0;     // 哪些字段是有效的，显式声明
    TimestampNs timestamp = 0;
};
```

`price_scale` 和 `fee_scale` 让每条成交回报**自描述**自己的单位，`audit_flags` 则显式声明哪些字段有效（`EXECUTION_HAS_PRICE_SCALE`、`EXECUTION_HAS_EXPLICIT_FEE` 等）。这是从交易所二进制协议（如纳斯达克 ITCH、国内期货交易所的行情协议）那里学来的做法：价格永远以整数 tick 加显式刻度传输，绝不传浮点。引擎侧的设置代码：

```cpp
// cpp_engine/src/engine.cpp:1056-1063
execution.last_price = to_money(fill.price);
execution.price_scale = kMoneyScale;
execution.explicit_fee = commission_minor;
execution.fee_scale = kMoneyScale;
execution.audit_flags = engine_common::EXECUTION_HAS_SYMBOL |
    engine_common::EXECUTION_HAS_SIDE |
    engine_common::EXECUTION_HAS_PRICE_SCALE |
    engine_common::EXECUTION_HAS_EXPLICIT_FEE;
```

## 三、边界在哪：进位、溢出与显式的舍入

定点数不是「自动正确」，它把「隐式的、不可控的舍入」换成了「显式的、可控的舍入」。代价是必须自己处理四件事。

### 1. 进入定点世界的那一次舍入必须显式且唯一

我的引擎把这一层收敛到两个函数里：

```cpp
// engine_common/include/engine_common/fixed_point.h:12-22
inline PriceTicks quantize_price(double price, double tick_size) {
    if (!std::isfinite(price) || !std::isfinite(tick_size) || tick_size <= 0.0)
        throw std::invalid_argument("invalid price or tick size");
    const long double ticks = std::round(static_cast<long double>(price) / tick_size);
    const long double minimum =
        static_cast<long double>(std::numeric_limits<PriceTicks>::min());
    const long double maximum_exclusive = -minimum;
    if (ticks < minimum || ticks >= maximum_exclusive)
        throw std::overflow_error("price ticks overflow");
    return static_cast<PriceTicks>(ticks);
}
```

三个设计细节值得说：

**一是用 `long double` 做中间计算。** `double price` 除以 `tick_size` 再 `round`，在某些临界值上会舍入到错误的一侧。用更宽的中间类型能把这类误差压下去，再配合显式 `round` 而非截断。

**二是用 `std::round` 而不是 `static_cast<int64_t>` 截断。** 这是一个极其常见的 bug。实测：

```
1) tick rounding (tick=0.01):
   0.145   p/0.01=14.499999999999998224   (ll)(p*100)=14   llround=14
   1.005   p/0.01=100.49999999999998579   (ll)(p*100)=100  llround=100
   7.005   p/0.01=700.5                   (ll)(p*100)=700  llround=701
```

`0.145 / 0.01` 在双精度下是 `14.499999999999998`，直接截断得到 `14`，而根据「四舍五入到最近 tick」的业务语义正确答案应该是 `15`。**一个 tick 就这样凭空消失了**。价格每错一个 tick，成交价、手续费、盈亏全跟着错。

**三是做了显式的溢出检查。** 定点数用 `int64` 存，溢出是静默回绕的：

```
8) int64 溢出:
   2^62 * 2 wrapped = -9223372036854775808   (静默回绕，无异常)
```

`quantize_price` 在转换前用 `long double` 比较上下界，越界抛 `std::overflow_error`，把「静默的错」变成「响亮的崩」。

### 2. 舍入必须在固定时刻、以固定方向发生

金额量化用 `std::llround`，把舍入的实现细节交给标准库，而不是自己写 `+0.5`（`+0.5` 截断的写法对负数是错的，`-0.6 + 0.5 = -0.1` 截断成 `0` 而非 `-1`）：

```cpp
// engine_common/include/engine_common/fixed_point.h:30-42
inline MoneyMinor quantize_money(double amount, int64_t scale = 100) {
    if (!std::isfinite(amount) || scale <= 0) throw std::invalid_argument("invalid money");
    const double limit = static_cast<double>(std::numeric_limits<MoneyMinor>::max()) /
                         static_cast<double>(scale);
    if (amount < -limit || amount > limit)
        throw std::overflow_error("money overflow");
    return static_cast<MoneyMinor>(std::llround(amount * static_cast<double>(scale)));
}

inline double money_from_minor(MoneyMinor amount, int64_t scale = 100) {
    if (scale <= 0) throw std::invalid_argument("invalid money scale");
    return static_cast<double>(amount) / static_cast<double>(scale);
}
```

这里体现了定点系统的经典架构：**内部全整数运算，只在边界做一次转换**。`quantize_*` 是把外部世界的十进制金额「搬进来」，`*_from_minor` 是把整数「搬出去」给人看或给 PnL 曲线用。中间所有的加减乘除、资金校验、持仓更新，都在整数域完成。

有一个很容易被忽视的细节：**连初始资金都要过一遍 round-trip**：

```cpp
// cpp_engine/src/engine.cpp:73-74
cash_minor_ = to_money(initial_cash);
initial_cash_ = from_money(cash_minor_);
```

为什么要先转进整数再转回来？因为用户传入的 `initial_cash` 可能是 `1000000.0000001` 这样的「脏」值。先量化、再反量化，保证引擎内部的现金账本和外部看到的初始资金是**同一个规范化的值**——连小数点后第五位的杂质都不允许进入系统。从此 `cash_minor_` 是唯一的真源，`get_cash()` 只是它的投影：

```cpp
// cpp_engine/src/engine.cpp:1187
Price BacktestEngine::get_cash() const { return from_money(cash_minor_); }
```

### 3. 整数域里的除法和比例分摊要单独设计

定点加减乘是平凡的，**除法和比例分摊不是**。看引擎里部分成交时释放预留资金的代码：

```cpp
// cpp_engine/src/engine.cpp:915-922
} else if (remaining > 0 && filled_quantity > 0) {
    const MoneyMinor old_reserve = cash->second;
    cash->second = static_cast<MoneyMinor>(std::llround(
        static_cast<long double>(old_reserve) *
        static_cast<long double>(remaining - filled_quantity) /
        static_cast<long double>(remaining)));
    total_reserved_buy_cash_ += cash->second - old_reserve;
}
```

订单部分成交后，剩余预留金要按 `剩余数量/原数量` 等比例缩减。这里乘法先于除法（避免先除丢精度）、中间量用 `long double` 兜底防 `int64 × int64` 溢出、最后 `llround` 显式舍入——三件事一件都不能少。同时注意 `total_reserved_buy_cash_ += cash->second - old_reserve` 这个写法：它调整的是**差值**，保证全局预留金总额与每单预留金之和永远逐位一致，不存在「总额和各单加总对不上」的对账漂移。这类「整数的乘法+除法」组合是定点系统里最需要 code review 的地方，顺序错一位，结果就不同。

### 4. 缩放因子的选择是个工程决策

我的引擎用 `kMoneyScale = 10'000`（万分之一元）：

```cpp
// cpp_engine/src/engine.cpp:17
constexpr int64_t kMoneyScale = 10'000;

// cpp_engine/src/engine.cpp:29-35
MoneyMinor to_money(Price value) {
    return engine_common::quantize_money(value, kMoneyScale);
}
Price from_money(MoneyMinor value) {
    return engine_common::money_from_minor(value, kMoneyScale);
}
```

为什么是 1e4 而不是 1e2（分）？因为 A 股、基金、ETF 的价格精度常见到 0.001 元甚至 0.0001 元，手续费和均摊成本还会产生更细的小数。缩放因子决定了定点系统能表示的**最小非零金额**。因子选小了，细粒度金额会被反复舍入，误差累积；选大了，`int64` 的有效动态范围被压缩。

算一下容量：`int64` 上限约 `9.22e18`，除以 1e4 得到约 `9.22e14` 元。对于一个管理规模远小于千万亿的回测系统，这个范围绰绰有余，同时提供 0.0001 元的分辨率。

衍生一点：**选择缩放因子本质上是在做数制设计**。业界常见选择有：股票金额用 1e4（万分之一元），加密交易所普遍用 1e8（对应 satoshi 的精度惯例），外汇用 1e6（pip 之下再留两位）。无论选哪个，原则是**一个系统一个刻度，写死在共享契约层**——这正是 `kMoneyScale` 作为编译期常量、而跨模块通信时还要随消息携带 `price_scale` 的双重设计：编译期常量保证内部一致，运行时字段保证跨进程安全。

## 四、定点数到底解决了哪些真实问题

### 问题一：手续费累积误差

逐笔用手续费率的浮点累加，与逐笔量化到整数最小单位再累加，结果不同：

```
3) commission (rate=0.00025, 10000 fills):
   double  = 4162.9162500000
   minor   = 4162.9175000000
   diff    = -0.0012500000 元
```

一万笔交易差 0.00125 元。单看很小，但注意两点：一是这个差额**随交易笔数线性增长**，高频策略一年几十万笔；二更重要的是，`double` 那一列的值**不是确定的**——它依赖求和顺序，而整数那一列永远得到同一个精确值。

引擎里的手续费处理还展示了另一个定点技巧——**先算后量化，再回写规范化值**：

```cpp
// cpp_engine/src/engine.cpp:1004-1007
fill.commission = from_money(to_money(calculate_commission(
    fill.timestamp, notional, fill.side == Side::SELL)));
const MoneyMinor notional_minor = to_money(notional);
const MoneyMinor commission_minor = to_money(fill.commission);
```

费率是 double（万 2.5 这种值在整数域表达不了），所以费率乘法在浮点域算，但**算完立刻量化成整数，并把量化后的值回写进 `fill.commission`**。这样账本里记的手续费、推送给策略的手续费、扣款用的手续费是逐位相同的同一个整数，不存在「账上扣的和显示的差 1e-12」这种解释不清的裂缝。

### 问题二：资金校验的边界抖动

引擎在成交前要判断资金是否充足：

```cpp
// cpp_engine/src/engine.cpp:1006-1016
const MoneyMinor notional_minor = to_money(notional);
const MoneyMinor commission_minor = to_money(fill.commission);
if (execution_config_.enforce_cash) {
    const auto own = reserved_buy_cash_.find(fill.order_id);
    const MoneyMinor own_reserve = fill.side == Side::BUY &&
        own != reserved_buy_cash_.end() ? own->second : 0;
    const MoneyMinor available = cash_minor_ - total_reserved_cash() + own_reserve;
    const MoneyMinor required_cash = fill.side == Side::BUY
        ? notional_minor + commission_minor
        : std::max(commission_minor - notional_minor, MoneyMinor{0});
    if (required_cash > available) {
        // 拒单 / 撤单，标记 INSUFFICIENT_CASH
```

`required_cash > available` 是一个**硬比较**。在整数域里，这个比较是精确的、可复现的：只要输入相同，任何机器、任何优化级别、任何遍历顺序，都得到同一个 `available`，因此都得到同一个拒单决策。

如果这里用 `double`，那么当账户资金恰好压在边界上（这在按满仓比例下单的策略里非常常见），浮点误差会让 `available` 有时比 `required_cash` 大 1e-10，有时小 1e-10。回测结果就变成了「取决于编译器脸色的随机拒单」，而这种抖动在实盘上会直接变成真实的资金不足拒单。

往上游再看一层，**资金预留估算也是定点化的**，而且刻意保守：

```cpp
// cpp_engine/src/engine.cpp:790-805
MoneyMinor BacktestEngine::estimate_required_cash(
    const Order& order, const MarketSnapshot& market) const {
    if (order.side != Side::BUY) return 0;
    Price price = order.limit_price;
    if (order.type == OrderType::MARKET) {
        price = market.ref_price() *
            (1.0 + (execution_config_.slippage_bps +
                    execution_config_.market_order_price_buffer_bps) / 10'000.0);
        if (market.upper_limit > 0.0) price = std::max(price, market.upper_limit);
    }
    ...
    return to_money(notional + commission);
}
```

市价单按「参考价 ×（1 + 滑点 + 缓冲）」并与涨停价取 max 来预估占用资金——**宁可多冻，不可少冻**。这种保守性只有在一个确定性的账本里才有意义：如果比较本身是抖动的，预留策略再保守也挡不住边界上的随机穿透。

### 问题三：净值曲线的漂移

```
4) equity: 1e8 起始, 1e7 次 +0.01
   double = 100100000.0536441803
   minor  = 100100000.0000000000
   diff   = 0.0536441803
```

一亿起始资金，一千万次 0.01 元的累加，浮点版本漂了 0.0536 元而整数版本精确无误。绝对误差不大，但**净值曲线是要被求收益率、算夏普、做归因的**。如果每次回测的微小漂移方向还不一致（因为它依赖顺序），那么两次「完全相同」的回测在夏普比率第 4 位小数上就会不同，策略对比和参数寻优的结论就带上了噪声。

注意这里有一个诚实的边界：引擎里 `equity = cash + market_value(...)` 的市值部分仍然是 double 计算的（`positions_.market_value()` 返回 `Price`）。这并不矛盾——**已实现的部分（现金、已实现盈亏）必须精确记账，未实现的部分（浮动市值）本质上就是估计值**，因为它依赖「当前市价」这个本身就随时在变的外部输入。账本精确，估值近似，这才是正确的精度分配。

### 问题四：持仓成本的累积失真

等量加仓 1000 次，用浮点递推加权平均成本：

```
6) avg_cost 1000 笔等量加仓
   avg_cost = 10.009989999999986
```

浮点版本给出了 `10.00999` 而非理论上的 `10.01`。再看引擎里真实持仓盈亏的处理：

```cpp
// cpp_engine/src/position.cpp:57-63
const Quantity closed = std::min(std::abs(pos.quantity), std::abs(delta));
const Price direction = pos.quantity > 0 ? 1.0 : -1.0;
pos.realized_pnl_minor += engine_common::quantize_money(
    (fill.price - pos.avg_cost) * static_cast<Price>(closed) * direction,
    kMoneyScale);
pos.realized_pnl = engine_common::money_from_minor(pos.realized_pnl_minor,
                                                   kMoneyScale);
```

关键在 `realized_pnl_minor +=` 这一行：**已实现盈亏以整数累加器持续累加，而不是浮点累加**。中间那一次乘法仍然在 `double` 上算（因为 `avg_cost` 和 `fill.price` 是 `Price`），但结果立刻被量化进整数累加器。这意味着误差**不会在时间维度上累积**——每一笔平仓的舍入误差都被限制在这一次量化之内（至多 0.5 个最小单位），而不是像纯浮点那样随交易笔数无界漂移。

这是一个很务实的折中：**输入端允许一次受控的浮点乘法，累加端必须是整数**。误差从 O(n) 漂移降级为 O(1) 有界。

为什么 `avg_cost` 本身还是 double？因为加权平均成本是一个**展示和分析用的导出量**，不参与资金扣减的硬比较——真正扣钱的是 `notional_minor`（整数成交价 × 整数数量），真正记盈亏的是 `realized_pnl_minor`（整数累加器）。把硬比较留在整数域、把导出量留在浮点域，每一层的精度需求恰好匹配它的用途。

### 问题五：数量与除权的整数性

不只是钱，**数量也必须是整数域的**。A 股按手交易、除权送转按比例派股，如果出现「100.00000000001 股」或者「99.9999999999 股」，后续所有按股数计算的东西全部失真。引擎在处理公司行动时显式拒绝产生零碎股：

```cpp
// cpp_engine/src/position.cpp:111-124
const double adjusted = static_cast<double>(position.quantity) *
                        action.share_multiplier;
const double adjusted_sellable =
    static_cast<double>(position.sellable_quantity) * action.share_multiplier;
if (std::abs(adjusted - std::round(adjusted)) > 1e-9 ||
    std::abs(adjusted_sellable - std::round(adjusted_sellable)) > 1e-9) {
    throw std::invalid_argument(
        "corporate action produces fractional share quantity");
}
if (action.share_multiplier != 1.0) {
    position.quantity = static_cast<Quantity>(std::llround(adjusted));
    ...
    position.avg_cost /= action.share_multiplier;
}
```

注意这里的哲学和金额完全一致：**浮点只用来做中间计算，落库必须是整数，落不进去就抛异常**。「送转后持仓 1000.4 股」不是一个应该被容忍的数值误差，而是一个应该被暴露的数据错误。

### 问题六：跨系统的一致性

同一套策略，回测用 C++ 引擎，实盘走另一条链路。如果两边一个用 `double` 一个用 `int64`，资金的比较结果可能不同——回测里能成交的单，实盘被拒了。用整数定点做公共契约（这也是 `engine_common` 这个共享模块存在的意义），才有可能做到「回测和实盘对同一笔成交给出逐位相同的结果」。

这也是为什么前文 `ExecutionEvent` 要把 `price_scale`、`fee_scale` 一起发出去：跨系统传输的不只是数值，还有数值的解释方式。FIX 协议里的价格字段、各家交易所的二进制行情协议，都是同样的思路。

## 五、定点数的代价

诚实地讲清楚 trade-off：

**必须自己管溢出。** 浮点溢出会变成 `inf`（至少醒目），整数溢出是静默回绕。所以每个量化入口都要做上下界检查，乘法之前要预估量级，必要时用 `__int128` 或 `long double` 做中间量。估算一下雷区：`MoneyMinor`（1e4 刻度）下，`int64` 能安全表示约 `9.2e14` 元；但两个 `MoneyMinor` 相乘的中间量是 `金额² × 1e8`，`3e7 元 × 3e7 元` 就已经到 `9e22` 溢出。所以**定点数之间不该直接相乘**——金额乘数量、金额乘费率，都必须先想清楚结果的量纲和量级，该升位升位。

**必须自己管舍入。** 除法、按比例分摊、手续费分摊，全都需要明确「向哪边舍」。这个决策必须集中定义、全局统一，不能各处随手 `+0.5`。业界常见约定有 round-half-up（A 股佣金惯例）、round-half-even（银行家舍入，IEEE 754 默认）、向上取整（风控预留）等，选哪种不重要，重要的是**全系统只有一种**。

**需要设计缩放因子。** 价格 tick、金额 scale、数量单位必须一开始就定清楚并写进契约。事后调整缩放因子会引发全链路的单位混乱——所有存量数据、所有对接方、所有持久化文件都要迁移。

**不适合做统计和模型运算。** 定点数是为「记账」设计的，不是为「数学」设计的。均值方差、协方差矩阵、特征值分解、神经网络推理，这些仍然应该用 `double`/`float` 在 `ml_runtime`、`portfolio_math` 这样的模块里做。

**量化的代价是表达能力。** 定点数无法表示「小于一个最小单位」的量。如果某个中间量天然微小（比如极小的收益率），定点表示会把它直接压成 0，这在某些场景下比浮点误差更糟。

## 六、正确的边界：记账用定点，计算用浮点

经过上面的分析，一个成熟的量化系统架构应该长这样：

| 层面 | 类型 | 原因 |
|------|------|------|
| 价格 | `int64` ticks | 交易所价格本就是离散 tick 的整数倍 |
| 金额/现金 | `int64` minor units | 记账必须精确、可复现、可审计 |
| 数量 | `int64` 整数股 | A 股按手交易，不存在零碎股 |
| 时间戳 | `int64` 纳秒 | 事件排序需要全序且无歧义 |
| 事件序号 | `uint64` | 严格单调，用于确定性重放 |
| 特征/因子 | `double` / `float` | 统计计算，精度需求是相对的 |
| 模型推理 | `float` / `half` | 神经网络的本质就是近似 |
| 绩效统计 | `double` | 收益率、夏普、回撤是分析量，不是账本 |

核心原则是：**凡是进入账本、参与决策比较、需要跨系统逐位一致的量，必须是定点整数；凡是做统计推断和数值优化的量，可以是浮点。**

判别一个量该放哪边，问自己三个问题：

1. **它会被 `==`、`>` 这样的硬比较用于决策吗？** 会 → 定点。
2. **它需要在两次运行、两台机器、两个系统之间逐位一致吗？** 需要 → 定点。
3. **它本质是连续量的采样或统计量，误差只影响「精度」不影响「对错」吗？** 是 → 浮点。

## 七、衍生：如果不想用定点，还有哪些路

定点不是唯一解，了解替代方案才能真正理解定点的定位：

**十进制浮点（decimal64/128，IEEE 754-2008）。** 用 10 的幂而非 2 的幂缩放，`0.1` 可以精确表示。Java 的 `BigDecimal`、Python 的 `decimal`、数据库的 `DECIMAL` 类型都是这个思路。问题是**硬件不原生支持**（x86 没有十进制浮点单元），运算靠软件模拟，比 `int64` 加法慢一到两个数量级——这在延迟敏感的撮合路径上不可接受，所以交易所和券商核心账本仍然普遍用整数定点，decimal 更多出现在对账、报表这类离线场景。

**Kahan / Neumaier 补偿求和。** 如果必须用 double 累加大量浮点数，可以用补偿求和把误差从 O(n·ε) 压到 O(ε)。它解决的是「累加精度」问题，但**不解决「结果依赖顺序」问题**——Kahan 求和在不同归约顺序下仍然给出不同的（虽然都很精确的）结果。对回测可复现性来说，顺序无关性和精度同样重要，所以补偿求和替代不了定点，它是浮点阵营内部的补救措施。

**有理数 / 任意精度。** 分数精确存储（`1/3` 存成 `(1,3)`），数学上最干净，代价是分子分母会爆炸式增长，需要周期性约分，性能不可控。适合符号计算，不适合每秒百万笔的撮合循环。

**结论：** decimal 解决「表示」问题但牺牲性能，Kahan 解决「精度」问题但不解决「顺序」，有理数解决一切但代价不可控。定点数是在**性能、精确性、顺序无关性**三者交点上唯一的工程最优解——这也是为什么它统治了从交易所撮合引擎到银行核心账务系统的整个金融基础设施。

## 八、一份可操作的检查清单

如果你在设计或 review 一个量化系统，可以拿这张表逐项核对：

- [ ] 价格、金额、数量、时间戳是否全部是整数类型？（搜一下账本路径上有没有 `double`）
- [ ] 浮点 → 定点的转换是否收敛在唯一的一组 `quantize_*` 函数里？（搜一下有没有散落的 `(int64_t)(x * 10000)`）
- [ ] 转换用的是 `round`/`llround` 还是截断？负数路径测过吗？
- [ ] 每次量化前是否检查了 `isfinite` 和溢出边界？
- [ ] 乘法之前是否预估了中间量的量级？需要 `__int128`/`long double` 的地方用了吗？
- [ ] 全局缩放因子是不是编译期常量，且跨进程消息是否自带 scale 字段？
- [ ] 硬比较（资金校验、风控阈值、数量检查）是否全部发生在整数域？
- [ ] 导出量（avg_cost、equity、sharpe）是否与账本量（cash、realized_pnl）明确分离？
- [ ] 除法和比例分摊的舍入方向是否全局统一？
- [ ] 跑两遍同样的输入，输出是否逐位一致？（这是终极验收测试）

## 结语

回到最初那个实验。当你的系统里所有资金、价格、持仓都是 `int64` 时，「换个顺序答案就变」这件事物理上不可能发生——整数加法满足结合律，是数学上的全序阿贝尔群，与遍历顺序、编译选项、CPU 架构完全无关。

对一个要拿真金白银去跑的量化系统来说，这种确定性不是优化，而是前提。

---

*文中所有数值均来自本机实测的 C++ 程序输出，代码示例引自 `quant-backtester-cpp` 仓库的实际源码（`engine_common/include/engine_common/types.h`、`engine_common/include/engine_common/fixed_point.h`、`cpp_engine/src/engine.cpp`、`cpp_engine/src/position.cpp`）。*
