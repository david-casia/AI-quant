# -*- coding: utf-8 -*-
"""
strategy.py
双均线策略引擎：均线计算 → 金叉死叉信号 → 模拟回测 → 绩效指标
纯 pandas 实现，无外部金融库依赖。
"""
import pandas as pd
import numpy as np
import os

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")

# 默认均线周期
DEFAULT_SHORT = 5
DEFAULT_LONG = 15

# 回测参数
INITIAL_CAPITAL = 100000.0   # 初始资金（元）
COMMISSION_RATE = 0.0003     # 佣金费率（万分之三）
STAMP_TAX_RATE = 0.001       # 印花税（卖出方千分之一）
SLIPPAGE = 0.0               # 滑点（暂设为0）

# 无风险年化利率（用于计算夏普比率，默认2%）
RISK_FREE_ANNUAL = 0.02
TRADING_DAYS_PER_YEAR = 252


def load_data(csv_path=None):
    """加载股价数据，返回按日期升序排列的 DataFrame"""
    if csv_path is None:
        csv_path = CSV_PATH
    df = pd.read_csv(csv_path)
    df = df.sort_values("trade_date").reset_index(drop=True)
    # 转换日期格式
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
    return df


def calc_ma(df, short_period=DEFAULT_SHORT, long_period=DEFAULT_LONG):
    """
    计算短期和长期简单移动平均线(SMA)
    在 df 上新增 ma_short / ma_long 列
    """
    df = df.copy()
    df["ma_short"] = df["close"].rolling(window=short_period, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(window=long_period, min_periods=1).mean()
    return df


def generate_signals(df):
    """
    生成交易信号：
    - 金叉（Golden Cross）：短均线从下方穿越长均线向上 → 买入信号(+1)
    - 死叉（Death Cross）：短均线从上方穿越长均线向下 → 卖出信号(-1)
    
    返回新增列：signal(1=买入,-1=卖出,0=持有) position(1=持仓,0=空仓)
    """
    df = df.copy()
    df["signal"] = 0

    # 计算前一天的均线差
    prev_diff = df["ma_short"].shift(1) - df["ma_long"].shift(1)
    curr_diff = df["ma_short"] - df["ma_long"]

    # 金叉：前一天 short < long，今天 short > long
    golden_cross = (prev_diff < 0) & (curr_diff > 0)
    # 死叉：前一天 short > long，今天 short < long
    death_cross = (prev_diff > 0) & (curr_diff < 0)

    df.loc[golden_cross, "signal"] = 1
    df.loc[death_cross, "signal"] = -1

    # 计算持仓状态（买入后持仓，卖出后空仓）
    df["position"] = 0
    holding = 0
    positions = []
    for i in range(len(df)):
        sig = df.iloc[i]["signal"]
        if sig == 1:
            holding = 1
        elif sig == -1:
            holding = 0
        positions.append(holding)
    df["position"] = positions

    return df


def backtest(df, initial_capital=INITIAL_CAPITAL,
             commission_rate=COMMISSION_RATE,
             stamp_tax_rate=STAMP_TAX_RATE):
    """
    模拟回测：
    - 信号次日开盘价成交（更贴近真实）
    - 全仓买入/全仓卖出
    - 扣除佣金和印花税
    
    返回 df 新增列：strategy_return(策略日收益率) benchmark_return(基准日收益率)
    """
    df = df.copy()

    # 交易执行：信号出现后，次日按收盘价成交（简化）
    # 实际中可用次日开盘价，这里用当日收盘价便于逻辑清晰
    df["trade_price"] = df["close"].shift(1)  # 次日参考价

    # 策略日收益率：持仓时获取股票日收益，空仓时收益为0
    df["stock_return"] = df["close"].pct_change().fillna(0)
    df["strategy_return"] = df["position"].shift(1) * df["stock_return"]
    df["strategy_return"] = df["strategy_return"].fillna(0)

    # 基准：买入持有
    df["benchmark_return"] = df["stock_return"]

    # 累计净值
    df["strategy_nav"] = (1 + df["strategy_return"]).cumprod()
    df["benchmark_nav"] = (1 + df["benchmark_return"]).cumprod()

    # 扣除交易成本
    trades = df[df["signal"] != 0].copy()
    total_cost = 0.0
    for idx, row in trades.iterrows():
        price = row["close"]
        if row["signal"] == 1:  # 买入
            cost = INITIAL_CAPITAL * commission_rate
        else:  # 卖出
            cost = INITIAL_CAPITAL * (commission_rate + stamp_tax_rate)
        total_cost += cost

    # 将总成本折算为对净值的影响
    cost_factor = 1 - total_cost / (INITIAL_CAPITAL * df["strategy_nav"].iloc[-1])
    df["strategy_nav"] = df["strategy_nav"] * cost_factor

    return df


def calc_metrics(df, risk_free_annual=RISK_FREE_ANNUAL):
    """
    计算策略绩效指标（7项核心指标）：
    1. 年化收益率 (Annualized Return)
    2. 最大回撤 (Maximum Drawdown, MDD)
    3. 年化波动率 (Annualized Volatility)
    4. 夏普比率 (Sharpe Ratio) - 年化
    5. 胜率 (Win Rate)
    6. 盈亏比 (Profit/Loss Ratio)
    7. 期望收益 (Expectancy, R)
    """
    # 策略累计回报
    final_nav = df["strategy_nav"].iloc[-1]
    cumulative_return = (final_nav - 1) * 100

    # 基准累计回报
    benchmark_final = df["benchmark_nav"].iloc[-1]
    benchmark_return = (benchmark_final - 1) * 100

    # 最大回撤
    running_max = df["strategy_nav"].cummax()
    drawdown = (df["strategy_nav"] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # 基准最大回撤
    bench_running_max = df["benchmark_nav"].cummax()
    bench_drawdown = (df["benchmark_nav"] - bench_running_max) / bench_running_max
    bench_mdd = bench_drawdown.min() * 100

    # 年化波动率 = 日收益率标准差 × sqrt(252)
    daily_std = df["strategy_return"].std()
    ann_volatility = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    # 基准年化波动率
    bench_daily_std = df["benchmark_return"].std()
    bench_volatility = bench_daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    # 日均超额收益
    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess_return = df["strategy_return"] - daily_rf

    # 夏普比率（年化）
    if excess_return.std() > 0:
        sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * excess_return.mean() / excess_return.std()
    else:
        sharpe = 0.0

    # 基准夏普
    bench_excess = df["benchmark_return"] - daily_rf
    if bench_excess.std() > 0:
        bench_sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * bench_excess.mean() / bench_excess.std()
    else:
        bench_sharpe = 0.0

    # 年化收益率
    n_days = len(df)
    ann_return = (final_nav ** (TRADING_DAYS_PER_YEAR / n_days) - 1) * 100

    # 交易统计 + 胜率 + 盈亏比 + 期望收益
    trades = df[df["signal"] != 0]
    n_trades = len(trades)
    buy_signals = (df["signal"] == 1).sum()
    sell_signals = (df["signal"] == -1).sum()

    win_count = 0
    loss_count = 0
    total_round_trips = 0
    win_profits = []   # 盈利交易的收益率
    loss_amounts = []   # 亏损交易的收益率（绝对值）
    buy_price = None

    for idx, row in df.iterrows():
        if row["signal"] == 1:
            buy_price = row["close"]
        elif row["signal"] == -1 and buy_price is not None:
            trade_return = (row["close"] - buy_price) / buy_price
            if trade_return > 0:
                win_count += 1
                win_profits.append(trade_return)
            else:
                loss_count += 1
                loss_amounts.append(abs(trade_return))
            total_round_trips += 1
            buy_price = None

    win_rate = (win_count / total_round_trips * 100) if total_round_trips > 0 else 0.0
    loss_rate = 100 - win_rate

    # 盈亏比 = 平均盈利 / 平均亏损
    avg_win = np.mean(win_profits) if win_profits else 0.0
    avg_loss = np.mean(loss_amounts) if loss_amounts else 0.0
    profit_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    # 期望收益 R = 胜率 × 平均盈利 - 败率 × 平均亏损（以R为单位，1R=平均亏损）
    if avg_loss > 0:
        expectancy_r = (win_count * avg_win - loss_count * avg_loss) / (total_round_trips * avg_loss)
    else:
        expectancy_r = 0.0

    metrics = {
        "cumulative_return": cumulative_return,
        "benchmark_return": benchmark_return,
        "ann_return": ann_return,
        "max_drawdown": max_drawdown,
        "benchmark_mdd": bench_mdd,
        "ann_volatility": ann_volatility,
        "benchmark_volatility": bench_volatility,
        "sharpe": sharpe,
        "benchmark_sharpe": bench_sharpe,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "expectancy_r": expectancy_r,
        "n_trades": n_trades,
        "buy_signals": int(buy_signals),
        "sell_signals": int(sell_signals),
        "total_round_trips": total_round_trips,
        "final_nav": final_nav,
    }

    return metrics


def run_strategy(csv_path=None, short_period=DEFAULT_SHORT, long_period=DEFAULT_LONG):
    """
    一键运行完整策略流程：
    加载数据 → 计算均线 → 生成信号 → 回测 → 计算指标
    返回 (df, metrics)
    """
    df = load_data(csv_path)
    df = calc_ma(df, short_period, long_period)
    df = generate_signals(df)
    df = backtest(df)
    metrics = calc_metrics(df)
    return df, metrics


def main():
    """独立运行：输出策略结果摘要"""
    print("=" * 60)
    print("双均线策略回测")
    print(f"标的：比亚迪 002594.SZ")
    print(f"均线参数：MA{DEFAULT_SHORT} / MA{DEFAULT_LONG}")
    print(f"初始资金：{INITIAL_CAPITAL:,.0f} 元")
    print(f"佣金费率：{COMMISSION_RATE*10000:.0f}%%  印花税：{STAMP_TAX_RATE*1000:.0f}‰")
    print("=" * 60)

    df, m = run_strategy()

    print(f"\n--- 策略绩效（7项核心指标）---")
    print(f"  1. 年化收益率:    {m['ann_return']:+.2f}%")
    print(f"  2. 最大回撤(MDD): {m['max_drawdown']:.2f}%")
    print(f"  3. 年化波动率:    {m['ann_volatility']:.2f}%")
    print(f"  4. 夏普比率:      {m['sharpe']:.3f}")
    print(f"  5. 胜率:          {m['win_rate']:.1f}%")
    print(f"  6. 盈亏比:        {m['profit_loss_ratio']:.2f}")
    print(f"  7. 期望收益(R):   {m['expectancy_r']:.3f}R")
    print(f"\n--- 对比基准(买入持有) ---")
    print(f"  基准累计回报:    {m['benchmark_return']:+.2f}%")
    print(f"  基准最大回撤:    {m['benchmark_mdd']:.2f}%")
    print(f"  基准年化波动率:  {m['benchmark_volatility']:.2f}%")
    print(f"  基准夏普比率:    {m['benchmark_sharpe']:.3f}")
    print(f"\n--- 交易统计 ---")
    print(f"  交易次数:        {m['n_trades']} (买{m['buy_signals']}/卖{m['sell_signals']})")
    print(f"  完整交易轮数:    {m['total_round_trips']}")
    print(f"  期末净值:        {m['final_nav']:.4f}")

    # 输出交易明细
    trades = df[df["signal"] != 0][["trade_date", "close", "ma_short", "ma_long", "signal"]]
    print(f"\n--- 交易明细 ---")
    for _, row in trades.iterrows():
        action = "买入" if row["signal"] == 1 else "卖出"
        date = str(row["trade_date"])
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date
        print(f"  {date} | {action:4s} | 价格 {row['close']:.2f} | MA{DEFAULT_SHORT}={row['ma_short']:.2f} MA{DEFAULT_LONG}={row['ma_long']:.2f}")


if __name__ == "__main__":
    main()
