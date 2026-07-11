# -*- coding: utf-8 -*-
"""
strategy.py
海龟策略引擎：唐奇安通道 → ATR → 突破信号 → ATR止损 → 模拟回测 → 绩效指标
纯 pandas 实现，无外部金融库依赖。
"""
import pandas as pd
import numpy as np
import os

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")

# 默认通道周期（唐奇安通道）
DEFAULT_ENTRY_PERIOD = 20      # 入场通道周期（20日高低点）
DEFAULT_EXIT_PERIOD = 10       # 出场通道周期（10日高低点）
DEFAULT_ATR_PERIOD = 20        # ATR计算周期

# ATR止损倍数（海龟法则经典值：2倍ATR）
DEFAULT_STOP_ATR_MULT = 2.0

# 回测参数
INITIAL_CAPITAL = 100000.0     # 初始资金（元）
COMMISSION_RATE = 0.0003       # 佣金费率（万分之三）
STAMP_TAX_RATE = 0.001         # 印花税（卖出方千分之一）
SLIPPAGE = 0.0                 # 滑点（暂设为0）

# 无风险年化利率（用于计算夏普比率，默认2%）
RISK_FREE_ANNUAL = 0.02
TRADING_DAYS_PER_YEAR = 252


def load_data(csv_path=None):
    """加载股价数据，返回按日期升序排列的 DataFrame"""
    if csv_path is None:
        csv_path = CSV_PATH
    df = pd.read_csv(csv_path)
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
    return df


def calc_donchian(df, entry_period=DEFAULT_ENTRY_PERIOD, exit_period=DEFAULT_EXIT_PERIOD):
    """
    计算唐奇安通道（Donchian Channel）：
    - entry_upper: 入场通道上轨 = 过去N日最高价（不含当日）
    - entry_lower: 入场通道下轨 = 过去N日最低价（不含当日）
    - exit_upper:  出场通道上轨 = 过去M日最高价（不含当日）
    - exit_lower:  出场通道下轨 = 过去M日最低价（不含当日）
    
    注意：海龟法则使用"前N日"的最高/最低价，不包含当日，
    即用 shift(1) 后的 rolling，避免未来数据泄露。
    """
    df = df.copy()
    # 入场通道（用前一日开始的rolling，不含当日）
    df["entry_upper"] = df["high"].shift(1).rolling(window=entry_period, min_periods=1).max()
    df["entry_lower"] = df["low"].shift(1).rolling(window=entry_period, min_periods=1).min()
    # 出场通道
    df["exit_upper"] = df["high"].shift(1).rolling(window=exit_period, min_periods=1).max()
    df["exit_lower"] = df["low"].shift(1).rolling(window=exit_period, min_periods=1).min()
    # 通道中轨（仅用于可视化）
    df["channel_mid"] = (df["entry_upper"] + df["entry_lower"]) / 2
    return df


def calc_atr(df, atr_period=DEFAULT_ATR_PERIOD):
    """
    计算平均真实波幅（Average True Range, ATR）：
    TR = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )
    ATR = TR 的 N日指数移动平均（Wilder平滑法）
    """
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder平滑法（类似EMA但alpha=1/N）
    df["atr"] = df["tr"].ewm(alpha=1.0 / atr_period, min_periods=1, adjust=False).mean()
    return df


def generate_signals(df, stop_atr_mult=DEFAULT_STOP_ATR_MULT):
    """
    生成海龟策略交易信号：
    - 买入信号(+1)：当日收盘价突破入场通道上轨（创N日新高）
    - 卖出信号(-1)：当日收盘价跌破出场通道下轨（创M日新低）
    - ATR止损：持仓期间若价格跌破 (买入价 - stop_atr_mult × ATR)，触发止损卖出
    
    返回新增列：signal(1=买入,-1=卖出,0=持有) position(1=持仓,0=空仓) stop_price(止损价)
    """
    df = df.copy()
    df["signal"] = 0
    df["position"] = 0
    df["stop_price"] = np.nan

    holding = 0
    entry_price = 0.0
    stop = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        close = row["close"]
        entry_upper = row["entry_upper"]
        exit_lower = row["exit_lower"]
        atr = row["atr"] if pd.notna(row["atr"]) else 0.0

        if holding == 0:
            # 空仓：检查是否突破上轨
            if pd.notna(entry_upper) and close > entry_upper:
                df.iloc[i, df.columns.get_loc("signal")] = 1
                holding = 1
                entry_price = close
                stop = close - stop_atr_mult * atr
                df.iloc[i, df.columns.get_loc("stop_price")] = stop
        else:
            # 持仓：检查止损或出场信号
            # 更新止损价（跟踪止损：仅上移不下移）
            new_stop = close - stop_atr_mult * atr
            if new_stop > stop:
                stop = new_stop
            df.iloc[i, df.columns.get_loc("stop_price")] = stop

            # 止损触发
            if close < stop:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                holding = 0
                entry_price = 0.0
                stop = 0.0
            # 出场信号：跌破出场通道下轨
            elif pd.notna(exit_lower) and close < exit_lower:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                holding = 0
                entry_price = 0.0
                stop = 0.0

        df.iloc[i, df.columns.get_loc("position")] = holding

    return df


def backtest(df, initial_capital=INITIAL_CAPITAL,
             commission_rate=COMMISSION_RATE, stamp_tax_rate=STAMP_TAX_RATE):
    """
    模拟回测：
    - 信号当日收盘价成交
    - 全仓买入/全仓卖出
    - 扣除佣金和印花税
    
    返回 df 新增列：strategy_return, benchmark_return, strategy_nav, benchmark_nav
    """
    df = df.copy()

    df["trade_price"] = df["close"]
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

    cost_factor = 1 - total_cost / (INITIAL_CAPITAL * df["strategy_nav"].iloc[-1])
    if cost_factor > 0:
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
    final_nav = df["strategy_nav"].iloc[-1]
    cumulative_return = (final_nav - 1) * 100

    benchmark_final = df["benchmark_nav"].iloc[-1]
    benchmark_return = (benchmark_final - 1) * 100

    running_max = df["strategy_nav"].cummax()
    drawdown = (df["strategy_nav"] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    bench_running_max = df["benchmark_nav"].cummax()
    bench_drawdown = (df["benchmark_nav"] - bench_running_max) / bench_running_max
    bench_mdd = bench_drawdown.min() * 100

    daily_std = df["strategy_return"].std()
    ann_volatility = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    bench_daily_std = df["benchmark_return"].std()
    bench_volatility = bench_daily_std * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess_return = df["strategy_return"] - daily_rf

    if excess_return.std() > 0:
        sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * excess_return.mean() / excess_return.std()
    else:
        sharpe = 0.0

    bench_excess = df["benchmark_return"] - daily_rf
    if bench_excess.std() > 0:
        bench_sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * bench_excess.mean() / bench_excess.std()
    else:
        bench_sharpe = 0.0

    n_days = len(df)
    ann_return = (final_nav ** (TRADING_DAYS_PER_YEAR / n_days) - 1) * 100

    trades = df[df["signal"] != 0]
    n_trades = len(trades)
    buy_signals = (df["signal"] == 1).sum()
    sell_signals = (df["signal"] == -1).sum()

    win_count = 0
    loss_count = 0
    total_round_trips = 0
    win_profits = []
    loss_amounts = []
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

    avg_win = np.mean(win_profits) if win_profits else 0.0
    avg_loss = np.mean(loss_amounts) if loss_amounts else 0.0
    profit_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

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


def run_strategy(csv_path=None, entry_period=DEFAULT_ENTRY_PERIOD,
                 exit_period=DEFAULT_EXIT_PERIOD, atr_period=DEFAULT_ATR_PERIOD,
                 stop_atr_mult=DEFAULT_STOP_ATR_MULT):
    """
    一键运行完整海龟策略流程：
    加载数据 → 计算通道 → 计算ATR → 生成信号 → 回测 → 计算指标
    返回 (df, metrics)
    """
    df = load_data(csv_path)
    df = calc_donchian(df, entry_period, exit_period)
    df = calc_atr(df, atr_period)
    df = generate_signals(df, stop_atr_mult)
    df = backtest(df)
    metrics = calc_metrics(df)
    return df, metrics


def main():
    """独立运行：输出策略结果摘要"""
    print("=" * 60)
    print("海龟策略（唐奇安通道突破 + ATR止损）回测")
    print(f"标的：比亚迪 002594.SZ")
    print(f"入场通道: {DEFAULT_ENTRY_PERIOD}日  出场通道: {DEFAULT_EXIT_PERIOD}日")
    print(f"ATR周期: {DEFAULT_ATR_PERIOD}日  止损倍数: {DEFAULT_STOP_ATR_MULT}×ATR")
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
    trades = df[df["signal"] != 0][["trade_date", "close", "entry_upper", "exit_lower", "atr", "signal", "stop_price"]]
    print(f"\n--- 交易明细 ---")
    for _, row in trades.iterrows():
        action = "买入" if row["signal"] == 1 else "卖出"
        date = str(row["trade_date"])
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date
        stop_str = f"止损={row['stop_price']:.2f}" if pd.notna(row["stop_price"]) and row["signal"] == 1 else ""
        print(f"  {date} | {action:4s} | 价格 {row['close']:.2f} | ATR={row['atr']:.2f} {stop_str}")


if __name__ == "__main__":
    main()
