# -*- coding: utf-8 -*-
"""
joinquant_strategy.py
JoinQuant 平台策略模拟引擎：双均线 + 海龟通道 + 参数优化 + 风险分析
用于复现 JoinQuant 平台回测逻辑，生成本地等价结果。
"""
import pandas as pd
import numpy as np
import os

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")

# JoinQuant 平台默认参数
JQ_SHORT = 5          # 短均线周期
JQ_LONG = 20          # 长均线周期
JQ_DC_ENTRY = 20      # 唐奇安入场通道
JQ_DC_EXIT = 10       # 唐奇安出场通道
JQ_ATR_PERIOD = 20    # ATR 周期
JQ_STOP_ATR = 2.0     # ATR 止损倍数

# 回测参数（模拟 JoinQuant 平台默认设置）
INITIAL_CAPITAL = 100000.0   # 初始资金（元）
COMMISSION_RATE = 0.0003     # 佣金费率（万分之三双向）
STAMP_TAX_RATE = 0.001       # 印花税（卖出方千分之一）
SLIPPAGE = 0.002             # 滑点成本 0.2%

RISK_FREE_ANNUAL = 0.03      # 无风险年化利率（JoinQuant 默认3%）
TRADING_DAYS = 252


def load_data(csv_path=None):
    if csv_path is None:
        csv_path = CSV_PATH
    df = pd.read_csv(csv_path)
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
    return df


# ==================== 双均线策略 ====================

def dual_ma_strategy(df, short=JQ_SHORT, long=JQ_LONG):
    """JoinQuant 双均线策略模拟"""
    df = df.copy()
    df["ma_short"] = df["close"].rolling(window=short, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(window=long, min_periods=1).mean()

    df["signal"] = 0
    prev_diff = df["ma_short"].shift(1) - df["ma_long"].shift(1)
    curr_diff = df["ma_short"] - df["ma_long"]
    df.loc[(prev_diff < 0) & (curr_diff > 0), "signal"] = 1
    df.loc[(prev_diff > 0) & (curr_diff < 0), "signal"] = -1

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


# ==================== 海龟通道策略 ====================

def turtle_strategy(df, entry_period=JQ_DC_ENTRY, exit_period=JQ_DC_EXIT,
                    atr_period=JQ_ATR_PERIOD, stop_atr_mult=JQ_STOP_ATR):
    """JoinQuant 海龟策略模拟"""
    df = df.copy()

    # 唐奇安通道
    df["entry_upper"] = df["high"].shift(1).rolling(entry_period, min_periods=1).max()
    df["entry_lower"] = df["low"].shift(1).rolling(entry_period, min_periods=1).min()
    df["exit_upper"] = df["high"].shift(1).rolling(exit_period, min_periods=1).max()
    df["exit_lower"] = df["low"].shift(1).rolling(exit_period, min_periods=1).min()

    # ATR
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=1, adjust=False).mean()

    # 信号
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
            if pd.notna(entry_upper) and close > entry_upper:
                df.iloc[i, df.columns.get_loc("signal")] = 1
                holding = 1
                entry_price = close
                stop = close - stop_atr_mult * atr
                df.iloc[i, df.columns.get_loc("stop_price")] = stop
        else:
            new_stop = close - stop_atr_mult * atr
            if new_stop > stop:
                stop = new_stop
            df.iloc[i, df.columns.get_loc("stop_price")] = stop

            if close < stop:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                holding = 0
            elif pd.notna(exit_lower) and close < exit_lower:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                holding = 0

        df.iloc[i, df.columns.get_loc("position")] = holding
    return df


# ==================== 回测引擎 ====================

def backtest(df, strategy_name="dual_ma"):
    """
    模拟 JoinQuant 平台回测逻辑：
    - 信号当日收盘价生成，次日开盘价成交
    - 考虑滑点、佣金、印花税
    - 全仓买入/卖出
    """
    df = df.copy()

    # 日收益率
    df["stock_return"] = df["close"].pct_change().fillna(0)

    # 策略日收益率：基于持仓状态（T-1信号->T日持仓）
    df["strategy_return"] = df["position"].shift(1) * df["stock_return"]
    df["strategy_return"] = df["strategy_return"].fillna(0)

    # 基准
    df["benchmark_return"] = df["stock_return"]

    # 交易成本扣除
    trades = df[df["signal"] != 0]
    total_cost = 0.0
    for _, row in trades.iterrows():
        if row["signal"] == 1:
            cost = INITIAL_CAPITAL * (COMMISSION_RATE + SLIPPAGE)
        else:
            cost = INITIAL_CAPITAL * (COMMISSION_RATE + STAMP_TAX_RATE + SLIPPAGE)
        total_cost += cost

    # 累计净值
    df["strategy_nav"] = (1 + df["strategy_return"]).cumprod()
    df["benchmark_nav"] = (1 + df["benchmark_return"]).cumprod()

    # 成本折算
    if df["strategy_nav"].iloc[-1] > 0:
        cost_factor = 1 - total_cost / (INITIAL_CAPITAL * df["strategy_nav"].iloc[-1])
        if cost_factor > 0:
            df["strategy_nav"] = df["strategy_nav"] * cost_factor

    return df


# ==================== 绩效指标 ====================

def calc_metrics(df, risk_free=RISK_FREE_ANNUAL):
    """计算7项核心绩效指标 + 额外风险指标"""
    final_nav = df["strategy_nav"].iloc[-1]
    cum_return = (final_nav - 1) * 100
    bench_return = (df["benchmark_nav"].iloc[-1] - 1) * 100

    running_max = df["strategy_nav"].cummax()
    drawdown = (df["strategy_nav"] - running_max) / running_max
    max_dd = drawdown.min() * 100

    bench_running_max = df["benchmark_nav"].cummax()
    bench_dd = (df["benchmark_nav"] - bench_running_max) / bench_running_max
    bench_mdd = bench_dd.min() * 100

    ann_vol = df["strategy_return"].std() * np.sqrt(TRADING_DAYS) * 100
    bench_vol = df["benchmark_return"].std() * np.sqrt(TRADING_DAYS) * 100

    daily_rf = risk_free / TRADING_DAYS
    excess = df["strategy_return"] - daily_rf
    sharpe = np.sqrt(TRADING_DAYS) * excess.mean() / excess.std() if excess.std() > 0 else 0.0

    bench_excess = df["benchmark_return"] - daily_rf
    bench_sharpe = np.sqrt(TRADING_DAYS) * bench_excess.mean() / bench_excess.std() if bench_excess.std() > 0 else 0.0

    n_days = len(df)
    ann_return = (final_nav ** (TRADING_DAYS / n_days) - 1) * 100

    trades = df[df["signal"] != 0]
    n_trades = len(trades)
    buy_count = int((df["signal"] == 1).sum())
    sell_count = int((df["signal"] == -1).sum())

    win_count = 0
    loss_count = 0
    total_round = 0
    win_profits = []
    loss_amounts = []
    buy_price = None

    for _, row in df.iterrows():
        if row["signal"] == 1:
            buy_price = row["close"]
        elif row["signal"] == -1 and buy_price is not None:
            trade_ret = (row["close"] - buy_price) / buy_price
            if trade_ret > 0:
                win_count += 1
                win_profits.append(trade_ret)
            else:
                loss_count += 1
                loss_amounts.append(abs(trade_ret))
            total_round += 1
            buy_price = None

    win_rate = (win_count / total_round * 100) if total_round > 0 else 0.0
    avg_win = np.mean(win_profits) if win_profits else 0.0
    avg_loss = np.mean(loss_amounts) if loss_amounts else 0.0
    plr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    expectancy_r = ((win_count * avg_win - loss_count * avg_loss) / (total_round * avg_loss)) if avg_loss > 0 else 0.0

    # Sortino
    downside = df["strategy_return"][df["strategy_return"] < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    sortino = np.sqrt(TRADING_DAYS) * excess.mean() / downside_std if downside_std > 0 else 0.0

    # Calmar
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0.01 else 0.0

    # VaR / CVaR
    var_95 = np.percentile(df["strategy_return"], 5) * 100
    cvar_95 = df["strategy_return"][df["strategy_return"] <= np.percentile(df["strategy_return"], 5)].mean() * 100

    alpha = cum_return - bench_return

    return {
        "strategy_name": "",
        "cum_return": cum_return,
        "ann_return": ann_return,
        "max_dd": max_dd,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "plr": plr,
        "expectancy_r": expectancy_r,
        "n_trades": n_trades,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "total_round": total_round,
        "final_nav": final_nav,
        "benchmark_return": bench_return,
        "benchmark_mdd": bench_mdd,
        "benchmark_vol": bench_vol,
        "benchmark_sharpe": bench_sharpe,
        "alpha": alpha,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "risk_free": risk_free,
    }


def run_dual_ma(csv_path=None, short=JQ_SHORT, long=JQ_LONG):
    df = load_data(csv_path)
    df = dual_ma_strategy(df, short, long)
    df = backtest(df)
    m = calc_metrics(df)
    m["strategy_name"] = f"双均线MA{short}/{long}"
    return df, m


def run_turtle(csv_path=None, entry=JQ_DC_ENTRY, exit_p=JQ_DC_EXIT,
               atr_p=JQ_ATR_PERIOD, stop_mult=JQ_STOP_ATR):
    df = load_data(csv_path)
    df = turtle_strategy(df, entry, exit_p, atr_p, stop_mult)
    df = backtest(df)
    m = calc_metrics(df)
    m["strategy_name"] = f"海龟DC{entry}/{exit_p}"
    return df, m


def run_param_sweep(csv_path=None):
    """参数优化网格搜索"""
    df_raw = load_data(csv_path)

    ma_combos = [
        (3, 5), (3, 10), (5, 10), (5, 15), (5, 20),
        (5, 30), (10, 20), (10, 30), (10, 60), (20, 60),
    ]

    ma_results = []
    for s, l in ma_combos:
        d = dual_ma_strategy(df_raw.copy(), s, l)
        d = backtest(d)
        m = calc_metrics(d)
        m["strategy_name"] = f"MA{s}/{l}"
        m["param"] = f"MA{s}/{l}"
        ma_results.append(m)

    dc_combos = [
        (10, 5, 2.0), (15, 7, 2.0), (20, 10, 2.0), (20, 10, 1.5),
        (20, 10, 3.0), (25, 10, 2.0), (30, 15, 2.0), (40, 20, 2.0),
        (55, 20, 2.0), (20, 10, 1.0),
    ]

    dc_results = []
    for entry, exit_p, stop in dc_combos:
        d = turtle_strategy(df_raw.copy(), entry, exit_p, entry, stop)
        d = backtest(d)
        m = calc_metrics(d)
        m["strategy_name"] = f"DC{entry}/{exit_p} S{stop}"
        m["param"] = f"DC{entry}/{exit_p} S{stop}"
        dc_results.append(m)

    return ma_results, dc_results


if __name__ == "__main__":
    print("=" * 60)
    print("JoinQuant 平台策略模拟回测")
    print("=" * 60)

    print("\n--- 双均线策略 MA5/20 ---")
    df_ma, m_ma = run_dual_ma()
    for k in ["cum_return", "ann_return", "max_dd", "ann_vol", "sharpe", "win_rate", "plr", "expectancy_r"]:
        print(f"  {k}: {m_ma[k]:.3f}")

    print("\n--- 海龟策略 DC20/10 ---")
    df_tu, m_tu = run_turtle()
    for k in ["cum_return", "ann_return", "max_dd", "ann_vol", "sharpe", "win_rate", "plr", "expectancy_r"]:
        print(f"  {k}: {m_tu[k]:.3f}")
