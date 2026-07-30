# -*- coding: utf-8 -*-
"""
JoinQuant平台 - 参数优化策略
用于在JoinQuant平台上进行参数网格搜索

使用方法：
1. 在JoinQuant"研究"模块中新建Notebook
2. 复制以下代码运行
3. 生成参数优化对比表和图表
"""

import jqdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jqdata import *

# ==================== 参数网格搜索（在研究模块运行） ====================

def run_param_optimization():
    """
    在JoinQuant研究模块中运行此函数
    对双均线和海龟策略进行参数网格搜索
    """

    security = '002594.XSHE'
    start_date = '2025-07-07'
    end_date = '2026-07-07'

    # 获取数据
    df = get_price(
        security, start_date=start_date, end_date=end_date,
        frequency='daily', fields=['open', 'high', 'low', 'close', 'volume'],
        fq='pre'  # 前复权
    )
    print("数据获取完成: %d条" % len(df))

    # ===== 双均线参数组合 =====
    ma_params = [
        (3, 5), (3, 10), (5, 10), (5, 15), (5, 20),
        (5, 30), (10, 20), (10, 30), (10, 60), (20, 60),
    ]

    ma_results = []
    for short, long in ma_params:
        result = backtest_dual_ma(df, short, long)
        result['params'] = 'MA%d/%d' % (short, long)
        ma_results.append(result)

    # ===== 海龟参数组合 =====
    dc_params = [
        (10, 5, 2.0), (15, 7, 2.0), (20, 10, 2.0), (20, 10, 1.5),
        (20, 10, 3.0), (25, 10, 2.0), (30, 15, 2.0), (40, 20, 2.0),
        (55, 20, 2.0), (20, 10, 1.0),
    ]

    dc_results = []
    for entry, exit_p, stop in dc_params:
        result = backtest_turtle(df, entry, exit_p, stop)
        result['params'] = 'DC%d/%d S%.1f' % (entry, exit_p, stop)
        dc_results.append(result)

    # ===== 输出结果 =====
    print("\n" + "=" * 80)
    print("双均线策略参数优化结果")
    print("=" * 80)
    ma_df = pd.DataFrame(ma_results)
    ma_df = ma_df.sort_values('sharpe', ascending=False)
    print(ma_df[['params', 'cum_return', 'ann_return', 'max_dd', 'sharpe',
                  'win_rate', 'n_trades']].to_string(index=False))

    print("\n" + "=" * 80)
    print("海龟策略参数优化结果")
    print("=" * 80)
    dc_df = pd.DataFrame(dc_results)
    dc_df = dc_df.sort_values('sharpe', ascending=False)
    print(dc_df[['params', 'cum_return', 'ann_return', 'max_dd', 'sharpe',
                  'win_rate', 'n_trades']].to_string(index=False))

    # ===== 可视化 =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 双均线 - 夏普比率
    ax = axes[0, 0]
    ma_sorted = ma_df.sort_values('sharpe', ascending=True)
    colors = ['#3fb950' if s > 0 else '#f85149' for s in ma_sorted['sharpe']]
    ax.barh(ma_sorted['params'], ma_sorted['sharpe'], color=colors)
    ax.set_title('双均线策略 - 夏普比率', fontsize=13)
    ax.axvline(x=0, color='white', linewidth=0.8)

    # 双均线 - 累计收益 vs 最大回撤
    ax = axes[0, 1]
    ax.scatter(ma_df['max_dd'].abs(), ma_df['cum_return'],
               c=ma_df['sharpe'], cmap='RdYlGn', s=100, edgecolors='white')
    for _, row in ma_df.iterrows():
        ax.annotate(row['params'], (row['max_dd'].abs(), row['cum_return']),
                    fontsize=7, alpha=0.8)
    ax.set_xlabel('最大回撤(%)')
    ax.set_ylabel('累计收益(%)')
    ax.set_title('双均线策略 - 风险收益分布', fontsize=13)

    # 海龟 - 夏普比率
    ax = axes[1, 0]
    dc_sorted = dc_df.sort_values('sharpe', ascending=True)
    colors = ['#3fb950' if s > 0 else '#f85149' for s in dc_sorted['sharpe']]
    ax.barh(dc_sorted['params'], dc_sorted['sharpe'], color=colors)
    ax.set_title('海龟策略 - 夏普比率', fontsize=13)
    ax.axvline(x=0, color='white', linewidth=0.8)

    # 海龟 - 累计收益 vs 最大回撤
    ax = axes[1, 1]
    ax.scatter(dc_df['max_dd'].abs(), dc_df['cum_return'],
               c=dc_df['sharpe'], cmap='RdYlGn', s=100, edgecolors='white')
    for _, row in dc_df.iterrows():
        ax.annotate(row['params'], (row['max_dd'].abs(), row['cum_return']),
                    fontsize=7, alpha=0.8)
    ax.set_xlabel('最大回撤(%)')
    ax.set_ylabel('累计收益(%)')
    ax.set_title('海龟策略 - 风险收益分布', fontsize=13)

    plt.tight_layout()
    plt.savefig('param_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()

    return ma_df, dc_df


def backtest_dual_ma(df, short, long_p):
    """双均线策略本地回测"""
    close = df['close'].values
    n = len(close)

    ma_s = pd.Series(close).rolling(short, min_periods=1).mean().values
    ma_l = pd.Series(close).rolling(long_p, min_periods=1).mean().values

    position = np.zeros(n)
    signal = np.zeros(n)
    holding = 0

    for i in range(1, n):
        if ma_s[i-1] <= ma_l[i-1] and ma_s[i] > ma_l[i]:
            holding = 1
            signal[i] = 1
        elif ma_s[i-1] >= ma_l[i-1] and ma_s[i] < ma_l[i]:
            holding = 0
            signal[i] = -1
        position[i] = holding

    returns = np.diff(close) / close[:-1]
    strategy_returns = position[:-1] * returns
    strategy_returns = np.insert(strategy_returns, 0, 0)

    # 交易成本
    trades = signal != 0
    for i in np.where(trades)[0]:
        if signal[i] == 1:
            strategy_returns[i] -= 0.0003 + 0.002  # 佣金+滑点
        else:
            strategy_returns[i] -= 0.0003 + 0.001 + 0.002  # 佣金+印花税+滑点

    nav = np.cumprod(1 + strategy_returns)
    bench_nav = np.cumprod(1 + returns)

    cum_return = (nav[-1] - 1) * 100
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min() * 100

    ann_return = (nav[-1] ** (252.0 / n) - 1) * 100
    ann_vol = np.std(strategy_returns) * np.sqrt(252) * 100

    rf_daily = 0.03 / 252
    excess = strategy_returns - rf_daily
    sharpe = np.sqrt(252) * np.mean(excess) / np.std(excess) if np.std(excess) > 0 else 0

    # 胜率
    buy_idx = np.where(signal == 1)[0]
    sell_idx = np.where(signal == -1)[0]
    n_trades = len(buy_idx)
    wins = 0
    for j in range(min(len(buy_idx), len(sell_idx))):
        if close[sell_idx[j]] > close[buy_idx[j]]:
            wins += 1
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    return {
        'cum_return': cum_return,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'final_nav': nav[-1],
    }


def backtest_turtle(df, entry_p, exit_p, stop_mult):
    """海龟策略本地回测"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)

    # 唐奇安通道
    entry_upper = pd.Series(high).shift(1).rolling(entry_p, min_periods=1).max().values
    exit_lower = pd.Series(low).shift(1).rolling(exit_p, min_periods=1).min().values

    # ATR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(alpha=1.0/20, adjust=False).mean().values

    position = np.zeros(n)
    signal = np.zeros(n)
    holding = 0
    stop_price = 0.0

    for i in range(n):
        if holding == 0:
            if not np.isnan(entry_upper[i]) and close[i] > entry_upper[i]:
                holding = 1
                signal[i] = 1
                stop_price = close[i] - stop_mult * atr[i]
        else:
            new_stop = close[i] - stop_mult * atr[i]
            if new_stop > stop_price:
                stop_price = new_stop

            if close[i] < stop_price:
                holding = 0
                signal[i] = -1
            elif not np.isnan(exit_lower[i]) and close[i] < exit_lower[i]:
                holding = 0
                signal[i] = -1

        position[i] = holding

    returns = np.diff(close) / close[:-1]
    strategy_returns = position[:-1] * returns
    strategy_returns = np.insert(strategy_returns, 0, 0)

    trades = signal != 0
    for i in np.where(trades)[0]:
        if signal[i] == 1:
            strategy_returns[i] -= 0.0003 + 0.002
        else:
            strategy_returns[i] -= 0.0003 + 0.001 + 0.002

    nav = np.cumprod(1 + strategy_returns)

    cum_return = (nav[-1] - 1) * 100
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min() * 100

    ann_return = (nav[-1] ** (252.0 / n) - 1) * 100
    ann_vol = np.std(strategy_returns) * np.sqrt(252) * 100

    rf_daily = 0.03 / 252
    excess = strategy_returns - rf_daily
    sharpe = np.sqrt(252) * np.mean(excess) / np.std(excess) if np.std(excess) > 0 else 0

    buy_idx = np.where(signal == 1)[0]
    sell_idx = np.where(signal == -1)[0]
    n_trades = len(buy_idx)
    wins = 0
    for j in range(min(len(buy_idx), len(sell_idx))):
        if close[sell_idx[j]] > close[buy_idx[j]]:
            wins += 1
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    return {
        'cum_return': cum_return,
        'ann_return': ann_return,
        'max_dd': max_dd,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'final_nav': nav[-1],
    }


# ==================== 滚动风险分析（在研究模块运行） ====================

def rolling_risk_analysis():
    """
    在JoinQuant研究模块中运行
    计算滚动夏普比率、滚动最大回撤、VaR/CVaR
    """
    security = '002594.XSHE'
    start_date = '2025-07-07'
    end_date = '2026-07-07'

    df = get_price(security, start_date=start_date, end_date=end_date,
                   frequency='daily', fields=['close'], fq='pre')

    close = df['close']
    daily_returns = close.pct_change().dropna()

    # 滚动夏普比率（60日窗口）
    rolling_sharpe = daily_returns.rolling(60).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
    )

    # 滚动波动率（20日窗口）
    rolling_vol = daily_returns.rolling(20).std() * np.sqrt(252) * 100

    # 滚动最大回撤（60日窗口）
    def rolling_max_dd(prices_window):
        running_max = prices_window.cummax()
        dd = (prices_window - running_max) / running_max
        return dd.min() * 100

    rolling_mdd = close.rolling(60).apply(rolling_max_dd)

    # VaR/CVaR
    var_95 = daily_returns.quantile(0.05) * 100
    var_99 = daily_returns.quantile(0.01) * 100
    cvar_95 = daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean() * 100
    cvar_99 = daily_returns[daily_returns <= daily_returns.quantile(0.01)].mean() * 100

    print("=" * 60)
    print("风险分析报告 - %s" % security)
    print("=" * 60)
    print("数据区间: %s ~ %s" % (start_date, end_date))
    print("交易日数: %d" % len(daily_returns))
    print()
    print("--- VaR/CVaR ---")
    print("日VaR(95%%): %.2f%%" % var_95)
    print("日VaR(99%%): %.2f%%" % var_99)
    print("日CVaR(95%%): %.2f%%" % cvar_95)
    print("日CVaR(99%%): %.2f%%" % cvar_99)
    print()
    print("--- 年化风险指标 ---")
    ann_vol = daily_returns.std() * np.sqrt(252) * 100
    ann_return = ((close.iloc[-1] / close.iloc[0]) ** (252.0 / len(daily_returns)) - 1) * 100
    rf = 3.0
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0
    print("年化收益率: %.2f%%" % ann_return)
    print("年化波动率: %.2f%%" % ann_vol)
    print("夏普比率: %.2f" % sharpe)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(close.index, close.values, color='#58a6ff', linewidth=1.2)
    ax.set_title('价格走势', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(rolling_vol.index, rolling_vol.values, color='#d29922', linewidth=1)
    ax.set_title('滚动年化波动率 (20日窗口)', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color='#3fb950', linewidth=1)
    ax.axhline(y=0, color='white', linewidth=0.8, linestyle='--')
    ax.axhline(y=1, color='#f85149', linewidth=0.8, linestyle=':', label='夏普=1')
    ax.set_title('滚动夏普比率 (60日窗口)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(rolling_mdd.index, rolling_mdd.values, color='#f85149', linewidth=1)
    ax.fill_between(rolling_mdd.index, rolling_mdd.values, 0, alpha=0.3, color='#f85149')
    ax.set_title('滚动最大回撤 (60日窗口)', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rolling_risk.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {
        'var_95': var_95, 'var_99': var_99,
        'cvar_95': cvar_95, 'cvar_99': cvar_99,
        'ann_vol': ann_vol, 'ann_return': ann_return,
        'sharpe': sharpe,
    }


# 运行（在研究模块中取消注释）
# ma_df, dc_df = run_param_optimization()
# risk = rolling_risk_analysis()
