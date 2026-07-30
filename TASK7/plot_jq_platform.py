# -*- coding: utf-8 -*-
"""
plot_jq_platform.py
生成 JoinQuant 平台部署相关的可视化图表
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 字体配置
for fp in [
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/simsun.ttc',
]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        break
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from joinquant_strategy import run_dual_ma, run_turtle, INITIAL_CAPITAL


def plot_jq_workflow():
    """
    图5: JoinQuant平台部署流程图 + 模拟交易净值
    展示从策略编写到实盘模拟的完整流程
    """
    df_ma, m_ma = run_dual_ma()
    df_tu, m_tu = run_turtle()

    fig = plt.figure(figsize=(14, 12))

    # ===== 子图1: JoinQuant 平台净值曲线（模拟实盘）=====
    ax1 = fig.add_subplot(2, 2, 1)
    dates = pd.to_datetime(df_ma['trade_date'], format='%Y%m%d')

    ax1.plot(dates, df_ma['strategy_nav'] * INITIAL_CAPITAL,
             color='#f85149', linewidth=1.5, label='双均线策略')
    ax1.plot(dates, df_tu['strategy_nav'] * INITIAL_CAPITAL,
             color='#a371f7', linewidth=1.5, label='海龟策略')
    ax1.plot(dates, df_ma['benchmark_nav'] * INITIAL_CAPITAL,
             color='#8b949e', linewidth=1.2, linestyle='--', label='买入持有')

    # 标注买卖信号
    buy_signals = df_ma[df_ma['signal'] == 1]
    sell_signals = df_ma[df_ma['signal'] == -1]
    if len(buy_signals) > 0:
        buy_dates = pd.to_datetime(buy_signals['trade_date'], format='%Y%m%d')
        buy_nav = buy_signals['strategy_nav'] * INITIAL_CAPITAL
        ax1.scatter(buy_dates, buy_nav, marker='^', color='#f85149', s=80, zorder=5, label='买入信号')
    if len(sell_signals) > 0:
        sell_dates = pd.to_datetime(sell_signals['trade_date'], format='%Y%m%d')
        sell_nav = sell_signals['strategy_nav'] * INITIAL_CAPITAL
        ax1.scatter(sell_dates, sell_nav, marker='v', color='#3fb950', s=80, zorder=5, label='卖出信号')

    ax1.axhline(y=INITIAL_CAPITAL, color='#d29922', linewidth=0.8, linestyle=':', alpha=0.7, label='初始资金')
    ax1.set_title('JoinQuant平台 - 策略模拟净值曲线', fontsize=13, fontweight='bold')
    ax1.set_ylabel('账户资产（元）')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=8)

    # ===== 子图2: 回测 vs 模拟交易对比 =====
    ax2 = fig.add_subplot(2, 2, 2)

    categories = ['累计收益\n(%)', '最大回撤\n(%)', '夏普比率', '年化波动\n(%)']
    ma_vals = [m_ma['cum_return'], m_ma['max_dd'], m_ma['sharpe']*100, m_ma['ann_vol']]
    tu_vals = [m_tu['cum_return'], m_tu['max_dd'], m_tu['sharpe']*100, m_tu['ann_vol']]
    bench_vals = [m_ma['benchmark_return'], m_ma['benchmark_mdd'], m_ma['benchmark_sharpe']*100, m_ma['benchmark_vol']]

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax2.bar(x - width, ma_vals, width, color='#f85149', alpha=0.85, label='双均线')
    bars2 = ax2.bar(x, tu_vals, width, color='#a371f7', alpha=0.85, label='海龟')
    bars3 = ax2.bar(x + width, bench_vals, width, color='#8b949e', alpha=0.6, label='基准')

    ax2.axhline(y=0, color='#30363d', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_title('策略绩效多维度对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=8)

    # ===== 子图3: 滚动风险指标 =====
    ax3 = fig.add_subplot(2, 2, 3)

    # 滚动60日夏普
    df = df_ma.copy()
    df['daily_ret'] = df['strategy_return']
    window = 60
    df['rolling_sharpe'] = df['daily_ret'].rolling(window).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
    )
    df['rolling_vol'] = df['daily_ret'].rolling(20).std() * np.sqrt(252) * 100

    ax3_twin = ax3.twinx()
    ax3.plot(dates[:len(df)], df['rolling_sharpe'].values,
             color='#3fb950', linewidth=1.2, label='滚动夏普比率(60日)')
    ax3.axhline(y=0, color='#30363d', linewidth=0.5)
    ax3.axhline(y=1, color='#f85149', linewidth=0.5, linestyle=':', alpha=0.7)
    ax3.fill_between(dates[:len(df)], df['rolling_sharpe'].values, 0,
                     where=df['rolling_sharpe'].values >= 0, alpha=0.15, color='#3fb950')
    ax3.fill_between(dates[:len(df)], df['rolling_sharpe'].values, 0,
                     where=df['rolling_sharpe'].values < 0, alpha=0.15, color='#f85149')

    ax3_twin.plot(dates[:len(df)], df['rolling_vol'].values,
                  color='#d29922', linewidth=1, alpha=0.7, label='年化波动率(20日)')

    ax3.set_title('滚动风险指标监控', fontsize=13, fontweight='bold')
    ax3.set_ylabel('夏普比率', color='#3fb950')
    ax3_twin.set_ylabel('年化波动率(%)', color='#d29922')
    ax3.tick_params(labelsize=8)
    ax3.grid(True, alpha=0.3)

    # 合并图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    # ===== 子图4: VaR/CVaR 风险分布 =====
    ax4 = fig.add_subplot(2, 2, 4)

    ma_returns = df_ma['strategy_return'].dropna() * 100
    tu_returns = df_tu['strategy_return'].dropna() * 100

    ax4.hist(ma_returns, bins=40, alpha=0.5, color='#f85149', label='双均线', edgecolor='#30363d')
    ax4.hist(tu_returns, bins=40, alpha=0.5, color='#a371f7', label='海龟', edgecolor='#30363d')

    # VaR线
    var_ma = np.percentile(ma_returns, 5)
    var_tu = np.percentile(tu_returns, 5)
    ax4.axvline(x=var_ma, color='#f85149', linewidth=2, linestyle='--', label=f'VaR95% MA: {var_ma:.2f}%')
    ax4.axvline(x=var_tu, color='#a371f7', linewidth=2, linestyle='--', label=f'VaR95% DC: {var_tu:.2f}%')

    ax4.set_title('日收益率分布与VaR', fontsize=13, fontweight='bold')
    ax4.set_xlabel('日收益率(%)')
    ax4.set_ylabel('频数')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=8)

    plt.suptitle('JoinQuant平台 - 实盘模拟部署与风险评估', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(BASE_DIR, 'jq_platform_deployment.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"图5已保存: {out_path}")
    return out_path


if __name__ == '__main__':
    plot_jq_workflow()
