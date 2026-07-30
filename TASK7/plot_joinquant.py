# -*- coding: utf-8 -*-
"""
plot_joinquant.py
JoinQuant 平台策略回���可视化：
  图1: 双均线策略信号图（买卖标记 + 持仓区间）
  图2: 双均线 vs 海龟 vs 基准 净值曲线 + 回撤
  图3: 参数优化网格搜索对比（双均线 + 海龟）
  图4: 风险暴露分析（VaR/CVaR/回撤分布/日收益分布）
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import platform

# ============ 跨平台中文字体 ============
def get_zh_font():
    system = platform.system()
    if system == "Windows":
        return {"fontname": "Microsoft YaHei"}
    elif system == "Darwin":
        return {"fontname": "Arial Unicode MS"}
    else:
        return {"fontname": "Noto Sans CJK SC"}

ZH = get_zh_font()
plt.rcParams["axes.unicode_minus"] = False

ZH_FONT_PATH = r"C:\Windows\Fonts\msyh.ttc" if platform.system() == "Windows" else None
ZH_FP = font_manager.FontProperties(fname=ZH_FONT_PATH, size=9) if ZH_FONT_PATH else None
ZH_FP_S = font_manager.FontProperties(fname=ZH_FONT_PATH, size=8) if ZH_FONT_PATH else None
ZH_FP_T = font_manager.FontProperties(fname=ZH_FONT_PATH, size=13) if ZH_FONT_PATH else None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from joinquant_strategy import (
    run_dual_ma, run_turtle, run_param_sweep, calc_metrics,
    JQ_SHORT, JQ_LONG, JQ_DC_ENTRY, JQ_DC_EXIT, JQ_ATR_PERIOD, JQ_STOP_ATR,
    INITIAL_CAPITAL
)

# 中国股市配色
C_PRICE = "#2196F3"
C_MA_S = "#FF9800"
C_MA_L = "#9C27B0"
C_BUY = "#E53935"
C_SELL = "#43A047"
C_STRATEGY_MA = "#E53935"
C_STRATEGY_TU = "#7B1FA2"
C_BENCH = "#78909C"


def fmt_dates(s):
    return pd.to_datetime(s.astype(str).str.replace("-", ""), format="%Y%m%d")


def plot_signals(df_ma, short, long_p):
    """图1: 双均线策略信号图"""
    dates = fmt_dates(df_ma["trade_date"])
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, df_ma["close"], color=C_PRICE, linewidth=1.2, label="收盘价", zorder=2)
    ax.plot(dates, df_ma["ma_short"], color=C_MA_S, linewidth=1.0, label=f"MA{short}", zorder=3)
    ax.plot(dates, df_ma["ma_long"], color=C_MA_L, linewidth=1.0, label=f"MA{long_p}", zorder=3)

    buys = df_ma[df_ma["signal"] == 1]
    sells = df_ma[df_ma["signal"] == -1]

    if len(buys) > 0:
        bd = fmt_dates(buys["trade_date"])
        ax.scatter(bd, buys["close"], color=C_BUY, marker="^", s=100, zorder=5, label="买入(金叉)")
    if len(sells) > 0:
        sd = fmt_dates(sells["trade_date"])
        ax.scatter(sd, sells["close"], color=C_SELL, marker="v", s=100, zorder=5, label="卖出(死叉)")

    # 持仓区间背景
    holding_ranges = []
    start = None
    for i, row in df_ma.iterrows():
        if row["position"] == 1 and start is None:
            start = i
        elif row["position"] == 0 and start is not None:
            holding_ranges.append((start, i - 1))
            start = None
    if start is not None:
        holding_ranges.append((start, len(df_ma) - 1))
    for s, e in holding_ranges:
        ax.axvspan(dates.iloc[s], dates.iloc[e], alpha=0.08, color=C_BUY, zorder=1)

    ax.set_title(f"JoinQuant 双均线策略信号图 — MA{short}/{long_p}", fontsize=13, fontweight="bold", **ZH)
    ax.set_xlabel("日期", **ZH)
    ax.set_ylabel("价格(元)", **ZH)
    ax.legend(prop=ZH_FP, loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "jq_strategy_signals.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图1] {out}")
    return out


def plot_nav_compare(df_ma, df_tu):
    """图2: 双均线 vs 海龟 vs 基准 净值 + 回撤"""
    dates = fmt_dates(df_ma["trade_date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # 上图：净值
    ax1.plot(dates, df_ma["strategy_nav"], color=C_STRATEGY_MA, linewidth=1.5, label="双均线策略净值")
    ax1.plot(dates, df_tu["strategy_nav"], color=C_STRATEGY_TU, linewidth=1.5, label="海龟策略净值")
    ax1.plot(dates, df_ma["benchmark_nav"], color=C_BENCH, linewidth=1.5, linestyle="--", label="基准(买入持有)")
    ax1.axhline(y=1.0, color="#666", linestyle=":", linewidth=0.8)
    ax1.set_title("策略净值对比（双均线 vs 海龟 vs 买入持有）", fontsize=13, fontweight="bold", **ZH)
    ax1.set_ylabel("累计净值", **ZH)
    ax1.legend(prop=ZH_FP, loc="best", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 下图：回撤
    def calc_dd(nav):
        rm = nav.cummax()
        return (nav - rm) / rm * 100

    dd_ma = calc_dd(df_ma["strategy_nav"])
    dd_tu = calc_dd(df_tu["strategy_nav"])
    dd_bench = calc_dd(df_ma["benchmark_nav"])

    ax2.fill_between(dates, dd_ma, 0, color=C_STRATEGY_MA, alpha=0.3, label="双均线回撤")
    ax2.fill_between(dates, dd_tu, 0, color=C_STRATEGY_TU, alpha=0.2, label="海龟回撤")
    ax2.plot(dates, dd_bench, color=C_BENCH, linewidth=0.8, linestyle="--", label="基准回撤")
    ax2.set_ylabel("回撤(%)", **ZH)
    ax2.set_xlabel("日期", **ZH)
    ax2.legend(prop=ZH_FP_S, loc="lower left", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "jq_nav_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图2] {out}")
    return out


def plot_param_optimization(ma_results, dc_results):
    """图3: 参数优化网格搜索对比"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    ma_labels = [m["param"] for m in ma_results]
    dc_labels = [m["param"] for m in dc_results]
    all_labels = ma_labels + dc_labels
    all_types = ["双均线"] * len(ma_labels) + ["海龟"] * len(dc_labels)

    def pos_neg_colors(values):
        return ["#E53935" if v >= 0 else "#43A047" for v in values]

    # 1. 累计回报
    ax = axes[0][0]
    vals = [m["cum_return"] for m in ma_results] + [m["cum_return"] for m in dc_results]
    colors = ["#E53935" if all_types[i] == "双均线" else "#7B1FA2" for i in range(len(all_labels))]
    faded = ["#43A047" if v < 0 else c for v, c in zip(vals, colors)]
    bars = ax.barh(range(len(all_labels)), vals, color=faded, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=7, **ZH)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    ax.set_title("累计回报率(%)", fontsize=11, fontweight="bold", **ZH)
    for bar, val in zip(bars, vals):
        ax.text(val + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=6, **ZH)

    # 2. 年化收益率
    ax = axes[0][1]
    vals = [m["ann_return"] for m in ma_results] + [m["ann_return"] for m in dc_results]
    bars = ax.barh(range(len(all_labels)), vals, color=pos_neg_colors(vals), edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=7, **ZH)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    ax.set_title("年化收益率(%)", fontsize=11, fontweight="bold", **ZH)
    for bar, val in zip(bars, vals):
        ax.text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=6, **ZH)

    # 3. 最大回撤
    ax = axes[0][2]
    vals = [m["max_dd"] for m in ma_results] + [m["max_dd"] for m in dc_results]
    dd_colors = ["#43A047" if m >= -10 else "#FF9800" if m >= -20 else "#E53935" for m in vals]
    bars = ax.barh(range(len(all_labels)), vals, color=dd_colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=7, **ZH)
    ax.set_title("最大回撤(%)", fontsize=11, fontweight="bold", **ZH)
    for bar, val in zip(bars, vals):
        ax.text(val - 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="right", fontsize=6, **ZH)

    # 4. 夏普比率
    ax = axes[0][3]
    vals = [m["sharpe"] for m in ma_results] + [m["sharpe"] for m in dc_results]
    bars = ax.barh(range(len(all_labels)), vals, color=pos_neg_colors(vals), edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=7, **ZH)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    ax.set_title("夏普比率", fontsize=11, fontweight="bold", **ZH)
    for bar, val in zip(bars, vals):
        offset = 0.04 if val >= 0 else -0.04
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=6, **ZH)

    # 5. 胜率
    ax = axes[1][0]
    vals = [m["win_rate"] for m in ma_results] + [m["win_rate"] for m in dc_results]
    bars = ax.barh(range(len(all_labels)), vals, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=7, **ZH)
    ax.set_xlim(0, 100)
    ax.set_title("胜率(%)", fontsize=11, fontweight="bold", **ZH)
    for bar, val in zip(bars, vals):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=6, **ZH)

    # 6. 盈亏比
    ax = axes[1][1]
    vals = [m["plr"] for m in ma_results] + [m["plr"] for m in dc_results]
    plr_colors = ["#E53935" if v >= 1 else "#FF9800" if v >= 0.5 else "#43A047" for v in vals]
    bars = ax.barh(range(len(all_labels)), vals, color=plr_colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=7, **ZH)
    ax.axvline(x=1, color="#333", linewidth=0.8, linestyle="--")
    ax.set_title("盈亏比", fontsize=11, fontweight="bold", **ZH)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=6, **ZH)

    # 7. 期望收益(R)
    ax = axes[1][2]
    vals = [m["expectancy_r"] for m in ma_results] + [m["expectancy_r"] for m in dc_results]
    bars = ax.barh(range(len(all_labels)), vals, color=pos_neg_colors(vals), edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=7, **ZH)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    ax.set_title("期望收益(R)", fontsize=11, fontweight="bold", **ZH)
    for bar, val in zip(bars, vals):
        offset = 0.02 if val >= 0 else -0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}R", va="center", ha="left" if val >= 0 else "right", fontsize=6, **ZH)

    # 8. Calmar 比率
    ax = axes[1][3]
    vals = [m["calmar"] for m in ma_results] + [m["calmar"] for m in dc_results]
    bars = ax.barh(range(len(all_labels)), vals, color=pos_neg_colors(vals), edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=7, **ZH)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    ax.set_title("Calmar比率", fontsize=11, fontweight="bold", **ZH)
    for bar, val in zip(bars, vals):
        offset = 0.02 if val >= 0 else -0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=6, **ZH)

    fig.suptitle("JoinQuant 参数优化网格搜索 — 双均线(红) vs 海龟(紫)", fontsize=14, fontweight="bold", **ZH)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "jq_param_optimization.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图3] {out}")
    return out


def plot_risk_analysis(df_ma, m_ma, df_tu, m_tu):
    """图4: 风险暴露分析"""
    dates = fmt_dates(df_ma["trade_date"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 日收益率分布（双均线）
    ax = axes[0][0]
    returns_ma = df_ma["strategy_return"] * 100
    ax.hist(returns_ma, bins=50, color=C_STRATEGY_MA, alpha=0.6, edgecolor="white", linewidth=0.3)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    ax.axvline(x=m_ma["var_95"], color=C_SELL, linewidth=1.5, linestyle="--", label=f"VaR(95%)={m_ma['var_95']:.2f}%")
    ax.axvline(x=m_ma["cvar_95"], color="#FF6F00", linewidth=1.5, linestyle=":", label=f"CVaR(95%)={m_ma['cvar_95']:.2f}%")
    ax.set_title("双均线策略 日收益率分布", fontsize=11, fontweight="bold", **ZH)
    ax.set_xlabel("日收益率(%)", **ZH)
    ax.set_ylabel("频次", **ZH)
    ax.legend(prop=ZH_FP_S, loc="upper right", framealpha=0.9)

    # 2. 日收益率分布（海龟）
    ax = axes[0][1]
    returns_tu = df_tu["strategy_return"] * 100
    ax.hist(returns_tu, bins=50, color=C_STRATEGY_TU, alpha=0.6, edgecolor="white", linewidth=0.3)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    ax.axvline(x=m_tu["var_95"], color=C_SELL, linewidth=1.5, linestyle="--", label=f"VaR(95%)={m_tu['var_95']:.2f}%")
    ax.axvline(x=m_tu["cvar_95"], color="#FF6F00", linewidth=1.5, linestyle=":", label=f"CVaR(95%)={m_tu['cvar_95']:.2f}%")
    ax.set_title("海龟策略 日收益率分布", fontsize=11, fontweight="bold", **ZH)
    ax.set_xlabel("日收益率(%)", **ZH)
    ax.set_ylabel("频次", **ZH)
    ax.legend(prop=ZH_FP_S, loc="upper right", framealpha=0.9)

    # 3. 回撤时间序列对比
    ax = axes[0][2]
    def calc_dd(nav):
        rm = nav.cummax()
        return (nav - rm) / rm * 100
    dd_ma = calc_dd(df_ma["strategy_nav"])
    dd_tu = calc_dd(df_tu["strategy_nav"])
    dd_bench = calc_dd(df_ma["benchmark_nav"])
    ax.plot(dates, dd_ma, color=C_STRATEGY_MA, linewidth=1.0, label="双均线回撤")
    ax.plot(dates, dd_tu, color=C_STRATEGY_TU, linewidth=1.0, label="海龟回撤")
    ax.plot(dates, dd_bench, color=C_BENCH, linewidth=0.8, linestyle="--", label="基准回撤")
    ax.fill_between(dates, dd_ma, 0, color=C_STRATEGY_MA, alpha=0.1)
    ax.set_title("回撤时间序列对比", fontsize=11, fontweight="bold", **ZH)
    ax.set_ylabel("回撤(%)", **ZH)
    ax.legend(prop=ZH_FP_S, loc="lower left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # 4. 持仓时间占比饼图
    ax = axes[1][0]
    ma_hold_pct = df_ma["position"].mean() * 100
    tu_hold_pct = df_tu["position"].mean() * 100
    bench_hold = 100.0
    labels_pie = ["双均线\n持仓", "双均线\n空仓"]
    sizes = [ma_hold_pct, 100 - ma_hold_pct]
    colors_pie = [C_STRATEGY_MA, "#E0E0E0"]
    ax.pie(sizes, labels=[f"持仓 {ma_hold_pct:.1f}%", f"空仓 {100-ma_hold_pct:.1f}%"],
           colors=colors_pie, autopct="", startangle=90,
           textprops={"fontproperties": ZH_FP_S})
    ax.set_title("双均线 持仓/空仓占比", fontsize=11, fontweight="bold", **ZH)

    # 5. 滚动夏普比率（60日窗口）
    ax = axes[1][1]
    window = 60
    daily_rf = 0.03 / 252
    roll_excess_ma = df_ma["strategy_return"] - daily_rf
    roll_excess_tu = df_tu["strategy_return"] - daily_rf
    roll_sharpe_ma = (roll_excess_ma.rolling(window).mean() / roll_excess_ma.rolling(window).std()) * np.sqrt(252)
    roll_sharpe_tu = (roll_excess_tu.rolling(window).mean() / roll_excess_tu.rolling(window).std()) * np.sqrt(252)
    ax.plot(dates, roll_sharpe_ma, color=C_STRATEGY_MA, linewidth=1.0, label="双均线滚动夏普")
    ax.plot(dates, roll_sharpe_tu, color=C_STRATEGY_TU, linewidth=1.0, label="海龟滚动夏普")
    ax.axhline(y=0, color="#333", linewidth=0.8)
    ax.axhline(y=1, color=C_SELL, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(f"滚动夏普比率（{window}日窗口）", fontsize=11, fontweight="bold", **ZH)
    ax.set_ylabel("夏普比率", **ZH)
    ax.legend(prop=ZH_FP_S, loc="best", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # 6. 风险指标雷达图
    ax = axes[1][2]
    # 归一化指标（0-1范围，越大越好）
    indicators = ["夏普", "Sortino", "Calmar", "胜率\n(/100)", "盈亏比\n(/3)", "期望R\n(/0.5)"]

    def normalize(val, max_val):
        return max(0, min(1, val / max_val))

    ma_vals = [
        normalize(m_ma["sharpe"] + 2, 4),
        normalize(m_ma["sortino"] + 2, 4),
        normalize(m_ma["calmar"] + 1, 3),
        m_ma["win_rate"] / 100,
        m_ma["plr"] / 3,
        (m_ma["expectancy_r"] + 0.5) / 1.0,
    ]
    tu_vals = [
        normalize(m_tu["sharpe"] + 2, 4),
        normalize(m_tu["sortino"] + 2, 4),
        normalize(m_tu["calmar"] + 1, 3),
        m_tu["win_rate"] / 100,
        m_tu["plr"] / 3,
        (m_tu["expectancy_r"] + 0.5) / 1.0,
    ]

    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    ma_vals += ma_vals[:1]
    tu_vals += tu_vals[:1]
    angles += angles[:1]

    ax = plt.subplot(2, 3, 6, polar=True)
    ax.plot(angles, ma_vals, color=C_STRATEGY_MA, linewidth=1.5, label="双均线")
    ax.fill(angles, ma_vals, color=C_STRATEGY_MA, alpha=0.15)
    ax.plot(angles, tu_vals, color=C_STRATEGY_TU, linewidth=1.5, label="海龟")
    ax.fill(angles, tu_vals, color=C_STRATEGY_TU, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators, fontsize=8, **ZH)
    ax.set_title("风险调整指标雷达图", fontsize=11, fontweight="bold", **ZH, pad=15)
    ax.legend(prop=ZH_FP_S, loc="upper right", bbox_to_anchor=(1.3, 1.1), framealpha=0.9)

    fig.suptitle("JoinQuant 平台策略风险暴露分析", fontsize=14, fontweight="bold", **ZH)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "jq_risk_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图4] {out}")
    return out


def main():
    print("=" * 60)
    print("JoinQuant 平台策略可视化")
    print("=" * 60)

    # 运行策略
    df_ma, m_ma = run_dual_ma()
    df_tu, m_tu = run_turtle()
    ma_results, dc_results = run_param_sweep()

    # 图1
    plot_signals(df_ma, JQ_SHORT, JQ_LONG)
    # 图2
    plot_nav_compare(df_ma, df_tu)
    # 图3
    plot_param_optimization(ma_results, dc_results)
    # 图4
    plot_risk_analysis(df_ma, m_ma, df_tu, m_tu)

    print("\n--- 策略绩效汇总 ---")
    print(f"双均线 MA{JQ_SHORT}/{JQ_LONG}:")
    for k in ["cum_return", "ann_return", "max_dd", "sharpe", "win_rate", "plr", "expectancy_r", "var_95", "cvar_95"]:
        print(f"  {k}: {m_ma[k]:.3f}")
    print(f"\n海龟 DC{JQ_DC_ENTRY}/{JQ_DC_EXIT}:")
    for k in ["cum_return", "ann_return", "max_dd", "sharpe", "win_rate", "plr", "expectancy_r", "var_95", "cvar_95"]:
        print(f"  {k}: {m_tu[k]:.3f}")

    return df_ma, m_ma, df_tu, m_tu, ma_results, dc_results


if __name__ == "__main__":
    main()
