# -*- coding: utf-8 -*-
"""
plot_strategy.py
双均线策略可视化：
  图1: 股价 + 短长均线 + 买卖信号标记
  图2: 策略净值 vs 基准净值 + 回撤区域
  图3: 不同均线参数组合的累计回报对比
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

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from strategy import run_strategy, calc_ma, generate_signals, backtest, calc_metrics, load_data, DEFAULT_SHORT, DEFAULT_LONG

CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")

# 中国股市配色：涨红跌绿
COLOR_PRICE = "#2196F3"      # 股价蓝
COLOR_MA_SHORT = "#FF9800"   # 短均线橙
COLOR_MA_LONG = "#9C27B0"    # 长均线紫
COLOR_BUY = "#E53935"        # 买入红
COLOR_SELL = "#43A047"       # 卖出绿
COLOR_STRATEGY = "#E53935"   # 策略红
COLOR_BENCHMARK = "#78909C"  # 基准灰


def fmt_date_series(s):
    """将 YYYYMMDD 字符串转为可排序的日期"""
    return pd.to_datetime(s.astype(str).str.replace("-", ""), format="%Y%m%d")


def plot_ma_signals(df, short_period, long_period):
    """图1: 股价 + 均线 + 买卖信号"""
    dates = fmt_date_series(df["trade_date"])

    fig, ax = plt.subplots(figsize=(14, 6))

    # 股价
    ax.plot(dates, df["close"], color=COLOR_PRICE, linewidth=1.2, label="收盘价", zorder=2)
    # 均线
    ax.plot(dates, df["ma_short"], color=COLOR_MA_SHORT, linewidth=1.0, label=f"MA{short_period}", zorder=3)
    ax.plot(dates, df["ma_long"], color=COLOR_MA_LONG, linewidth=1.0, label=f"MA{long_period}", zorder=3)

    # 买卖信号标记
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["signal"] == -1]

    if len(buy_signals) > 0:
        buy_dates = fmt_date_series(buy_signals["trade_date"])
        ax.scatter(buy_dates, buy_signals["close"], color=COLOR_BUY, marker="^",
                   s=100, zorder=5, label="买入信号(金叉)")
        # 买入标注
        for _, row in buy_signals.iterrows():
            d = pd.to_datetime(str(row["trade_date"]).replace("-", ""), format="%Y%m%d")
            ax.annotate(f"{row['close']:.1f}", (d, row["close"]),
                        textcoords="offset points", xytext=(0, -18),
                        fontsize=7, color=COLOR_BUY, ha="center", **ZH)

    if len(sell_signals) > 0:
        sell_dates = fmt_date_series(sell_signals["trade_date"])
        ax.scatter(sell_dates, sell_signals["close"], color=COLOR_SELL, marker="v",
                   s=100, zorder=5, label="卖出信号(死叉)")
        for _, row in sell_signals.iterrows():
            d = pd.to_datetime(str(row["trade_date"]).replace("-", ""), format="%Y%m%d")
            ax.annotate(f"{row['close']:.1f}", (d, row["close"]),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=7, color=COLOR_SELL, ha="center", **ZH)

    # 持仓区域背景
    holding_ranges = []
    start = None
    for i, row in df.iterrows():
        if row["position"] == 1 and start is None:
            start = i
        elif row["position"] == 0 and start is not None:
            holding_ranges.append((start, i - 1))
            start = None
    if start is not None:
        holding_ranges.append((start, len(df) - 1))

    for s, e in holding_ranges:
        ax.axvspan(dates.iloc[s], dates.iloc[e], alpha=0.08, color=COLOR_BUY, zorder=1)

    ax.set_title(f"比亚迪(002594.SZ) 双均线策略 — MA{short_period}/{long_period}",
                 fontsize=14, fontweight="bold", **ZH)
    ax.set_xlabel("日期", **ZH)
    ax.set_ylabel("价格(元)", **ZH)
    ax.legend(prop=font_manager.FontProperties(
        fname=r"C:\Windows\Fonts\msyh.ttc" if platform.system() == "Windows" else None,
        size=9), loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "002594_ma_signals.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图1] {out}")
    return out


def plot_nav_drawdown(df):
    """图2: 净值曲线 + 回撤"""
    dates = fmt_date_series(df["trade_date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # 上图：净值
    ax1.plot(dates, df["strategy_nav"], color=COLOR_STRATEGY, linewidth=1.5, label="策略净值")
    ax1.plot(dates, df["benchmark_nav"], color=COLOR_BENCHMARK, linewidth=1.5, label="基准(买入持有)", linestyle="--")
    ax1.axhline(y=1.0, color="#666", linestyle=":", linewidth=0.8)
    ax1.set_title("策略净值 vs 基准净值(买入持有)", fontsize=13, fontweight="bold", **ZH)
    ax1.set_ylabel("累计净值", **ZH)
    ax1.legend(prop=font_manager.FontProperties(
        fname=r"C:\Windows\Fonts\msyh.ttc" if platform.system() == "Windows" else None,
        size=9), loc="best", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 下图：回撤
    running_max = df["strategy_nav"].cummax()
    drawdown = (df["strategy_nav"] - running_max) / running_max * 100
    bench_running_max = df["benchmark_nav"].cummax()
    bench_dd = (df["benchmark_nav"] - bench_running_max) / bench_running_max * 100

    ax2.fill_between(dates, drawdown, 0, color=COLOR_STRATEGY, alpha=0.3, label="策略回撤")
    ax2.fill_between(dates, bench_dd, 0, color=COLOR_BENCHMARK, alpha=0.2, label="基准回撤")
    ax2.set_ylabel("回撤(%)", **ZH)
    ax2.set_xlabel("日期", **ZH)
    ax2.legend(prop=font_manager.FontProperties(
        fname=r"C:\Windows\Fonts\msyh.ttc" if platform.system() == "Windows" else None,
        size=9), loc="lower left", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "002594_nav_drawdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图2] {out}")
    return out


def plot_param_comparison():
    """图3: 不同均线参数组合对比"""
    df_raw = load_data()

    combos = [
        (3, 5), (3, 10), (5, 10), (5, 15), (5, 20),
        (5, 30), (10, 20), (10, 30), (10, 60), (20, 60),
    ]

    results = []
    for short, long in combos:
        df = calc_ma(df_raw.copy(), short, long)
        df = generate_signals(df)
        df = backtest(df)
        m = calc_metrics(df)
        results.append({
            "combo": f"MA{short}/{long}",
            "short": short,
            "long": long,
            "return": m["cumulative_return"],
            "mdd": m["max_drawdown"],
            "sharpe": m["sharpe"],
            "trades": m["n_trades"],
            "win_rate": m["win_rate"],
        })

    res_df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    bar_colors_ret = ["#E53935" if r >= 0 else "#43A047" for r in res_df["return"]]
    bar_colors_mdd = ["#43A047" if m >= -10 else "#FF9800" if m >= -20 else "#E53935" for m in res_df["mdd"]]

    # 累计回报
    ax = axes[0][0]
    bars = ax.barh(res_df["combo"], res_df["return"], color=bar_colors_ret, edgecolor="white", linewidth=0.5)
    ax.set_title("累计回报率(%)", fontsize=12, fontweight="bold", **ZH)
    ax.set_xlabel("回报率(%)", **ZH)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    for bar, val in zip(bars, res_df["return"]):
        ax.text(val + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=8, **ZH)

    # 最大回撤
    ax = axes[0][1]
    bars = ax.barh(res_df["combo"], res_df["mdd"], color=bar_colors_mdd, edgecolor="white", linewidth=0.5)
    ax.set_title("最大回撤(%)", fontsize=12, fontweight="bold", **ZH)
    ax.set_xlabel("回撤(%)", **ZH)
    for bar, val in zip(bars, res_df["mdd"]):
        ax.text(val - 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="right", fontsize=8, **ZH)

    # 夏普比率
    ax = axes[1][0]
    bar_colors_sharpe = ["#E53935" if s >= 0 else "#43A047" for s in res_df["sharpe"]]
    bars = ax.barh(res_df["combo"], res_df["sharpe"], color=bar_colors_sharpe, edgecolor="white", linewidth=0.5)
    ax.set_title("夏普比率", fontsize=12, fontweight="bold", **ZH)
    ax.set_xlabel("Sharpe Ratio", **ZH)
    ax.axvline(x=0, color="#333", linewidth=0.8)
    for bar, val in zip(bars, res_df["sharpe"]):
        offset = 0.05 if val >= 0 else -0.05
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=8, **ZH)

    # 胜率
    ax = axes[1][1]
    bars = ax.barh(res_df["combo"], res_df["win_rate"], color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set_title("胜率(%)", fontsize=12, fontweight="bold", **ZH)
    ax.set_xlabel("胜率(%)", **ZH)
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, res_df["win_rate"]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8, **ZH)

    fig.suptitle("不同均线参数组合绩效对比", fontsize=14, fontweight="bold", **ZH)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "002594_param_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图3] {out}")
    return out, res_df


def main():
    print("=" * 60)
    print("双均线策略可视化")
    print("=" * 60)

    # 图1+图2: 默认参数 MA5/15
    df, m = run_strategy()
    plot_ma_signals(df, DEFAULT_SHORT, DEFAULT_LONG)
    plot_nav_drawdown(df)

    # 图3: 参数对比
    _, res_df = plot_param_comparison()

    print("\n--- 参数对比汇总 ---")
    print(res_df.to_string(index=False))
    print(f"\n最佳回报组合: MA{res_df.loc[res_df['return'].idxmax(), 'short']}/{res_df.loc[res_df['return'].idxmax(), 'long']}")
    print(f"最低回撤组合: MA{res_df.loc[res_df['mdd'].idxmax(), 'short']}/{res_df.loc[res_df['mdd'].idxmax(), 'long']}")


if __name__ == "__main__":
    main()
