# -*- coding: utf-8 -*-
"""
plot_ml_strategy.py
机器学习选股策略可视化:
  图1: 三模型季度收益对比柱状图
  图2: 累计净值曲线 (三模型 vs 基准)
  图3: 随机森林因子重要性
  图4: 三模型绩效指标对比表图
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import platform

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from ml_strategy import main as run_strategy

ZH_FONT_PATH = r"C:\Windows\Fonts\msyh.ttc" if platform.system() == "Windows" else None

COLORS = {
    "线性回归": "#2196F3",
    "决策树": "#FF9800",
    "随机森林": "#E53935",
}
COLOR_BENCH = "#78909C"


def plot_quarterly_returns(all_backtest, all_perf):
    """图1: 三模型季度收益对比柱状图"""
    fig, ax = plt.subplots(figsize=(14, 6))

    quarters = all_backtest["随机森林"]["quarter_labels"]
    x = np.arange(len(quarters))
    width = 0.2

    # 基准
    bench = all_backtest["随机森林"]["benchmark_returns"]
    ax.bar(x - width*1.5, [b*100 for b in bench], width, color=COLOR_BENCH, label="市场基准(等权)", alpha=0.7)

    # 三模型
    for i, (name, color) in enumerate(COLORS.items()):
        rets = all_backtest[name]["strategy_returns"]
        ax.bar(x + (i-1)*width, [r*100 for r in rets], width, color=color, label=name, alpha=0.8)

    ax.set_xlabel("季度", **ZH)
    ax.set_ylabel("季度收益率(%)", **ZH)
    ax.set_title("三模型季度选股策略收益 vs 市场基准", fontsize=14, fontweight="bold", **ZH)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=9)
    ax.axhline(y=0, color="#333", linewidth=0.8)
    ax.legend(prop=font_manager.FontProperties(fname=ZH_FONT_PATH, size=10), loc="best")
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "quarterly_returns.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图1] {out}")
    return out


def plot_nav_curves(all_backtest):
    """图2: 累计净值曲线"""
    fig, ax = plt.subplots(figsize=(14, 6))

    quarters = all_backtest["随机森林"]["quarter_labels"]

    # 基准
    ax.plot(range(len(all_backtest["随机森林"]["benchmark_nav"])),
            all_backtest["随机森林"]["benchmark_nav"],
            color=COLOR_BENCH, linewidth=2, linestyle="--", label="市场基准(等权)")

    # 三模型
    for name, color in COLORS.items():
        nav = all_backtest[name]["strategy_nav"]
        ax.plot(range(len(nav)), nav, color=color, linewidth=1.5, label=f"{name}")

    ax.set_xlabel("季度", **ZH)
    ax.set_ylabel("累计净值", **ZH)
    ax.set_title("机器学习选股策略累计净值 vs 市场基准", fontsize=14, fontweight="bold", **ZH)
    ax.axhline(y=1.0, color="#666", linestyle=":", linewidth=0.8)

    # x轴标签
    n_points = len(all_backtest["随机森林"]["strategy_nav"])
    x_labels = ["起始"] + quarters
    ax.set_xticks(range(n_points))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

    ax.legend(prop=font_manager.FontProperties(fname=ZH_FONT_PATH, size=10), loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, "nav_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图2] {out}")
    return out


def plot_feature_importance(model_results):
    """图3: 随机森林因子重要性"""
    fi = model_results["随机森林"]["feature_importance"]
    if fi is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(12, len(fi))
    fi_top = fi.head(top_n)

    bars = ax.barh(fi_top.index[::-1], fi_top.values[::-1], color="#E53935",
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("重要性", **ZH)
    ax.set_title("随机森林因子重要性排序", fontsize=14, fontweight="bold", **ZH)
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    for bar, val in zip(bars, fi_top.values[::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, **ZH)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "feature_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图3] {out}")
    return out


def plot_performance_comparison(all_perf):
    """图4: 三模型绩效指标对比"""
    metrics_to_plot = [
        ("年化收益率(%)", "ann_return", "bench_ann_return"),
        ("最大回撤(%)", "mdd", "bench_mdd"),
        ("夏普比率", "sharpe", "bench_sharpe"),
        ("胜率(%)", "win_rate", None),
        ("盈亏比", "plr", None),
        ("累计回报(%)", "cum_return", "bench_cum_return"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    model_names = list(COLORS.keys())

    for idx, (title, key, bench_key) in enumerate(metrics_to_plot):
        ax = axes[idx // 3][idx % 3]
        values = [all_perf[name][key] for name in model_names]
        colors = [COLORS[name] for name in model_names]

        bars = ax.bar(model_names, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold", **ZH)
        ax.axhline(y=0, color="#333", linewidth=0.5)

        # 基准线
        if bench_key is not None:
            bench_val = all_perf["随机森林"][bench_key]
            ax.axhline(y=bench_val, color=COLOR_BENCH, linestyle="--", linewidth=1.5, label=f"基准: {bench_val:.2f}")
            ax.legend(prop=font_manager.FontProperties(fname=ZH_FONT_PATH, size=8), loc="best")

        for bar, val in zip(bars, values):
            offset = max(abs(val) * 0.05, 0.01)
            ax.text(bar.get_x() + bar.get_width() / 2, val + (offset if val >= 0 else -offset),
                    f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8, **ZH)

        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    fig.suptitle("三模型选股策略绩效指标对比", fontsize=14, fontweight="bold", **ZH)
    plt.tight_layout()
    out = os.path.join(BASE_DIR, "performance_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图4] {out}")
    return out


def main():
    print("=" * 60)
    print("机器学习选股策略可视化")
    print("=" * 60)

    # 运行策略
    model_results, all_backtest, all_perf, _ = run_strategy()

    # 生成图表
    plot_quarterly_returns(all_backtest, all_perf)
    plot_nav_curves(all_backtest)
    plot_feature_importance(model_results)
    plot_performance_comparison(all_perf)

    print("\n=== 全部图表生成完成 ===")


if __name__ == "__main__":
    main()
