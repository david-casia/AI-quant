# -*- coding: utf-8 -*-
"""
plot_ml.py
机器学习模型可视化：
  图1: 乳腺癌数据集 ROC曲线 (3模型对比)
  图2: 乳腺癌数据集 混淆矩阵 (3模型)
  图3: 股票数据集 ROC曲线 (3模型对比)
  图4: 股票数据集 混淆矩阵 (3模型)
  图5: 股票数据集 随机森林特征重要性
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from ml_model import load_cancer_data, load_stock_data, train_and_evaluate

ZH_FONT_PATH = r"C:\Windows\Fonts\msyh.ttc" if platform.system() == "Windows" else None

# 配色
COLORS = {
    "逻辑回归": "#2196F3",
    "决策树": "#FF9800",
    "随机森林": "#E53935",
}


def plot_roc(results, title, filename):
    """绘制ROC曲线对比图"""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, r in results.items():
        ax.plot(r["fpr"], r["tpr"], color=COLORS[name], linewidth=2,
                label=f"{name} (AUC = {r['auc']:.4f})")

    # 对角线
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1, label="随机分类器 (AUC = 0.5)")

    ax.set_xlabel("假正率 (FPR)", fontsize=12, **ZH)
    ax.set_ylabel("真正率 (TPR)", fontsize=12, **ZH)
    ax.set_title(title, fontsize=14, fontweight="bold", **ZH)
    ax.legend(prop=font_manager.FontProperties(fname=ZH_FONT_PATH, size=11),
              loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {out}")
    return out


def plot_confusion_matrices(results, title, filename):
    """绘制3个模型的混淆矩阵"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        cm = r["confusion_matrix"]

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name}\nAUC={r['auc']:.4f}", fontsize=12, fontweight="bold", **ZH)

        # 标注数值
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=16, fontweight="bold")

        labels = ["负类(0)", "正类(1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, **ZH)
        ax.set_yticklabels(labels, **ZH)
        ax.set_xlabel("预测标签", **ZH)
        ax.set_ylabel("真实标签", **ZH)

    fig.suptitle(title, fontsize=14, fontweight="bold", **ZH, y=1.02)
    plt.tight_layout()

    out = os.path.join(BASE_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {out}")
    return out


def plot_feature_importance(feature_importance, title, filename):
    """绘制特征重要性图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    top_n = min(15, len(feature_importance))
    fi = feature_importance.head(top_n)

    bars = ax.barh(fi.index[::-1], fi.values[::-1], color="#E53935",
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("重要性", **ZH)
    ax.set_title(title, fontsize=14, fontweight="bold", **ZH)
    ax.grid(True, linestyle="--", alpha=0.3, axis="x")

    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, **ZH)

    plt.tight_layout()

    out = os.path.join(BASE_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {out}")
    return out


def main():
    print("=" * 60)
    print("机器学习模型可视化")
    print("=" * 60)

    # === 乳腺癌数据集 ===
    print("\n--- 乳腺癌数据集 ---")
    X1, y1, name1 = load_cancer_data()
    results1, _, _, _, _ = train_and_evaluate(X1, y1)
    plot_roc(results1, "乳腺癌数据集 ROC曲线对比", "cancer_roc.png")
    plot_confusion_matrices(results1, "乳腺癌数据集 混淆矩阵对比", "cancer_confusion.png")

    # === 股票数据集 ===
    print("\n--- 股票收益数据集 ---")
    X2, y2, name2 = load_stock_data()
    results2, _, _, _, _ = train_and_evaluate(X2, y2)
    plot_roc(results2, "比亚迪股票收益数据集 ROC曲线对比", "stock_roc.png")
    plot_confusion_matrices(results2, "比亚迪股票收益数据集 混淆矩阵对比", "stock_confusion.png")

    # 特征重要性
    if results2["随机森林"]["feature_importance"] is not None:
        plot_feature_importance(
            results2["随机森林"]["feature_importance"],
            "随机森林特征重要性 (股票收益数据集)",
            "stock_feature_importance.png"
        )

    print("\n=== 全部图表生成完成 ===")


if __name__ == "__main__":
    main()
