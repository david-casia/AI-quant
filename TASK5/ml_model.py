# -*- coding: utf-8 -*-
"""
ml_model.py
机器学习分类模型引擎：数据构建 → 特征工程 → 三模型训练 → 评估(混淆矩阵/AUC/ROC)
使用 scikit-learn 乳腺癌数据集 + 比亚迪股票收益数据集
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")
RANDOM_STATE = 42


# ============ 数据集1: 乳腺癌数据集 ============
def load_cancer_data():
    """
    加载scikit-learn乳腺癌数据集
    应变量: 0(恶性) / 1(良性)
    特征: 30个细胞核形态特征
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, "乳腺癌数据集"


# ============ 数据集2: 股票收益数据集 ============
def load_stock_data(csv_path=None):
    """
    基于比亚迪日线数据构建股票收益分类数据集
    特征: 技术指标(均线、动量、波动率、成交量变化等)
    应变量: 0(次日下跌) / 1(次日上涨)
    """
    if csv_path is None:
        csv_path = CSV_PATH
    df = pd.read_csv(csv_path)
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 计算技术指标作为特征
    close = df["close"]
    vol = df["vol"]

    # 1. 收益率
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    # 2. 均线偏离度
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma5_dev"] = (close - df["ma5"]) / df["ma5"]
    df["ma10_dev"] = (close - df["ma10"]) / df["ma10"]
    df["ma20_dev"] = (close - df["ma20"]) / df["ma20"]

    # 3. 波动率
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # 4. 成交量变化
    df["vol_change_1d"] = vol.pct_change(1)
    df["vol_change_5d"] = vol.pct_change(5)
    df["vol_ma5_ratio"] = vol / vol.rolling(5).mean()

    # 5. 动量指标
    df["momentum_5d"] = close / close.shift(5) - 1
    df["momentum_10d"] = close / close.shift(10) - 1

    # 6. 价格形态
    df["high_low_ratio"] = (df["high"] - df["low"]) / close
    df["close_open_ratio"] = (close - df["open"]) / df["open"]

    # 7. RSI (简化版)
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 应变量: 次日涨跌 (1=涨, 0=跌)
    df["target"] = (close.shift(-1) > close).astype(int)

    # 选择特征列
    feature_cols = [
        "return_1d", "return_5d", "return_10d",
        "ma5_dev", "ma10_dev", "ma20_dev",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "vol_change_1d", "vol_change_5d", "vol_ma5_ratio",
        "momentum_5d", "momentum_10d",
        "high_low_ratio", "close_open_ratio", "rsi_14",
    ]

    # 去掉NaN行
    df_clean = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    X = df_clean[feature_cols].copy()
    y = df_clean["target"].copy()

    return X, y, "比亚迪股票收益数据集"


# ============ 模型训练与评估 ============
def train_and_evaluate(X, y, test_size=0.3, random_state=RANDOM_STATE):
    """
    训练三个分类模型并评估:
    1. 逻辑回归 (Logistic Regression)
    2. 决策树 (Decision Tree)
    3. 随机森林 (Random Forest)

    返回: results dict
    """
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 标准化 (逻辑回归需要, 决策树/随机森林不需要但统一处理)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义三个模型
    models = {
        "逻辑回归": LogisticRegression(max_iter=5000, random_state=random_state),
        "决策树": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "随机森林": RandomForestClassifier(n_estimators=100, max_depth=8,
                                       random_state=random_state, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        # 训练
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # 评估指标
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)

        # ROC曲线
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # 特征重要性 (随机森林)
        feature_importance = None
        if name == "随机森林" and hasattr(model, "feature_importances_"):
            feature_importance = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "confusion_matrix": cm,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "feature_importance": feature_importance,
        }

    return results, X_train, X_test, y_train, y_test


def main():
    print("=" * 60)
    print("机器学习分类模型 - TASK5")
    print("=" * 60)

    # 数据集1: 乳腺癌
    print("\n===== 数据集1: 乳腺癌数据集 =====")
    X1, y1, name1 = load_cancer_data()
    print(f"数据集: {name1}")
    print(f"样本数: {len(X1)}, 特征数: {X1.shape[1]}")
    print(f"类别分布: {y1.value_counts().to_dict()} (0=恶性, 1=良性)")

    results1, _, _, _, _ = train_and_evaluate(X1, y1)
    print("\n--- 模型评估结果 ---")
    for name, r in results1.items():
        print(f"\n  [{name}]")
        print(f"    准确率(Accuracy):  {r['accuracy']:.4f}")
        print(f"    精确率(Precision): {r['precision']:.4f}")
        print(f"    召回率(Recall):    {r['recall']:.4f}")
        print(f"    F1分数:            {r['f1']:.4f}")
        print(f"    AUC:               {r['auc']:.4f}")
        print(f"    混淆矩阵: TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}")

    # 数据集2: 股票收益
    print("\n\n===== 数据集2: 比亚迪股票收益数据集 =====")
    X2, y2, name2 = load_stock_data()
    print(f"数据集: {name2}")
    print(f"样本数: {len(X2)}, 特征数: {X2.shape[1]}")
    print(f"类别分布: {y2.value_counts().to_dict()} (0=下跌, 1=上涨)")

    results2, _, _, _, _ = train_and_evaluate(X2, y2)
    print("\n--- 模型评估结果 ---")
    for name, r in results2.items():
        print(f"\n  [{name}]")
        print(f"    准确率(Accuracy):  {r['accuracy']:.4f}")
        print(f"    精确率(Precision): {r['precision']:.4f}")
        print(f"    召回率(Recall):    {r['recall']:.4f}")
        print(f"    F1分数:            {r['f1']:.4f}")
        print(f"    AUC:               {r['auc']:.4f}")
        print(f"    混淆矩阵: TN={r['tn']} FP={r['fp']} FN={r['fn']} TP={r['tp']}")

    # 特征重要性
    if results2["随机森林"]["feature_importance"] is not None:
        print("\n--- 股票数据集随机森林特征重要性 (Top 10) ---")
        fi = results2["随机森林"]["feature_importance"].head(10)
        for feat, imp in fi.items():
            print(f"  {feat:25s} {imp:.4f}")


if __name__ == "__main__":
    main()
