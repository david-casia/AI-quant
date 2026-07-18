# -*- coding: utf-8 -*-
"""
ml_strategy.py
机器学习选股策略引擎：
  多股票数据加载 → 因子构建 → 模型训练(决策树/随机森林/线性回归)
  → 季度选股(预测收益Top30) → 回测评估 → 7项绩效指标
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stock_data")
RANDOM_STATE = 42
TRADING_DAYS_PER_YEAR = 252

# 回测参数
INITIAL_CAPITAL = 100000.0
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.001
TOP_N = 30  # 每季度选股数量


# ============ 1. 加载多股票数据 ============
def load_all_stocks():
    """加载stock_data目录下所有股票CSV，返回合并的DataFrame"""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    all_data = []
    stock_codes = []

    for f in csv_files:
        ts_code = os.path.basename(f).replace(".csv", "").replace("_", ".")
        df = pd.read_csv(f)
        df["ts_code"] = ts_code
        df = df.sort_values("trade_date").reset_index(drop=True)
        all_data.append(df)
        stock_codes.append(ts_code)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"加载 {len(stock_codes)} 只股票, 共 {len(combined)} 条记录")
    return combined, stock_codes


# ============ 2. 因子构建 ============
def build_factors(df):
    """
    为每只股票构建技术因子(自变量)和未来收益(应变量)
    因子列表:
    - return_5d: 5日收益率
    - return_20d: 20日收益率
    - ma5_dev: 偏离5日均线
    - ma20_dev: 偏离20日均线
    - volatility_20d: 20日波动率
    - vol_change_5d: 5日成交量变化
    - vol_ma5_ratio: 成交量/5日均量
    - rsi_14: 14日RSI
    - momentum_10d: 10日动量
    - high_low_ratio: 振幅
    - turnover_proxy: 换手率代理(成交额/收盘价)
    - price_ma_ratio: 价格/20日均线
    应变量:
    - forward_return_60d: 未来60个交易日收益率
    """
    results = []

    for ts_code, group in df.groupby("ts_code"):
        g = group.copy().sort_values("trade_date").reset_index(drop=True)
        close = g["close"]
        vol = g["vol"]
        high = g["high"]
        low = g["low"]
        open_p = g["open"]
        amount = g["amount"]

        # === 因子构建 ===
        # 1. 历史收益率
        g["return_5d"] = close.pct_change(5)
        g["return_20d"] = close.pct_change(20)

        # 2. 均线偏离
        g["ma5"] = close.rolling(5).mean()
        g["ma20"] = close.rolling(20).mean()
        g["ma5_dev"] = (close - g["ma5"]) / g["ma5"]
        g["ma20_dev"] = (close - g["ma20"]) / g["ma20"]
        g["price_ma_ratio"] = close / g["ma20"]

        # 3. 波动率
        daily_ret = close.pct_change()
        g["volatility_20d"] = daily_ret.rolling(20).std()

        # 4. 成交量因子
        g["vol_change_5d"] = vol.pct_change(5)
        g["vol_ma5"] = vol.rolling(5).mean()
        g["vol_ma5_ratio"] = vol / g["vol_ma5"]

        # 5. RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        g["rsi_14"] = 100 - (100 / (1 + rs))

        # 6. 动量
        g["momentum_10d"] = close / close.shift(10) - 1

        # 7. 振幅
        g["high_low_ratio"] = (high - low) / close

        # 8. 换手率代理
        g["turnover_proxy"] = amount / close

        # === 应变量: 未来60日收益率 ===
        g["forward_return_60d"] = close.shift(-60) / close - 1

        # 季度信息 (用于按季度分组)
        g["year"] = g["trade_date"].astype(str).str[:4]
        g["month"] = g["trade_date"].astype(str).str[4:6].astype(int)
        g["quarter"] = g["year"] + "Q" + (((g["month"] - 1) // 3) + 1).astype(str)

        results.append(g)

    combined = pd.concat(results, ignore_index=True)
    return combined


FACTOR_COLS = [
    "return_5d", "return_20d",
    "ma5_dev", "ma20_dev", "price_ma_ratio",
    "volatility_20d",
    "vol_change_5d", "vol_ma5_ratio",
    "rsi_14", "momentum_10d",
    "high_low_ratio", "turnover_proxy",
]

TARGET_COL = "forward_return_60d"


# ============ 3. 模型训练 ============
def train_models(X_train, y_train, X_test, y_test):
    """训练三个回归模型并评估"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "线性回归": LinearRegression(),
        "决策树": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "随机森林": RandomForestRegressor(n_estimators=100, max_depth=8,
                                     random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 特征重要性
        feature_importance = None
        if name == "随机森林" and hasattr(model, "feature_importances_"):
            feature_importance = pd.Series(
                model.feature_importances_, index=FACTOR_COLS
            ).sort_values(ascending=False)
        elif name == "决策树" and hasattr(model, "feature_importances_"):
            feature_importance = pd.Series(
                model.feature_importances_, index=FACTOR_COLS
            ).sort_values(ascending=False)

        results[name] = {
            "model": model,
            "scaler": scaler,
            "y_pred": y_pred,
            "mse": mse,
            "r2": r2,
            "feature_importance": feature_importance,
        }
        print(f"  [{name}] MSE={mse:.6f}, R²={r2:.4f}")

    return results


# ============ 4. 季度选股策略回测 ============
def backtest_strategy(df, model_result, top_n=TOP_N):
    """
    基于模型预测的季度选股策略:
    - 每季度初: 用模型预测所有股票未来60日收益, 选Top N
    - 持有一个季度(~60个交易日), 等权配置
    - 计算季度收益率, 与市场平均(所有股票等权)对比
    """
    model = model_result["model"]
    scaler = model_result["scaler"]

    # 准备因子数据
    df_clean = df.dropna(subset=FACTOR_COLS + [TARGET_COL]).copy()
    df_clean = df_clean.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 获取所有季度
    quarters = sorted(df_clean["quarter"].unique())

    # 策略: 每季度选股
    strategy_returns = []
    benchmark_returns = []
    quarter_labels = []
    selected_stocks_per_q = {}

    for i in range(len(quarters) - 1):
        q_train = quarters[i]  # 用当前季度数据训练/预测
        q_eval = quarters[i + 1]  # 评估下一季度实际收益

        # 当前季度的数据作为预测依据
        train_data = df_clean[df_clean["quarter"] == q_train].copy()

        if len(train_data) < 10:
            continue

        # 预测
        X = train_data[FACTOR_COLS].values
        X_scaled = scaler.transform(X)
        train_data["pred_return"] = model.predict(X_scaled)

        # 选Top N (按预测收益排序)
        top_stocks = train_data.nlargest(top_n, "pred_return")["ts_code"].tolist()
        selected_stocks_per_q[q_eval] = top_stocks

        # 计算下一季度的实际收益
        eval_data = df_clean[df_clean["quarter"] == q_eval].copy()

        # 策略收益: Top N股票等权平均
        top_eval = eval_data[eval_data["ts_code"].isin(top_stocks)]
        if len(top_eval) == 0:
            continue

        # 取每只股票在该季度的收益 (60日远期收益的均值)
        strategy_ret = top_eval[TARGET_COL].mean()

        # 基准收益: 所有股票等权平均
        benchmark_ret = eval_data[TARGET_COL].mean()

        strategy_returns.append(strategy_ret)
        benchmark_returns.append(benchmark_ret)
        quarter_labels.append(q_eval)

    # 计算累计收益
    strategy_nav = [1.0]
    benchmark_nav = [1.0]
    for sr, br in zip(strategy_returns, benchmark_returns):
        strategy_nav.append(strategy_nav[-1] * (1 + sr))
        benchmark_nav.append(benchmark_nav[-1] * (1 + br))

    return {
        "quarter_labels": quarter_labels,
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "strategy_nav": strategy_nav,
        "benchmark_nav": benchmark_nav,
        "selected_stocks": selected_stocks_per_q,
    }


# ============ 5. 绩效指标计算 ============
def calc_performance(backtest_result):
    """计算7项核心绩效指标"""
    sr = np.array(backtest_result["strategy_returns"])
    br = np.array(backtest_result["benchmark_returns"])
    snav = np.array(backtest_result["strategy_nav"])
    bnav = np.array(backtest_result["benchmark_nav"])

    n = len(sr)
    if n == 0:
        return {}

    # 年化收益率 (每季度~63个交易日, 4个季度=252交易日)
    ann_return = (snav[-1] ** (TRADING_DAYS_PER_YEAR / (n * 63)) - 1) * 100
    bench_ann_return = (bnav[-1] ** (TRADING_DAYS_PER_YEAR / (n * 63)) - 1) * 100

    # 最大回撤
    def max_dd(nav):
        running_max = np.maximum.accumulate(nav)
        dd = (nav - running_max) / running_max
        return dd.min() * 100

    mdd = max_dd(snav)
    bench_mdd = max_dd(bnav)

    # 年化波动率 (季度收益波动 × sqrt(4))
    ann_vol = np.std(sr) * np.sqrt(4) * 100
    bench_vol = np.std(br) * np.sqrt(4) * 100

    # 夏普比率
    rf = 0.02 / 4  # 季度无风险利率
    sharpe = np.sqrt(4) * (np.mean(sr) - rf) / (np.std(sr) + 1e-10)
    bench_sharpe = np.sqrt(4) * (np.mean(br) - rf) / (np.std(br) + 1e-10)

    # 胜率: 策略季度收益 > 基准季度收益
    wins = np.sum(sr > br)
    win_rate = wins / n * 100

    # 盈亏比
    profits = sr[sr > 0]
    losses = np.abs(sr[sr < 0])
    avg_profit = np.mean(profits) if len(profits) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    plr = (avg_profit / avg_loss) if avg_loss > 0 else 0

    # 期望收益
    exp_r = (np.mean(profits) * len(profits) - np.mean(losses) * len(losses)) / n if avg_loss > 0 else 0

    # 累计回报
    cum_return = (snav[-1] - 1) * 100
    bench_cum_return = (bnav[-1] - 1) * 100
    alpha = cum_return - bench_cum_return

    return {
        "ann_return": ann_return,
        "bench_ann_return": bench_ann_return,
        "mdd": mdd,
        "bench_mdd": bench_mdd,
        "ann_vol": ann_vol,
        "bench_vol": bench_vol,
        "sharpe": sharpe,
        "bench_sharpe": bench_sharpe,
        "win_rate": win_rate,
        "plr": plr,
        "exp_r": exp_r,
        "cum_return": cum_return,
        "bench_cum_return": bench_cum_return,
        "alpha": alpha,
        "n_quarters": n,
    }


# ============ 主流程 ============
def main():
    print("=" * 60)
    print("机器学习选股策略 - TASK6")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载多股票数据...")
    df, stock_codes = load_all_stocks()

    # 2. 构建因子
    print("\n[2] 构建技术因子...")
    df_factors = build_factors(df)
    print(f"因子数据: {len(df_factors)} 行, {len(FACTOR_COLS)} 个因子")

    # 3. 划分训练集/测试集 (按时间)
    print("\n[3] 划分训练集/测试集...")
    df_clean = df_factors.dropna(subset=FACTOR_COLS + [TARGET_COL]).copy()
    # 2023-2025训练, 2025-2026测试
    df_clean["year"] = df_clean["trade_date"].astype(str).str[:4].astype(int)
    train_data = df_clean[df_clean["year"] <= 2024].copy()
    test_data = df_clean[df_clean["year"] >= 2025].copy()
    print(f"训练集: {len(train_data)} 行, 测试集: {len(test_data)} 行")

    X_train = train_data[FACTOR_COLS].values
    y_train = train_data[TARGET_COL].values
    X_test = test_data[FACTOR_COLS].values
    y_test = test_data[TARGET_COL].values

    # 4. 训练模型
    print("\n[4] 训练三个回归模型...")
    model_results = train_models(X_train, y_train, X_test, y_test)

    # 5. 季度选股回测
    print(f"\n[5] 季度选股回测 (Top {TOP_N})...")
    all_backtest = {}
    all_perf = {}

    for name, mr in model_results.items():
        print(f"\n  --- {name} ---")
        bt = backtest_strategy(df_clean, mr, TOP_N)
        perf = calc_performance(bt)
        all_backtest[name] = bt
        all_perf[name] = perf

        print(f"  季度数: {perf['n_quarters']}")
        print(f"  累计回报: 策略 {perf['cum_return']:+.2f}% vs 基准 {perf['bench_cum_return']:+.2f}%")
        print(f"  年化收益: 策略 {perf['ann_return']:+.2f}% vs 基准 {perf['bench_ann_return']:+.2f}%")
        print(f"  最大回撤: 策略 {perf['mdd']:.2f}% vs 基准 {perf['bench_mdd']:.2f}%")
        print(f"  夏普比率: 策略 {perf['sharpe']:.3f} vs 基准 {perf['bench_sharpe']:.3f}")
        print(f"  胜率: {perf['win_rate']:.1f}%")
        print(f"  盈亏比: {perf['plr']:.2f}")

    # 6. 特征重要性
    if model_results["随机森林"]["feature_importance"] is not None:
        print("\n[6] 随机森林因子重要性 (Top 10):")
        fi = model_results["随机森林"]["feature_importance"].head(10)
        for feat, imp in fi.items():
            print(f"  {feat:25s} {imp:.4f}")

    return model_results, all_backtest, all_perf, df_clean


if __name__ == "__main__":
    main()
