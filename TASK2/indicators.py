# -*- coding: utf-8 -*-
"""
indicators.py
技术指标计算模块（手动实现，无依赖 ta-lib）
包含：RSI、MACD、布林带(Bollinger Bands)、KDJ
"""
import pandas as pd
import numpy as np
import os

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")


def load_data(csv_path=CSV_PATH):
    """加载股价数据，返回排序后的 DataFrame"""
    df = pd.read_csv(csv_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def calc_rsi(close, period=14):
    """
    RSI 相对强弱指数（Wilder 平滑法）
    公式：
      ΔP = Close_t - Close_{t-1}
      Gain = max(ΔP, 0), Loss = max(-ΔP, 0)
      AvgGain = α·Gain + (1-α)·AvgGain_prev   (α = 1/period, Wilder平滑)
      RS = AvgGain / AvgLoss
      RSI = 100 - 100/(1+RS)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder 平滑等价于 ewm(alpha=1/period, adjust=False)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # 当 AvgLoss=0 时（全涨无跌），RSI=100
    rsi = rsi.where(avg_loss != 0, 100)
    return rsi


def calc_macd(close, fast=12, slow=26, signal=9):
    """
    MACD 指数平滑异同移动平均线
    公式：
      DIF = EMA(fast) - EMA(slow)
      DEA = EMA(DIF, signal)
      MACD柱 = (DIF - DEA) × 2
    返回：(dif, dea, macd_hist)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist


def calc_boll(close, period=20, num_std=2):
    """
    布林带 Bollinger Bands
    公式：
      中轨 MB = SMA(close, period)
      标准差 σ = STD(close, period)  (总体标准差 ddof=0)
      上轨 UB = MB + num_std × σ
      下轨 LB = MB - num_std × σ
    返回：(upper, mid, lower)
    """
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    """
    KDJ 随机指标
    公式：
      RSV(n) = (Close - min(Low,n)) / (max(High,n) - min(Low,n)) × 100
      K_t = (2/3)·K_{t-1} + (1/3)·RSV_t   (初始K=50)
      D_t = (2/3)·D_{t-1} + (1/3)·K_t       (初始D=50)
      J = 3K - 2D
    返回：(k, d, j)
    """
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    denom = high_n - low_n
    rsv = (close - low_n) / denom * 100
    # 防除零 + 初始值设定：当9日内最高=最低（窗口未填满或价格无波动）时
    # RSV取50（中性值），使K/D初始值恰好为50，符合KDJ传统定义
    rsv = rsv.fillna(50)
    # K、D 的递推等价于 ewm(alpha=1/m1, adjust=False)，初始值50
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_all_indicators(df):
    """计算所有技术指标，添加到 DataFrame 中"""
    df = df.copy()
    df["rsi"] = calc_rsi(df["close"])
    df["dif"], df["dea"], df["macd"] = calc_macd(df["close"])
    df["boll_upper"], df["boll_mid"], df["boll_lower"] = calc_boll(df["close"])
    df["k"], df["d"], df["j"] = calc_kdj(df["high"], df["low"], df["close"])
    return df


def data_diagnostics(df):
    """数据诊断分析：缺失值检查 + 描述性统计量"""
    print("=" * 60)
    print("数据诊断分析")
    print("=" * 60)

    # 缺失值检查
    print("\n【缺失值检查】")
    missing = df.isnull().sum()
    print(f"总缺失值数量: {missing.sum()}")
    if missing.sum() > 0:
        print("各字段缺失值:")
        print(missing[missing > 0])
    else:
        print("所有字段均无缺失值。")

    # 描述性统计量
    print("\n【描述性统计量】")
    stat_cols = ["open", "high", "low", "close", "vol"]
    available_cols = [c for c in stat_cols if c in df.columns]
    desc = df[available_cols].describe()
    print(desc.round(2).to_string())

    print(f"\n数据记录数: {len(df)}")
    print(f"日期范围: {df['trade_date'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{df['trade_date'].iloc[-1].strftime('%Y-%m-%d')}")

    return desc


if __name__ == "__main__":
    df = load_data()
    desc = data_diagnostics(df)

    # 计算所有指标
    df = calc_all_indicators(df)

    # 输出指标统计
    print("\n" + "=" * 60)
    print("技术指标计算结果")
    print("=" * 60)

    print("\n【RSI 统计】")
    rsi_valid = df["rsi"].dropna()
    print(f"  有效值: {len(rsi_valid)}")
    print(f"  最大值: {rsi_valid.max():.2f}")
    print(f"  最小值: {rsi_valid.min():.2f}")
    print(f"  末值: {df['rsi'].iloc[-1]:.2f}")
    print(f"  超买(>70)天数: {(rsi_valid > 70).sum()}")
    print(f"  超卖(<30)天数: {(rsi_valid < 30).sum()}")

    print("\n【MACD 统计】")
    print(f"  末日 DIF: {df['dif'].iloc[-1]:.4f}")
    print(f"  末日 DEA: {df['dea'].iloc[-1]:.4f}")
    print(f"  末日 MACD柱: {df['macd'].iloc[-1]:.4f}")
    print(f"  DIF最大值: {df['dif'].max():.4f}")
    print(f"  DIF最小值: {df['dif'].min():.4f}")

    print("\n【布林带统计】")
    boll_valid = df.dropna(subset=["boll_upper"])
    print(f"  有效值: {len(boll_valid)}")
    print(f"  末日上轨: {df['boll_upper'].iloc[-1]:.2f}")
    print(f"  末日中轨: {df['boll_mid'].iloc[-1]:.2f}")
    print(f"  末日下轨: {df['boll_lower'].iloc[-1]:.2f}")
    print(f"  末日收盘价: {df['close'].iloc[-1]:.2f}")

    print("\n【KDJ 统计】")
    k_valid = df["k"].dropna()
    print(f"  有效值: {len(k_valid)}")
    print(f"  末日 K: {df['k'].iloc[-1]:.2f}")
    print(f"  末日 D: {df['d'].iloc[-1]:.2f}")
    print(f"  末日 J: {df['j'].iloc[-1]:.2f}")
    print(f"  超买(K>80)天数: {(k_valid > 80).sum()}")
    print(f"  超卖(K<20)天数: {(k_valid < 20).sum()}")

    # 保存带指标的完整数据
    output_csv = os.path.join(BASE_DIR, "002594_with_indicators.csv")
    df_out = df.copy()
    df_out["trade_date"] = df_out["trade_date"].dt.strftime("%Y%m%d")
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n带指标数据已保存至: {output_csv}")
