# -*- coding: utf-8 -*-
"""
plot_indicators.py
绘制四张技术指标图表：RSI、MACD、布林带、KDJ
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import platform
import os
import sys

# 添加当前目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from indicators import load_data, calc_all_indicators

# ============ 字体配置（跨平台自适应） ============
def _get_zh_font_path():
    """根据操作系统选择可用的中文字体文件路径"""
    sys_name = platform.system()
    if sys_name == "Windows":
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
        ]
    elif sys_name == "Darwin":
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

FONT_PATH = _get_zh_font_path()
if FONT_PATH:
    font_manager.fontManager.addfont(FONT_PATH)
    zh_font = font_manager.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.sans-serif"] = [zh_font.get_name()]
    plt.rcParams["font.family"] = zh_font.get_name()
else:
    zh_font = font_manager.FontProperties()
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                        "PingFang SC", "WenQuanYi Micro Hei",
                                        "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ============ 除权日标记（前复权数据已消除跳变，无需标注） ============
# 使用前复权数据后，指标曲线连续平滑，不再有除权断崖

# ============ 输出路径 ============
RSI_PNG = os.path.join(BASE_DIR, "002594_rsi.png")
MACD_PNG = os.path.join(BASE_DIR, "002594_macd.png")
BOLL_PNG = os.path.join(BASE_DIR, "002594_boll.png")
KDJ_PNG = os.path.join(BASE_DIR, "002594_kdj.png")


def add_ex_div_line(ax, y_pos_factor=0.95):
    """前复权数据无需除权日标注，此函数保留为空操作以保持调用兼容"""
    pass


def setup_xaxis(ax):
    """配置X轴日期格式"""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def plot_rsi(df):
    """图1: RSI 相对强弱指数"""
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)

    # RSI 曲线
    ax.plot(df["trade_date"], df["rsi"],
            color="#7B68EE", linewidth=1.3, label="RSI(14)")

    # 参考线
    ax.axhline(70, color="#FF0000", linestyle="--", linewidth=0.8,
               alpha=0.7, label="超买线(70)")
    ax.axhline(50, color="#999999", linestyle=":", linewidth=0.6,
               alpha=0.5, label="多空分界(50)")
    ax.axhline(30, color="#00AA00", linestyle="--", linewidth=0.8,
               alpha=0.7, label="超卖线(30)")

    # 超买超卖区域填充
    rsi_valid = df.dropna(subset=["rsi"])
    ax.fill_between(rsi_valid["trade_date"], 70, rsi_valid["rsi"],
                    where=rsi_valid["rsi"] > 70,
                    color="#FFB6C1", alpha=0.5, interpolate=True)
    ax.fill_between(rsi_valid["trade_date"], 30, rsi_valid["rsi"],
                    where=rsi_valid["rsi"] < 30,
                    color="#90EE90", alpha=0.5, interpolate=True)

    ax.set_ylim(0, 100)
    ax.set_title("图1  比亚迪(002594.SZ) RSI(14) 相对强弱指数",
                 fontproperties=zh_font, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("交易日期", fontproperties=zh_font, fontsize=11)
    ax.set_ylabel("RSI 值", fontproperties=zh_font, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(prop=zh_font, loc="upper right", framealpha=0.9, fontsize=9)

    setup_xaxis(ax)
    add_ex_div_line(ax, 0.95)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.savefig(RSI_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"图1 RSI 已保存: {RSI_PNG}")


def plot_macd(df):
    """图2: MACD 指数平滑异同移动平均线（双子图）"""
    fig = plt.figure(figsize=(11, 7), dpi=150)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.15)

    # 上方子图：DIF / DEA
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df["trade_date"], df["dif"],
             color="#2F7CE4", linewidth=1.2, label="DIF")
    ax1.plot(df["trade_date"], df["dea"],
             color="#FF9F40", linewidth=1.2, label="DEA")
    ax1.axhline(0, color="#999999", linestyle="-", linewidth=0.5)
    ax1.set_title("图2  比亚迪(002594.SZ) MACD(12,26,9) 指数平滑异同移动平均线",
                  fontproperties=zh_font, fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("DIF / DEA", fontproperties=zh_font, fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(prop=zh_font, loc="upper right", framealpha=0.9, fontsize=9)
    add_ex_div_line(ax1, 0.95)

    # 下方子图：MACD 柱状图
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    macd_colors = ["#C23531" if v >= 0 else "#2E8B57" for v in df["macd"]]
    ax2.bar(df["trade_date"], df["macd"], color=macd_colors,
            width=1.0, alpha=0.8, label="MACD柱")
    ax2.axhline(0, color="#999999", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("交易日期", fontproperties=zh_font, fontsize=11)
    ax2.set_ylabel("MACD 柱", fontproperties=zh_font, fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(prop=zh_font, loc="upper right", framealpha=0.9, fontsize=9)
    add_ex_div_line(ax2, 0.95)

    setup_xaxis(ax2)
    fig.autofmt_xdate(rotation=30)
    plt.savefig(MACD_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"图2 MACD 已保存: {MACD_PNG}")


def plot_boll(df):
    """图3: 布林带 Bollinger Bands"""
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

    boll_valid = df.dropna(subset=["boll_upper"])

    # 带宽填充
    ax.fill_between(boll_valid["trade_date"],
                    boll_valid["boll_upper"],
                    boll_valid["boll_lower"],
                    color="#2F7CE4", alpha=0.1, label="布林带通道")

    # 收盘价
    ax.plot(df["trade_date"], df["close"],
            color="#C23531", linewidth=1.2, label="收盘价")

    # 上轨/下轨/中轨
    ax.plot(boll_valid["trade_date"], boll_valid["boll_upper"],
            color="#2F7CE4", linewidth=0.9, linestyle="--", label="上轨(UB)")
    ax.plot(boll_valid["trade_date"], boll_valid["boll_mid"],
            color="#FF9F40", linewidth=0.9, linestyle="-.", label="中轨(MB)")
    ax.plot(boll_valid["trade_date"], boll_valid["boll_lower"],
            color="#2F7CE4", linewidth=0.9, linestyle="--", label="下轨(LB)")

    ax.set_title("图3  比亚迪(002594.SZ) 布林带(20,2) Bollinger Bands",
                 fontproperties=zh_font, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("交易日期", fontproperties=zh_font, fontsize=11)
    ax.set_ylabel("价格（元）", fontproperties=zh_font, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(prop=zh_font, loc="upper right", framealpha=0.9, fontsize=9)

    setup_xaxis(ax)
    add_ex_div_line(ax, 0.95)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.savefig(BOLL_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"图3 布林带 已保存: {BOLL_PNG}")


def plot_kdj(df):
    """图4: KDJ 随机指标"""
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

    ax.plot(df["trade_date"], df["k"],
            color="#2F7CE4", linewidth=1.2, label="K")
    ax.plot(df["trade_date"], df["d"],
            color="#FF9F40", linewidth=1.2, label="D")
    ax.plot(df["trade_date"], df["j"],
            color="#9B30FF", linewidth=1.0, label="J")

    # 参考线
    ax.axhline(80, color="#FF0000", linestyle="--", linewidth=0.8,
               alpha=0.6, label="超买线(80)")
    ax.axhline(50, color="#999999", linestyle=":", linewidth=0.6,
               alpha=0.4, label="中轴线(50)")
    ax.axhline(20, color="#00AA00", linestyle="--", linewidth=0.8,
               alpha=0.6, label="超卖线(20)")

    ax.set_title("图4  比亚迪(002594.SZ) KDJ(9,3,3) 随机指标",
                 fontproperties=zh_font, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("交易日期", fontproperties=zh_font, fontsize=11)
    ax.set_ylabel("K / D / J 值", fontproperties=zh_font, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(prop=zh_font, loc="upper right", framealpha=0.9, fontsize=9)

    setup_xaxis(ax)
    add_ex_div_line(ax, 0.95)
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.savefig(KDJ_PNG, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"图4 KDJ 已保存: {KDJ_PNG}")


def main():
    print("=" * 60)
    print("开始绘制技术指标图表")
    print("=" * 60)

    # 加载数据并计算指标
    df = load_data()
    df = calc_all_indicators(df)
    print(f"数据加载完成: {len(df)} 条记录\n")

    # 绘制四张图
    plot_rsi(df)
    plot_macd(df)
    plot_boll(df)
    plot_kdj(df)

    print(f"\n所有图表绘制完成！")
    print(f"输出目录: {BASE_DIR}")


if __name__ == "__main__":
    main()
