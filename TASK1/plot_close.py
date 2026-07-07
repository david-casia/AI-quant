# -*- coding: utf-8 -*-
"""
plot_close.py
绘制比亚迪(002594.SZ)过去一年每日收盘价曲线图
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import platform
import os

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
    # 回退：依赖 matplotlib 内置字体名匹配
    zh_font = font_manager.FontProperties()
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                        "PingFang SC", "WenQuanYi Micro Hei",
                                        "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ============ 路径 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")
PNG_PATH = os.path.join(BASE_DIR, "002594_close.png")


def main():
    # 读取数据
    df = pd.read_csv(CSV_PATH)
    # 转换日期格式
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)

    print(f"读取数据: {len(df)} 条记录")
    print(f"日期范围: {df['trade_date'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{df['trade_date'].iloc[-1].strftime('%Y-%m-%d')}")

    # ============ 绘图 ============
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

    # 收盘价曲线
    ax.plot(df["trade_date"], df["close"],
            color="#C23531", linewidth=1.2, label="收盘价")

    # 标题（含图号）
    ax.set_title("图1  比亚迪(002594.SZ)过去一年每日收盘价走势",
                 fontproperties=zh_font, fontsize=13, fontweight="bold",
                 pad=12)

    # 坐标轴标签
    ax.set_xlabel("交易日期", fontproperties=zh_font, fontsize=11)
    ax.set_ylabel("收盘价（元）", fontproperties=zh_font, fontsize=11)

    # X轴日期格式
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)

    # ============ 除权除息标注（前复权数据已消除跳变，无需标注） ============
    # 使用前复权数据后，价格曲线连续平滑，不再有除权断崖

    # 网格和图例
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(prop=zh_font, loc="upper right", framealpha=0.9)

    # Y轴留出空间
    y_min = df["close"].min()
    y_max = df["close"].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

    # 布局调整
    plt.tight_layout()

    # 保存
    plt.savefig(PNG_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"图表已保存至: {PNG_PATH}")

    # 输出统计信息
    print(f"\n收盘价统计:")
    print(f"  最高价: {df['close'].max():.2f} 元 "
          f"({df.loc[df['close'].idxmax(), 'trade_date'].strftime('%Y-%m-%d')})")
    print(f"  最低价: {df['close'].min():.2f} 元 "
          f"({df.loc[df['close'].idxmin(), 'trade_date'].strftime('%Y-%m-%d')})")
    print(f"  均价: {df['close'].mean():.2f} 元")
    print(f"  首日收盘: {df.iloc[0]['close']:.2f} 元")
    print(f"  末日收盘: {df.iloc[-1]['close']:.2f} 元")
    change = df.iloc[-1]["close"] - df.iloc[0]["close"]
    change_pct = change / df.iloc[0]["close"] * 100
    print(f"  区间涨跌: {change:+.2f} 元 ({change_pct:+.2f}%)")


if __name__ == "__main__":
    main()
