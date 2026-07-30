# -*- coding: utf-8 -*-
"""
fetch_data.py
获取比亚迪(002594.SZ)过去一年每个交易日的日线数据
三级降级策略：pro_bar → pro.daily → akshare
"""
import tushare as ts
import akshare as ak
import pandas as pd
import datetime
import os
import sys

# ============ 配置 ============
# Token 优先从环境变量 TUSHARE_TOKEN 读取，未设置时回退到内置默认值
# 生产环境建议通过 `export TUSHARE_TOKEN=xxx`（Linux）或 `set TUSHARE_TOKEN=xxx`（Windows）注入
TS_TOKEN = os.environ.get("TUSHARE_TOKEN", "8e30265bd9e9d5264093844281c8abb3123772e96ece890c08f15b9f")
TS_CODE = "002594.SZ"
AK_SYMBOL = "002594"

# 过去一年日期范围
end_date = datetime.date.today().strftime("%Y%m%d")
start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y%m%d")

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "002594_daily.csv")

# 统一字段名（akshare中文→英文映射）
FIELD_MAP = {
    "日期": "trade_date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "vol",
    "成交额": "amount",
    "振幅": "pct_chg",
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
    "换手率": "turnover",
}


def fetch_pro_bar():
    """方案A: tushare pro_bar 日线数据（120积分可用）"""
    print(f"[方案A] 尝试 ts.pro_bar ...")
    ts.set_token(TS_TOKEN)
    df = ts.pro_bar(
        ts_code=TS_CODE,
        start_date=start_date,
        end_date=end_date,
        adj="qfq",   # 前复权（消除除权除息跳变，技术分析标准做法）
        freq="D",
    )
    if df is None or len(df) == 0:
        raise Exception("pro_bar 返回空数据")
    df = df.sort_values("trade_date").reset_index(drop=True)
    print(f"[方案A] 成功！共 {len(df)} 条记录")
    return df, "tushare pro_bar"


def fetch_pro_daily():
    """方案B: tushare pro.daily（需2000积分）"""
    print(f"[方案B] 尝试 pro.daily ...")
    pro = ts.pro_api(TS_TOKEN)
    df = pro.daily(
        ts_code=TS_CODE,
        start_date=start_date,
        end_date=end_date,
    )
    if df is None or len(df) == 0:
        raise Exception("pro.daily 返回空数据")
    df = df.sort_values("trade_date").reset_index(drop=True)
    print(f"[方案B] 成功！共 {len(df)} 条记录")
    return df, "tushare pro.daily"


def fetch_akshare():
    """方案C: akshare 免费兜底"""
    print(f"[方案C] 尝试 akshare.stock_zh_a_hist ...")
    df = ak.stock_zh_a_hist(
        symbol=AK_SYMBOL,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or len(df) == 0:
        raise Exception("akshare 返回空数据")
    # 中文列名映射为英文
    df = df.rename(columns=FIELD_MAP)
    df["ts_code"] = TS_CODE
    # 确保日期格式统一为 YYYYMMDD
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    print(f"[方案C] 成功！共 {len(df)} 条记录")
    return df, "akshare stock_zh_a_hist"


def main():
    print(f"=" * 60)
    print(f"目标股票: {TS_CODE} (比亚迪)")
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"=" * 60)

    # 三级降级
    for fn in [fetch_pro_bar, fetch_pro_daily, fetch_akshare]:
        try:
            df, source = fn()
            break
        except Exception as e:
            print(f"[失败] {fn.__name__}: {e}")
            print()
    else:
        print("所有数据源均失败！")
        sys.exit(1)

    # 保存 CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n数据已保存至: {OUTPUT_CSV}")
    print(f"数据来源: {source}")
    print(f"记录数: {len(df)}")
    print(f"字段: {list(df.columns)}")
    print(f"\n前5行预览:")
    print(df.head().to_string())
    print(f"\n后5行预览:")
    print(df.tail().to_string())

    # 输出统计信息供后续使用
    if "close" in df.columns:
        print(f"\n收盘价统计:")
        print(f"  最高: {df['close'].max():.2f}")
        print(f"  最低: {df['close'].min():.2f}")
        print(f"  均值: {df['close'].mean():.2f}")
        print(f"  首日: {df.iloc[0]['trade_date']} -> {df.iloc[0]['close']}")
        print(f"  末日: {df.iloc[-1]['trade_date']} -> {df.iloc[-1]['close']}")
        change_pct = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100
        print(f"  年度涨跌幅: {change_pct:+.2f}%")


if __name__ == "__main__":
    main()
