# -*- coding: utf-8 -*-
"""
fetch_multi_stocks.py
批量获取50只A股前复权日线数据，用于TASK6多股票选股策略
"""
import os
import time
import pandas as pd
import tushare as ts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "stock_data")
os.makedirs(DATA_DIR, exist_ok=True)

TS_TOKEN = os.environ.get("TUSHARE_TOKEN", "8e30265bd9e9d5264093844281c8abb3123772e96ece890c08f15b9f")
ts.set_token(TS_TOKEN)

# 50只代表性A股 (覆盖多个行业)
STOCKS = [
    # 银行金融
    "600519.SH",  # 贵州茅台
    "000858.SZ",  # 五粮液
    "601318.SH",  # 中国平安
    "600036.SH",  # 招商银行
    "601166.SH",  # 兴业银行
    "000333.SZ",  # 美的集团
    "000651.SZ",  # 格力电器
    "600276.SH",  # 恒瑞医药
    "000725.SZ",  # 京东方A
    "002415.SZ",  # 海康威视
    # 新能源
    "002594.SZ",  # 比亚迪
    "300750.SZ",  # 宁德时代
    "601012.SH",  # 隆基绿能
    "002129.SZ",  # TCL中环
    "600438.SH",  # 通威股份
    # 消费
    "600887.SH",  # 伊利股份
    "000568.SZ",  # 泸州老窖
    "600690.SH",  # 海尔智家
    "603288.SH",  # 海天味业
    "002714.SZ",  # 牧原股份
    # 科技
    "002230.SZ",  # 科大讯飞
    "300059.SZ",  # 东方财富
    "300015.SZ",  # 爱尔眼科
    "002241.SZ",  # 歌尔股份
    "300760.SZ",  # 迈瑞医疗
    # 周期
    "601899.SH",  # 紫金矿业
    "601600.SH",  # 中国铝业
    "600028.SH",  # 中国石化
    "601857.SH",  # 中国石油
    "600019.SH",  # 宝钢股份
    # 基建
    "601668.SH",  # 中国建筑
    "601390.SH",  # 中国中铁
    "601186.SH",  # 中国铁建
    "600048.SH",  # 保利发展
    "600340.SH",  # 华夏幸福
    # 电子
    "002475.SZ",  # 立讯精密
    "300433.SZ",  # 蓝思科技
    "002241.SZ",  # 歌尔股份
    "603160.SH",  # 汇顶科技
    "300124.SZ",  # 汇川技术
    # 医药
    "600436.SH",  # 片仔癀
    "000538.SZ",  # 云南白药
    "300003.SZ",  # 乐普医疗
    "002007.SZ",  # 华兰生物
    "300015.SZ",  # 爱尔眼科
    # 汽车
    "600104.SH",  # 上汽集团
    "601633.SH",  # 长城汽车
    "000625.SZ",  # 长安汽车
    # 其他
    "600009.SH",  # 上海机场
    "601888.SH",  # 中国中免
]

# 去重
STOCKS = list(dict.fromkeys(STOCKS))

START_DATE = "20230101"
END_DATE = "20260718"


def fetch_all():
    pro = ts.pro_api(TS_TOKEN)
    success = 0
    fail = 0

    for i, ts_code in enumerate(STOCKS, 1):
        csv_path = os.path.join(DATA_DIR, f"{ts_code.replace('.', '_')}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            if len(existing) > 200:
                print(f"  [{i}/{len(STOCKS)}] {ts_code} 已存在 ({len(existing)}行), 跳过")
                success += 1
                continue

        try:
            # 使用 pro.daily() 接口(50次/分钟), 避免adj_factor频率限制(1次/分钟)
            df = pro.daily(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE,
            )
            if df is None or len(df) == 0:
                print(f"  [{i}/{len(STOCKS)}] {ts_code} 无数据")
                fail += 1
                continue

            df = df.sort_values("trade_date").reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            print(f"  [{i}/{len(STOCKS)}] {ts_code} OK ({len(df)}行)")
            success += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"  [{i}/{len(STOCKS)}] {ts_code} FAIL: {str(e)[:80]}")
            fail += 1
            time.sleep(1)

    print(f"\n完成: 成功{success} 失败{fail} 共{len(STOCKS)}")
    return success


if __name__ == "__main__":
    print("=" * 60)
    print(f"批量获取 {len(STOCKS)} 只A股前复权日线数据")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print(f"存储目录: {DATA_DIR}")
    print("=" * 60)
    fetch_all()
