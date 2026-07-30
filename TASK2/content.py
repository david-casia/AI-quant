# -*- coding: utf-8 -*-
"""
content.py
TASK2 四个问题的理论文本内容 + 代码片段
"""

# ============ 问题一：数据诊断分析 ============
Q1_TITLE = "问题一：数据诊断分析"

# Q1 正文模板：使用 {占位符} 由 build_pdf.py 在运行时填充实际统计数据
# 这样即使重新拉取数据（日期范围或记录数变化），文本与统计表始终保持一致
Q1_BODY = """
数据诊断分析是量化交易研究的基础环节，旨在在使用数据进行技术指标计算和策略分析之前，全面了解数据的质量状况和统计特征。本次分析以比亚迪(002594.SZ)过去一年的前复权(qfq)日线交易数据为对象，共包含{n_records}个交易日的观测值（数据范围：{date_start}至{date_end}），数据字段包括交易日期(trade_date)、开盘价(open)、最高价(high)、最低价(low)、收盘价(close)、前收盘价(pre_close)、涨跌额(change)、涨跌幅(pct_chg)、成交量(vol)和成交额(amount)。

首先进行缺失值检查。通过Python的pandas库调用df.isnull().sum()方法对数据框中所有字段进行逐列扫描，结果显示全部{n_fields}个字段的缺失值数量均为0，总缺失值为0条，表明数据完整无缺失。这一结果说明Tushare数据接口提供的数据质量良好，无需进行缺失值填充或删除处理，可以直接用于后续的技术指标计算。

其次计算描述性统计量。利用df.describe()方法对开盘价、最高价、最低价、收盘价和成交量五个核心字段进行描述性统计分析，统计量包括计数(count)、均值(mean)、标准差(std)、最小值(min)、第一四分位数(25%)、中位数(50%)、第三四分位数(75%)和最大值(max)。从统计结果可以看出，收盘价的均值为{close_mean:.2f}元，标准差为{close_std:.2f}元，反映了该股票在观察期内价格波动幅度较大。最小值{close_min:.2f}元与最大值{close_max:.2f}元之间的差异体现了期间的价格波动范围。成交量的均值为{vol_mean:,.0f}手，标准差为{vol_std:,.0f}手，表明交易活跃度也存在较大波动。

本数据采用前复权（qfq）方式处理，已自动消除除权除息（如高送转、派息等公司行为）带来的价格跳变，使价格序列连续平滑。前复权是技术分析和量化回测的标准做法，能够保证技术指标计算结果的准确性和连续性。综上所述，该数据集完整性良好，无缺失值，描述性统计量真实反映了比亚迪股票在过去一年中的价格和成交量分布特征，可直接用于后续技术指标分析。
""".strip()

# ============ 问题二：技术指标理论解释 ============
Q2_TITLE = "问题二：基础技术指标理论解释——RSI、MACD、布林带"

Q2_BODY = """
（一）RSI 相对强弱指数（Relative Strength Index）

RSI是由技术分析大师韦尔斯·威尔德（J. Welles Wilder Jr.）于1978年提出的一种动量类技术指标，通过衡量一段时间内价格上涨幅度与下跌幅度的相对比例，来评估市场的超买超卖状态。RSI的取值范围为0至100，通常以14日作为计算周期。

RSI的计算方法如下：首先计算每个交易日与前一日收盘价的变动值ΔP，将正向变动记为收益（Gain），负向变动取绝对值记为损失（Loss）。然后采用Wilder平滑法计算平均收益和平均损失，其递推公式为：AvgGain_t = α × Gain_t + (1-α) × AvgGain_{t-1}，其中α = 1/周期数（14日周期时α = 1/14）。进而计算相对强度RS = AvgGain / AvgLoss，最终RSI = 100 - 100/(1+RS)。当AvgLoss为0时（即连续上涨无下跌），RSI取值100。

RSI的主要作用在于判断市场的超买超卖状态。一般而言，RSI高于70时视为超买区间，表明市场可能过度乐观，存在回调风险；RSI低于30时视为超卖区间，表明市场可能过度悲观，存在反弹机会。此外，RSI的背离现象（价格创新高但RSI未创新高，或反之）也是重要的趋势反转信号。RSI还可以用于识别中长期趋势的方向，RSI在50以上通常表示多头占优，50以下表示空头占优。

（二）MACD 指数平滑异同移动平均线（Moving Average Convergence Divergence）

MACD是由杰拉尔德·阿佩尔（Gerald Appel）于1970年代提出的一种趋势类技术指标，通过计算两条不同周期的指数移动平均线（EMA）之间的差值，来捕捉价格趋势的方向和动量变化。MACD的标准参数为12日（快线）、26日（慢线）和9日（信号线）。

MACD的计算方法如下：首先分别计算收盘价的12日EMA和26日EMA，两者之差即为DIF（差离值，也称MACD线）：DIF = EMA(12) - EMA(26)。然后对DIF计算9日EMA，得到DEA（信号线）：DEA = EMA(DIF, 9)。最后计算MACD柱状图（也称MACD能量柱）：MACD柱 = (DIF - DEA) × 2。其中EMA的计算采用递推公式：EMA_t = α × Price_t + (1-α) × EMA_{t-1}，α = 2/(N+1)，N为周期数。

MACD的主要作用在于判断价格趋势的方向和转折点。DIF上穿DEA称为"金叉"，通常视为买入信号；DIF下穿DEA称为"死叉"，通常视为卖出信号。DIF和DEA均在零轴上方时表示多头市场，均在零轴下方时表示空头市场。MACD柱状图反映了DIF与DEA之间差距的变化速度，柱状图由负转正表示动量由空转多，由正转负表示动量由多转空。此外，MACD的背离现象（价格与指标走势不一致）也是重要的趋势预警信号。

（三）布林带（Bollinger Bands）

布林带是由约翰·布林格（John Bollinger）于1980年代提出的一种波动率类技术指标，通过构建以移动平均线为中心、以标准差为宽度的价格通道，来衡量价格的相对高低和波动率变化。布林带的标准参数为20日周期和2倍标准差。

布林带的计算方法如下：中轨（Middle Band）为收盘价的20日简单移动平均线：MB = SMA(Close, 20)。然后计算20日收盘价的标准差σ（通常采用总体标准差，ddof=0）。上轨（Upper Band）和下轨（Lower Band）分别为：UB = MB + 2σ，LB = MB - 2σ。上下轨之间的距离构成布林带通道的宽度，反映了市场波动率的大小。

布林带的主要作用在于评估价格的相对位置和波动率变化。当价格触及或突破上轨时，可能处于超买状态；当价格触及或跌破下轨时，可能处于超卖状态。布林带通道的收窄（即带宽缩小）通常预示着市场即将出现大幅波动（即" squeeze"效应），而通道的扩张则表明波动率正在增加。此外，价格沿上轨或下轨运行时，往往表示当前趋势较强。布林带还可以与其他指标配合使用，如价格从下轨反弹并突破中轨时视为多头信号，从上轨回落并跌破中轨时视为空头信号。
""".strip()

# ============ 问题三标题（内容由 build_pdf.py 动态生成） ============
Q3_TITLE = "问题三：Python编程实现——技术指标计算与可视化"

# ============ 问题四：KDJ 扩展指标 ============
Q4_TITLE = "问题四：扩展指标——KDJ 随机指标"

Q4_BODY = """
KDJ随机指标是金融市场中最常用的短线技术指标之一，由乔治·莱恩（George Lane）提出。与RSI仅使用收盘价不同，KDJ综合考虑了最高价、最低价和收盘价三个价格信息，因此能够更全面地反映价格在一定周期内的相对位置。KDJ指标由K线、D线和J线三条曲线组成，标准参数为9日周期、K值3日平滑、D值3日平滑（即9,3,3参数）。

KDJ的计算方法如下：首先计算未成熟随机值（RSV），公式为RSV(n) = (Close - min(Low, n)) / (max(High, n) - min(Low, n)) × 100，其中n通常取9日。RSV反映了当日收盘价在过去n日价格区间中的相对位置，取值范围为0至100。然后对RSV进行指数平滑得到K值：K_t = (2/3) × K_{t-1} + (1/3) × RSV_t，初始K值通常取50。再对K值进行指数平滑得到D值：D_t = (2/3) × D_{t-1} + (1/3) × K_t，初始D值也取50。最后计算J值：J = 3K - 2D，J值是K值和D值的线性组合，反映了两者的偏离程度，其取值可以超过100或低于0。

KDJ的主要作用在于判断市场的超买超卖状态和短期转折信号。一般而言，K值或D值高于80时视为超买区间，低于20时视为超卖区间。K线上穿D线称为"金叉"，为买入信号；K线下穿D线称为"死叉"，为卖出信号。J值的变化更为灵敏，当J值超过100时表示极度超买，低于0时表示极度超卖。与RSI相比，KDJ由于引入了最高价和最低价信息，对价格的短期波动更为敏感，适合短线交易参考，但也因此容易产生虚假信号，通常需要结合其他指标综合判断。

除了RSI、MACD、布林带和KDJ之外，金融市场中还有许多其他典型的技术指标。常见的包括：移动平均线（MA），用于识别中长期趋势方向；平均真实波幅（ATR），用于衡量市场波动率；能量潮指标（OBV），基于成交量分析资金流向；威廉指标（Williams %R），衡量收盘价在一定周期内的相对位置；商品通道指数（CCI），测量价格偏离统计平均值的程度；趋向指标（DMI），判断趋势的方向和强度等。这些指标各有侧重，在实际应用中往往需要多种指标配合使用，形成互补的技术分析体系。
""".strip()

# ============ 代码片段常量（展示在PDF中） ============

LOAD_CODE = """import pandas as pd
import numpy as np

# 加载已存储的股价数据
df = pd.read_csv('002594_daily.csv')
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values('trade_date').reset_index(drop=True)
print(f'共加载 {len(df)} 条日线数据')"""

RSI_CODE = """# RSI 相对强弱指数（14日，Wilder平滑法）
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder平滑: alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False,
                        min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False,
                        min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)  # 全涨无跌时RSI=100
    return rsi

df['rsi'] = calc_rsi(df['close'])"""

MACD_CODE = """# MACD 指数平滑异同移动平均线（12,26,9）
def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow           # 差离值
    dea = dif.ewm(span=signal, adjust=False).mean()  # 信号线
    macd_hist = (dif - dea) * 2         # MACD柱
    return dif, dea, macd_hist

df['dif'], df['dea'], df['macd'] = calc_macd(df['close'])"""

BOLL_CODE = """# 布林带 Bollinger Bands（20日，2倍标准差）
def calc_boll(close, period=20, num_std=2):
    mid = close.rolling(period).mean()           # 中轨 SMA
    std = close.rolling(period).std(ddof=0)      # 总体标准差
    upper = mid + num_std * std                   # 上轨
    lower = mid - num_std * std                   # 下轨
    return upper, mid, lower

df['boll_upper'], df['boll_mid'], df['boll_lower'] = \\
    calc_boll(df['close'])"""

KDJ_CODE = """# KDJ 随机指标（9,3,3）
def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    low_n = low.rolling(n).min()
    high_n = high.rolling(n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)  # 防除零
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()  # K值
    d = k.ewm(alpha=1/m2, adjust=False).mean()    # D值
    j = 3 * k - 2 * d                              # J值
    return k, d, j

df['k'], df['d'], df['j'] = calc_kdj(
    df['high'], df['low'], df['close'])"""
