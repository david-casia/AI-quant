# -*- coding: utf-8 -*-
"""
build_pdf.py
生成 TASK2 PDF 文档（weasyprint）
格式：宋体(Noto Serif CJK SC)、五号字(10.5pt)、1.5倍行距、0段间距、两端对齐

注意：weasyprint 在 Windows 上需要 GTK 运行时（libgobject-2.0）。
若环境缺少 GTK，脚本仍可正常 import 并调用各 build_xxx_analysis() 函数，
仅在执行 main() 生成 PDF 时报错。
"""
import os
import sys
import pandas as pd
import numpy as np

try:
    from weasyprint import HTML, CSS
    _HAS_WEASYPRINT = True
except OSError:
    # Windows 缺少 GTK 运行时会触发 OSError，降级处理
    _HAS_WEASYPRINT = False
    HTML = CSS = None

# 添加当前目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from indicators import load_data, calc_all_indicators, data_diagnostics
from content import (Q1_TITLE, Q1_BODY, Q2_TITLE, Q2_BODY,
                     Q3_TITLE, Q4_TITLE, Q4_BODY,
                     LOAD_CODE, RSI_CODE, MACD_CODE, BOLL_CODE, KDJ_CODE)

# ============ 配置 ============
STUDENT_NAME = "薛德刚"
PDF_NAME = f"{STUDENT_NAME}+TASK2.pdf"
PDF_PATH = os.path.join(BASE_DIR, PDF_NAME)
CSS_PATH = os.path.join(BASE_DIR, "style.css")
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")

# 图片路径
RSI_PNG = "002594_rsi.png"
MACD_PNG = "002594_macd.png"
BOLL_PNG = "002594_boll.png"
KDJ_PNG = "002594_kdj.png"

# 除权日（前复权数据已消除跳变，保留变量供兼容，实际不使用）
EX_DIV_DATE = None


def text_to_html_paragraphs(text):
    """纯文本转HTML段落（每段一个<p>标签）"""
    paragraphs = text.strip().split("\n")
    html = ""
    for para in paragraphs:
        para = para.strip()
        if para:
            para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f"<p>{para}</p>\n"
    return html


def escape_code(code):
    """转义代码中的HTML特殊字符"""
    return code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_describe_table():
    """生成描述性统计量HTML表格（表1）"""
    df = load_data()
    stat_cols = ["open", "high", "low", "close", "vol"]
    desc = df[stat_cols].describe()

    # 中文表头映射
    header_map = {
        "open": "开盘价", "high": "最高价", "low": "最低价",
        "close": "收盘价", "vol": "成交量(手)"
    }
    stat_names = {
        "count": "计数", "mean": "均值", "std": "标准差",
        "min": "最小值", "25%": "第一四分位",
        "50%": "中位数", "75%": "第三四分位", "max": "最大值"
    }

    html = '<table class="data">\n'
    html += '<caption>表1 比亚迪(002594.SZ)描述性统计量</caption>\n'
    html += "<thead><tr><th>统计量</th>"
    for col in stat_cols:
        html += f"<th>{header_map[col]}</th>"
    html += "</tr></thead>\n<tbody>\n"

    for stat_name in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        html += f"<tr><td>{stat_names[stat_name]}</td>"
        for col in stat_cols:
            val = desc.loc[stat_name, col]
            if stat_name == "count":
                html += f"<td>{int(val)}</td>"
            elif col == "vol":
                html += f"<td>{val:,.0f}</td>"
            else:
                html += f"<td>{val:.2f}</td>"
        html += "</tr>\n"

    html += "</tbody></table>\n"
    return html


def build_rsi_analysis(df):
    """RSI 图表解读（前复权数据，动态生成）"""
    rsi = df["rsi"].dropna()
    last_rsi = df["rsi"].iloc[-1]
    max_rsi = rsi.max()
    min_rsi = rsi.min()
    overbought = (rsi > 70).sum()
    oversold = (rsi < 30).sum()

    # 末值状态
    if last_rsi > 70:
        status = "处于超买区间（>70），短期存在回调风险"
    elif last_rsi < 30:
        status = "处于超卖区间（<30），短期存在反弹可能"
    else:
        status = f"处于中性区域（{last_rsi:.2f}），多空力量相对均衡"

    analysis = f"""从图1可以看出，比亚迪(002594.SZ)的RSI(14)指标在过去一年中呈现出显著的波动特征。RSI最大值为{max_rsi:.2f}，最小值为{min_rsi:.2f}，末值为{last_rsi:.2f}，{status}。在观察期内，RSI进入超买区间（>70）的天数为{overbought}天，进入超卖区间（<30）的天数为{oversold}天。

由于采用了前复权数据，RSI指标的计算不受除权除息跳变的干扰，能够真实反映市场多空力量的变化。从走势来看，RSI在超买与超卖区间之间规律性交替，指标信号具有较高的参考价值。当RSI从超卖区反弹突破50时，往往预示着短期价格企稳回升；当RSI从超买区回落跌破50时，则提示短期价格面临调整压力。"""

    return analysis.strip()


def build_macd_analysis(df):
    """MACD 图表解读（前复权数据，动态生成）"""
    last_dif = df["dif"].iloc[-1]
    last_dea = df["dea"].iloc[-1]
    last_macd = df["macd"].iloc[-1]
    max_dif = df["dif"].max()
    min_dif = df["dif"].min()

    # 金叉死叉判断
    if last_dif > last_dea:
        cross_status = "DIF位于DEA之上（金叉状态），短期动量偏多"
    else:
        cross_status = "DIF位于DEA之下（死叉状态），短期动量偏空"

    # 零轴判断（分别考察DIF和DEA的符号，避免误判）
    if last_dif > 0 and last_dea > 0:
        zero_status = "DIF和DEA均在零轴上方，整体处于多头格局"
    elif last_dif < 0 and last_dea < 0:
        zero_status = "DIF和DEA均在零轴下方，整体处于空头格局"
    else:
        zero_status = "DIF与DEA在零轴附近交错，趋势方向尚不明确，需结合后续走势确认"

    analysis = f"""从图2可以看出，比亚迪(002594.SZ)的MACD(12,26,9)指标在过去一年中经历了显著的趋势变化。末日DIF值为{last_dif:.4f}，DEA值为{last_dea:.4f}，MACD柱为{last_macd:.4f}，{cross_status}。DIF的最大值为{max_dif:.4f}，最小值为{min_dif:.4f}，振幅较大。

由于采用了前复权数据，MACD指标的计算基于连续平滑的价格序列，金叉死叉信号和零轴穿越均具有真实的分析意义。从走势来看，DIF与DEA的交叉较好地捕捉了价格趋势的转折点，MACD柱状图的红绿交替清晰反映了多空动量的切换节奏。{zero_status}，后续需持续关注MACD柱的变化方向以确认趋势的延续性。"""

    return analysis.strip()


def build_boll_analysis(df):
    """布林带图表解读（前复权数据，动态生成）"""
    boll_valid = df.dropna(subset=["boll_upper"])
    last_close = df["close"].iloc[-1]
    last_upper = df["boll_upper"].iloc[-1]
    last_mid = df["boll_mid"].iloc[-1]
    last_lower = df["boll_lower"].iloc[-1]

    # %B 计算（%B = (Close - LB) / (UB - LB)，反映收盘价在通道中的相对位置）
    # %B > 100 表示突破上轨，< 0 表示跌破下轨，= 50 表示位于中轨
    pct_b = (last_close - last_lower) / (last_upper - last_lower) * 100

    # %B 解读
    if pct_b > 100:
        pct_b_hint = "大于100，价格突破上轨，短期超买"
    elif pct_b < 0:
        pct_b_hint = "小于0，价格跌破下轨，短期超卖"
    elif pct_b > 50:
        pct_b_hint = "介于50至100之间，位于中轨上方，多头偏强"
    else:
        pct_b_hint = "介于0至50之间，位于中轨下方，空头偏强"

    # 价格相对通道位置
    if last_close > last_upper:
        position = "突破上轨，处于超强状态"
    elif last_close < last_lower:
        position = "跌破下轨，处于超弱状态"
    elif last_close > last_mid:
        position = "位于中轨与上轨之间，多头偏强"
    else:
        position = "位于下轨与中轨之间，空头偏强"

    # 触及上下轨次数
    touch_upper = (df["close"] > df["boll_upper"]).sum()
    touch_lower = (df["close"] < df["boll_lower"]).sum()

    analysis = f"""从图3可以看出，比亚迪(002594.SZ)的布林带(20,2)指标在过去一年中呈现出合理的扩张与收敛交替特征。末日收盘价为{last_close:.2f}元，上轨为{last_upper:.2f}元，中轨为{last_mid:.2f}元，下轨为{last_lower:.2f}元，{position}。布林带%B指标（收盘价在通道中的相对位置）为{pct_b:.1f}%，{pct_b_hint}。%B指标的取值含义为：大于100表示价格突破上轨（超买），小于0表示价格跌破下轨（超卖），等于50表示价格恰好位于中轨。

由于采用了前复权数据，布林带的计算基于连续平滑的价格序列，通道宽度真实反映了市场波动率的变化。在观察期内，收盘价触及或突破上轨的次数为{touch_upper}次，触及或跌破下轨的次数为{touch_lower}次。通道收窄（squeeze）现象通常预示着市场即将出现趋势性突破，投资者可结合带宽变化与价格突破方向判断后续走势。"""

    return analysis.strip()


def build_kdj_analysis(df):
    """KDJ 图表解读（前复权数据，动态生成）"""
    last_k = df["k"].iloc[-1]
    last_d = df["d"].iloc[-1]
    last_j = df["j"].iloc[-1]

    k_valid = df["k"].dropna()
    overbought = (k_valid > 80).sum()
    oversold = (k_valid < 20).sum()

    # 金叉死叉
    if last_k > last_d:
        cross_status = "K线位于D线之上（金叉状态），短期偏向多头"
    else:
        cross_status = "K线位于D线之下（死叉状态），短期偏向空头"

    # J值状态
    if last_j > 100:
        j_status = "J值超过100，处于极度超买状态"
    elif last_j < 0:
        j_status = "J值低于0，处于极度超卖状态"
    elif last_j > 80:
        j_status = "J值处于超买区域"
    elif last_j < 20:
        j_status = "J值处于超卖区域"
    else:
        j_status = f"J值为{last_j:.2f}，处于正常区间"

    analysis = f"""从图4可以看出，比亚迪(002594.SZ)的KDJ(9,3,3)随机指标在过去一年中波动较为剧烈。末日K值为{last_k:.2f}，D值为{last_d:.2f}，J值为{last_j:.2f}，{cross_status}，{j_status}。在观察期内，K值进入超买区间（>80）的天数为{overbought}天，进入超卖区间（<20）的天数为{oversold}天。

由于采用了前复权数据，KDJ指标的计算不受除权跳变的干扰，金叉死叉信号和超买超卖判断均具有真实的分析意义。与RSI相比，KDJ指标由于引入了最高价和最低价信息，对短期价格波动的敏感性更强，信号更为频繁。在实际应用中，建议将KDJ与RSI、MACD等指标配合使用，通过多指标交叉验证降低虚假信号的风险。"""

    return analysis.strip()


def build_q1_body(df_with_indicators):
    """根据实际数据填充 Q1_BODY 模板中的统计占位符
    参数为已计算指标的 DataFrame，但统计基于原始行情字段。
    """
    # 原始行情字段（不含技术指标列），用于字段计数和缺失值检查
    original_cols = [c for c in df_with_indicators.columns
                     if c not in ("rsi", "dif", "dea", "macd",
                                  "boll_upper", "boll_mid", "boll_lower",
                                  "k", "d", "j")]
    df_orig = df_with_indicators[original_cols]

    # 缺失值检查（基于原始字段）
    n_missing = int(df_orig.isnull().sum().sum())
    n_fields = len(original_cols)

    # 描述性统计
    desc = df_orig[["open", "high", "low", "close", "vol"]].describe()
    close_mean = desc.loc["mean", "close"]
    close_std = desc.loc["std", "close"]
    close_min = desc.loc["min", "close"]
    close_max = desc.loc["max", "close"]
    vol_mean = desc.loc["mean", "vol"]
    vol_std = desc.loc["std", "vol"]

    date_start = df_orig.iloc[0]["trade_date"].strftime("%Y-%m-%d")
    date_end = df_orig.iloc[-1]["trade_date"].strftime("%Y-%m-%d")

    return Q1_BODY.format(
        n_records=len(df_orig),
        n_fields=n_fields,
        date_start=date_start,
        date_end=date_end,
        close_mean=close_mean,
        close_std=close_std,
        close_min=close_min,
        close_max=close_max,
        vol_mean=vol_mean,
        vol_std=vol_std,
    )


def build_html():
    """构建完整HTML文档（CSS内联+图片base64嵌入，便于浏览器直接打印PDF）"""
    # 读取CSS内容
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css_content = f.read()

    # 加载数据和计算指标
    df = load_data()
    df = calc_all_indicators(df)

    # 构建各部分HTML（Q1 正文使用实际统计数据动态填充）
    q1_text = build_q1_body(df)
    q1_html = text_to_html_paragraphs(q1_text)
    q2_html = text_to_html_paragraphs(Q2_BODY)
    q4_body_html = text_to_html_paragraphs(Q4_BODY)

    # 描述性统计表
    desc_table = build_describe_table()

    # 动态解读
    rsi_analysis = text_to_html_paragraphs(build_rsi_analysis(df))
    macd_analysis = text_to_html_paragraphs(build_macd_analysis(df))
    boll_analysis = text_to_html_paragraphs(build_boll_analysis(df))
    kdj_analysis = text_to_html_paragraphs(build_kdj_analysis(df))

    # 代码片段转义
    load_code = escape_code(LOAD_CODE)
    rsi_code = escape_code(RSI_CODE)
    macd_code = escape_code(MACD_CODE)
    boll_code = escape_code(BOLL_CODE)
    kdj_code = escape_code(KDJ_CODE)

    # 数据基本信息
    total = len(df)
    date_start = df.iloc[0]["trade_date"].strftime("%Y-%m-%d")
    date_end = df.iloc[-1]["trade_date"].strftime("%Y-%m-%d")

    # 图片转base64嵌入
    import base64
    def img_to_b64(filename):
        path = os.path.join(BASE_DIR, filename)
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    rsi_img = img_to_b64(RSI_PNG)
    macd_img = img_to_b64(MACD_PNG)
    boll_img = img_to_b64(BOLL_PNG)
    kdj_img = img_to_b64(KDJ_PNG)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>量化交易作业 TASK2 - {STUDENT_NAME}</title>
    <style>
{css_content}
    </style>
</head>
<body>

<div class="info-block">
    <h1>量化交易作业 TASK2</h1>
    <p>姓名：{STUDENT_NAME}</p>
    <p>日期：{date_end[:4]}年{date_end[5:7]}月{date_end[8:]}日</p>
</div>

<h2>{Q1_TITLE}</h2>
{q1_html}
{desc_table}

<h2>{Q2_TITLE}</h2>
{q2_html}

<h2>{Q3_TITLE}</h2>

<h3>3.1 数据加载与预处理</h3>
<p>本任务加载TASK1中已存储的比亚迪(002594.SZ)日线交易数据（002594_daily.csv），共{total}条记录，日期范围为{date_start}至{date_end}。数据加载与预处理代码如下：</p>
<pre class="code">{load_code}</pre>

<h3>3.2 RSI 计算与可视化</h3>
<p>使用Wilder平滑法计算14日RSI指标，计算代码如下：</p>
<pre class="code">{rsi_code}</pre>
<p>计算完成后，使用matplotlib绘制RSI走势图，如图1所示。图中紫色曲线为RSI(14)，红色虚线为超买线（70），绿色虚线为超卖线（30），灰色点线为多空分界线（50）。</p>
<div class="figure">
    <img src="{rsi_img}" alt="RSI走势图" />
    <div class="fig-caption">图1 比亚迪(002594.SZ) RSI(14) 相对强弱指数</div>
</div>
<p>图1解读：</p>
{rsi_analysis}

<h3>3.3 MACD 计算与可视化</h3>
<p>使用标准参数(12,26,9)计算MACD指标，计算代码如下：</p>
<pre class="code">{macd_code}</pre>
<p>计算完成后，绘制MACD指标图如图2所示。上方子图为DIF和DEA折线，下方子图为MACD柱状图（红色为正值，绿色为负值）。</p>
<div class="figure">
    <img src="{macd_img}" alt="MACD指标图" />
    <div class="fig-caption">图2 比亚迪(002594.SZ) MACD(12,26,9) 指数平滑异同移动平均线</div>
</div>
<p>图2解读：</p>
{macd_analysis}

<h3>3.4 布林带计算与可视化</h3>
<p>使用20日周期和2倍标准差计算布林带指标，计算代码如下：</p>
<pre class="code">{boll_code}</pre>
<p>计算完成后，绘制布林带通道图如图3所示。红色曲线为收盘价，蓝色虚线为上下轨，橙色点划线为中轨，浅蓝色填充为通道区域。</p>
<div class="figure">
    <img src="{boll_img}" alt="布林带通道图" />
    <div class="fig-caption">图3 比亚迪(002594.SZ) 布林带(20,2) Bollinger Bands</div>
</div>
<p>图3解读：</p>
{boll_analysis}

<h3>3.5 综合解读</h3>
<p>综合RSI、MACD和布林带三项指标的分析结果，可以得出以下结论：由于采用了前复权数据，三项指标均基于连续平滑的价格序列计算，信号质量可靠。RSI的超买超卖信号、MACD的金叉死叉信号和布林带的通道突破信号在走势图中清晰可辨，且方向基本一致，相互印证。在实际应用中，建议采用多指标组合策略，通过交叉验证降低单一指标的虚假信号风险，并结合基本面分析和仓位管理形成完整的交易决策体系。</p>

<h2>{Q4_TITLE}</h2>
{q4_body_html}

<h3>4.1 KDJ 计算与可视化</h3>
<p>使用标准参数(9,3,3)计算KDJ随机指标，计算代码如下：</p>
<pre class="code">{kdj_code}</pre>
<p>计算完成后，绘制KDJ指标图如图4所示。蓝色曲线为K值，橙色曲线为D值，紫色曲线为J值，红色虚线为超买线（80），绿色虚线为超卖线（20）。</p>
<div class="figure">
    <img src="{kdj_img}" alt="KDJ指标图" />
    <div class="fig-caption">图4 比亚迪(002594.SZ) KDJ(9,3,3) 随机指标</div>
</div>
<p>图4解读：</p>
{kdj_analysis}

</body>
</html>"""

    return html


def _print_pdf_via_browser(html_path, pdf_path):
    """用 Edge/Chrome headless 模式将 HTML 打印为 PDF（无需GTK）"""
    import subprocess
    import shutil as _shutil

    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    browser = None
    for c in candidates:
        if os.path.exists(c):
            browser = c
            break
    if browser is None:
        browser = _shutil.which("msedge") or _shutil.which("chrome")
    if browser is None:
        raise RuntimeError(
            "未找到 Edge 或 Chrome 浏览器，无法生成PDF。\n"
            "请安装 Microsoft Edge 或 Google Chrome。"
        )

    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html.replace(os.sep, '/')}",
    ]
    print(f"使用浏览器生成PDF: {browser}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if not os.path.exists(abs_pdf):
        raise RuntimeError(
            f"PDF生成失败。\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def main():
    print("=" * 60)
    print(f"开始生成 TASK2 PDF 文档")
    print(f"学生姓名: {STUDENT_NAME}")
    print(f"输出文件: {PDF_NAME}")
    print("=" * 60)

    # 检查依赖文件
    for f in ["002594_daily.csv", "style.css", "002594_rsi.png",
              "002594_macd.png", "002594_boll.png", "002594_kdj.png"]:
        path = os.path.join(BASE_DIR, f)
        if not os.path.exists(path):
            print(f"错误：找不到文件 {path}")
            sys.exit(1)

    # 构建HTML（CSS内联 + 图片base64嵌入）
    print("构建HTML内容...")
    html_content = build_html()

    # 先保存独立HTML
    html_path = PDF_PATH.replace(".pdf", ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML 已保存: {html_path}")

    # 生成PDF
    print("生成PDF...")
    if _HAS_WEASYPRINT:
        html_doc = HTML(string=html_content, base_url=BASE_DIR)
        css = CSS(filename=CSS_PATH)
        html_doc.write_pdf(PDF_PATH, stylesheets=[css])
    else:
        # 降级方案：用 Edge/Chrome headless 模式打印PDF
        _print_pdf_via_browser(html_path, PDF_PATH)

    print(f"\nPDF 生成成功！")
    print(f"文件路径: {PDF_PATH}")
    print(f"文件大小: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
