# -*- coding: utf-8 -*-
"""
build_pdf.py
生成符合格式要求的 PDF 文档（weasyprint）
格式：宋体(Noto Serif CJK SC)、五号字(10.5pt)、1.5倍行距、0段间距、两端对齐

注意：weasyprint 在 Windows 上需要 GTK 运行时（libgobject-2.0）。
若环境缺少 GTK，脚本仍可正常 import 并调用各分析函数，
仅在执行 main() 生成 PDF 时报错。
"""
import os
import sys
import pandas as pd

try:
    from weasyprint import HTML, CSS
    _HAS_WEASYPRINT = True
except OSError:
    _HAS_WEASYPRINT = False
    HTML = CSS = None

# 添加当前目录到路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from content import Q1_TITLE, Q1_BODY, Q2_TITLE, Q2_BODY, Q3_TITLE, CODE_SNIPPET

# ============ 配置 ============
STUDENT_NAME = "薛德刚"
PDF_NAME = f"{STUDENT_NAME}+TASK1.pdf"
PDF_PATH = os.path.join(BASE_DIR, PDF_NAME)
CSS_PATH = os.path.join(BASE_DIR, "style.css")
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")
PNG_PATH = os.path.join(BASE_DIR, "002594_close.png")


def text_to_html_paragraphs(text):
    """将纯文本转换为HTML段落（每段一个<p>标签）"""
    paragraphs = text.strip().split("\n")
    html = ""
    for para in paragraphs:
        para = para.strip()
        if para:
            # 转义HTML特殊字符
            para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f"<p>{para}</p>\n"
    return html


def build_data_table():
    """构建CSV前5行数据的HTML表格"""
    df = pd.read_csv(CSV_PATH)

    # 选取要展示的列
    display_cols = ["trade_date", "open", "high", "low", "close", "vol"]
    available_cols = [c for c in display_cols if c in df.columns]
    df_display = df[available_cols].head()

    # 中文表头
    header_map = {
        "trade_date": "交易日期",
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "vol": "成交量",
    }

    # 构建表格HTML
    html = '<table class="data">\n'
    html += '<caption>表1 比亚迪(002594.SZ)前5个交易日数据</caption>\n'
    html += "<thead><tr>"
    for col in available_cols:
        html += f"<th>{header_map.get(col, col)}</th>"
    html += "</tr></thead>\n<tbody>\n"

    for _, row in df_display.iterrows():
        html += "<tr>"
        for col in available_cols:
            val = row[col]
            if col == "trade_date":
                # 格式化日期为 YYYY-MM-DD
                val = str(val).replace("-", "")[:8]
                if len(val) == 8:
                    val = f"{val[:4]}-{val[4:6]}-{val[6:]}"
            elif isinstance(val, float):
                val = f"{val:.2f}"
            html += f"<td>{val}</td>"
        html += "</tr>\n"

    html += "</tbody></table>\n"
    return html


def build_analysis_text():
    """根据实际数据生成图表解读文字（前复权数据，价格连续平滑）"""
    df = pd.read_csv(CSV_PATH)

    # 确保close列存在
    if "close" not in df.columns:
        return "数据解读失败：缺少收盘价字段。"

    df = df.sort_values("trade_date").reset_index(drop=True)
    close = df["close"]
    total = len(df)

    # 格式化日期
    def fmt_date(d):
        s = str(d).replace("-", "")[:8]
        return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 else str(d)

    first_date = fmt_date(df.iloc[0]["trade_date"])
    last_date = fmt_date(df.iloc[-1]["trade_date"])
    first_close = close.iloc[0]
    last_close = close.iloc[-1]
    max_close = close.max()
    min_close = close.min()
    avg_close = close.mean()

    # 最高最低价对应日期
    max_date = fmt_date(df.loc[close.idxmax(), "trade_date"])
    min_date = fmt_date(df.loc[close.idxmin(), "trade_date"])

    # 涨跌幅
    change = last_close - first_close
    change_pct = (change / first_close) * 100

    # 波动率（标准差）
    volatility = close.std()

    analysis = f"""从图1可以看出，比亚迪(002594.SZ)在过去一年（{first_date}至{last_date}）共{total}个交易日中，收盘价整体呈现出明显的波动特征。统计数据显示，该期间收盘价最高达到{max_close:.2f}元（出现在{max_date}），最低为{min_close:.2f}元（出现在{min_date}），均价约为{avg_close:.2f}元。

本数据采用前复权（qfq）方式处理，已自动消除除权除息带来的价格跳变，价格曲线连续平滑，能够真实反映股价的运行趋势。期间收盘价从首日的{first_close:.2f}元变动至末日的{last_close:.2f}元，区间涨跌幅为{change_pct:+.2f}%。

从整体走势来看，收盘价的标准差为{volatility:.2f}元，反映出该股票在观察期内具有一定的价格波动幅度。该走势图所呈现的价格变化与新能源汽车行业的政策环境、市场竞争格局以及公司基本面变化等因素密切相关，为投资者提供了直观的价格变动参考。

该收盘价走势图为投资者提供了直观的价格变动参考，有助于识别价格的趋势方向和波动区间，为后续的技术分析和投资决策提供数据支撑。前复权数据消除了除权除息的干扰，使技术指标的计算结果更加准确可靠。"""

    return analysis.strip()


def build_html():
    """构建完整HTML文档（CSS内联，便于浏览器直接打印PDF）"""
    # 读取CSS内容
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css_content = f.read()

    # 读取数据
    df = pd.read_csv(CSV_PATH)
    total_records = len(df)

    # 构建各部分HTML
    q1_html = text_to_html_paragraphs(Q1_BODY)
    q2_html = text_to_html_paragraphs(Q2_BODY)
    analysis_text = build_analysis_text()
    analysis_html = text_to_html_paragraphs(analysis_text)
    data_table = build_data_table()

    # 代码片段（转义HTML特殊字符）
    code_escaped = CODE_SNIPPET.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 数据来源说明
    data_source = "tushare pro_bar 接口"
    start_date = str(df.iloc[0]["trade_date"]).replace("-", "")[:8]
    end_date = str(df.iloc[-1]["trade_date"]).replace("-", "")[:8]

    # 图片转base64嵌入，确保独立可移植
    import base64
    with open(PNG_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    img_src = f"data:image/png;base64,{img_b64}"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>量化交易作业 TASK1 - {STUDENT_NAME}</title>
    <style>
{css_content}
    </style>
</head>
<body>

<div class="info-block">
    <h1>量化交易作业 TASK1</h1>
    <p>姓名：{STUDENT_NAME}</p>
    <p>日期：{end_date[:4]}年{end_date[4:6]}月{end_date[6:]}日</p>
</div>

<h2>{Q1_TITLE}</h2>
{q1_html}

<h2>{Q2_TITLE}</h2>
{q2_html}

<h2>{Q3_TITLE}</h2>

<h3>3.1 数据获取</h3>
<p>本任务通过 Tushare 平台获取比亚迪(002594.SZ)过去一年的日线交易数据。首先在 Tushare 官网（https://www.tushare.pro/）注册账号并获取 API Token，然后使用 Python 调用 pro_bar 接口获取前复权日线数据。核心代码如下：</p>
<pre class="code">{code_escaped}</pre>
<p>上述代码通过 ts.pro_bar() 函数获取了比亚迪从{start_date[:4]}年{start_date[4:6]}月{start_date[6:8]}日至{end_date[:4]}年{end_date[4:6]}月{end_date[6:8]}日期间的日线行情数据，共获取{total_records}个交易日的数据记录。数据来源为{data_source}，字段包括交易日期(trade_date)、开盘价(open)、最高价(high)、最低价(low)、收盘价(close)、成交量(vol)、成交额(amount)等。</p>

<h3>3.2 每日收盘价曲线图</h3>
<p>基于获取的交易数据，使用 Python 的 matplotlib 库绘制了比亚迪过去一年的每日收盘价走势图，如图1所示。</p>
<div class="figure">
    <img src="{img_src}" alt="比亚迪收盘价走势图" />
    <div class="fig-caption">图1 比亚迪(002594.SZ)过去一年每日收盘价走势</div>
</div>
<p>图1解读：</p>
{analysis_html}

<h3>3.3 数据保存为CSV格式</h3>
<p>获取的交易数据已保存为 CSV 格式文件（002594_daily.csv），编码方式为 UTF-8-SIG（带BOM），可确保在 Excel 中打开时中文不乱码。文件包含{total_records}条日线记录，每条记录包含交易日期、开盘价、最高价、最低价、收盘价、成交量等字段。下表展示了前5个交易日的数据示例：</p>
{data_table}
<p>该CSV文件可作为后续量化分析任务的数据基础，便于进行技术指标计算、策略回测和模型训练等进一步分析工作。</p>

</body>
</html>"""

    return html


def _print_pdf_via_browser(html_path, pdf_path):
    """用 Edge/Chrome headless 模式将 HTML 打印为 PDF（无需GTK）"""
    import subprocess
    import shutil as _shutil

    # 按优先级查找浏览器
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

    # Edge/Chrome headless 打印PDF
    # --print-to-pdf-no-header 去掉页眉页脚
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if not os.path.exists(abs_pdf):
        raise RuntimeError(
            f"PDF生成失败。\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def main():
    print("=" * 60)
    print(f"开始生成 PDF 文档")
    print(f"学生姓名: {STUDENT_NAME}")
    print(f"输出文件: {PDF_NAME}")
    print("=" * 60)

    # 检查依赖文件
    if not os.path.exists(CSV_PATH):
        print(f"错误：找不到数据文件 {CSV_PATH}")
        print("请先运行 fetch_data.py 获取数据")
        sys.exit(1)

    if not os.path.exists(PNG_PATH):
        print(f"错误：找不到图表文件 {PNG_PATH}")
        print("请先运行 plot_close.py 生成图表")
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
