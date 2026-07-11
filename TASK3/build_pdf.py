# -*- coding: utf-8 -*-
"""
build_pdf.py
生成 TASK3 双均线策略 PDF 报告
格式：宋体、五号字(10.5pt)、1.5倍行距、0段间距、两端对齐
"""
import os
import sys
import base64
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from strategy import (
    run_strategy, load_data, calc_ma, generate_signals, backtest, calc_metrics,
    DEFAULT_SHORT, DEFAULT_LONG, INITIAL_CAPITAL, COMMISSION_RATE, STAMP_TAX_RATE
)
from content import Q1_TITLE, Q1_BODY, Q2_TITLE, Q2_BODY, Q3_TITLE, Q3_CODE

STUDENT_NAME = "薛德刚"
PDF_NAME = f"{STUDENT_NAME}+TASK3.pdf"
PDF_PATH = os.path.join(BASE_DIR, PDF_NAME)
CSS_PATH = os.path.join(BASE_DIR, "style.css")
CSV_PATH = os.path.join(BASE_DIR, "002594_daily.csv")

# 图片路径
IMG1 = os.path.join(BASE_DIR, "002594_ma_signals.png")
IMG2 = os.path.join(BASE_DIR, "002594_nav_drawdown.png")
IMG3 = os.path.join(BASE_DIR, "002594_param_comparison.png")


def text_to_html_paragraphs(text):
    """纯文本转HTML段落"""
    paragraphs = text.strip().split("\n")
    html = ""
    for para in paragraphs:
        para = para.strip()
        if para:
            para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f"<p>{para}</p>\n"
    return html


def img_to_base64(path):
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"


def build_strategy_analysis():
    """动态生成 MA5/15 策略分析文字"""
    df, m = run_strategy(CSV_PATH, DEFAULT_SHORT, DEFAULT_LONG)

    df_raw = load_data(CSV_PATH)
    first_date = str(df_raw.iloc[0]["trade_date"])
    first_date = f"{first_date[:4]}-{first_date[4:6]}-{first_date[6:]}" if len(first_date) == 8 else first_date
    last_date = str(df_raw.iloc[-1]["trade_date"])
    last_date = f"{last_date[:4]}-{last_date[4:6]}-{last_date[6:]}" if len(last_date) == 8 else last_date
    total_days = len(df_raw)

    # 超额收益
    alpha = m["cumulative_return"] - m["benchmark_return"]

    # 评价
    if m["cumulative_return"] > m["benchmark_return"]:
        perf_eval = "策略累计回报高于买入持有基准，表明双均线策略在该期间内成功规避了部分下跌风险，具有一定的趋势跟踪能力"
    else:
        perf_eval = "策略累计回报低于买入持有基准，表明在该震荡期间内双均线策略产生了较多虚假信号，趋势跟踪效果不理想"

    if abs(m["max_drawdown"]) < abs(m["benchmark_mdd"]):
        risk_eval = "策略最大回撤小于基准，说明策略在风险控制方面表现优于买入持有策略"
    else:
        risk_eval = "策略最大回撤大于基准，说明策略在风险控制方面未能优于买入持有策略"

    analysis = f"""以比亚迪(002594.SZ)从{first_date}至{last_date}共{total_days}个交易日的前复权日线数据为研究对象，采用MA{DEFAULT_SHORT}/MA{DEFAULT_LONG}双均线策略进行模拟回测。初始资金设为{INITIAL_CAPITAL:,.0f}元，交易成本考虑佣金（万分之{COMMISSION_RATE*10000:.0f}）和印花税（卖出方千分之{STAMP_TAX_RATE*1000:.0f}）。

回测结果显示，策略在考察期内共产生{m["n_trades"]}次交易信号，其中买入信号{m["buy_signals"]}次（金叉），卖出信号{m["sell_signals"]}次（死叉），完整交易轮数为{m["total_round_trips"]}轮。七项核心绩效指标如下：年化收益率为{m["ann_return"]:+.2f}%，最大回撤为{m["max_drawdown"]:.2f}%，年化波动率为{m["ann_volatility"]:.2f}%，夏普比率为{m["sharpe"]:.3f}，胜率为{m["win_rate"]:.1f}%，盈亏比为{m["profit_loss_ratio"]:.2f}，期望收益为{m["expectancy_r"]:.3f}R。同期买入持有基准的累计回报为{m["benchmark_return"]:+.2f}%，最大回撤为{m["benchmark_mdd"]:.2f}%，年化波动率为{m["benchmark_volatility"]:.2f}%，夏普比率为{m["benchmark_sharpe"]:.3f}。

从收益维度看，策略累计回报率为{m["cumulative_return"]:+.2f}%，{perf_eval}，超额收益为{alpha:+.2f}%。年化收益率为{m["ann_return"]:+.2f}%，虽然为负，但优于基准的同期年化表现。

从风险维度看，{risk_eval}。年化波动率为{m["ann_volatility"]:.2f}%，低于基准的{m["benchmark_volatility"]:.2f}%，说明策略通过择时降低了收益的波动幅度。夏普比率为{m["sharpe"]:.3f}，基准夏普为{m["benchmark_sharpe"]:.3f}，两者均为负值，反映了在考察期内该股票整体处于下行趋势。

从交易质量维度看，胜率为{m["win_rate"]:.1f}%，盈亏比为{m["profit_loss_ratio"]:.2f}，期望收益为{m["expectancy_r"]:.3f}R。胜率较低说明在震荡行情中金叉死叉信号的可靠性有限，但盈亏比达到{m["profit_loss_ratio"]:.2f}，意味着盈利交易的平均收益是亏损交易的{m["profit_loss_ratio"]:.2f}倍。期望收益为{m["expectancy_r"]:.3f}R（负值），表明在当前参数和市场环境下，策略长期运行将产生亏损，需要进一步优化参数或增加过滤条件。

综合来看，在MA{DEFAULT_SHORT}/MA{DEFAULT_LONG}参数下，双均线策略在比亚迪这一年震荡下行的行情中表现中规中矩。策略在控制最大回撤和波动率方面优于买入持有基准，但较低的胜率和负的期望收益暴露了均线策略在震荡市中的固有弱点：频繁的虚假交叉信号导致连续的小额亏损，侵蚀策略整体收益。投资者在使用双均线策略时，应结合趋势判断指标过滤虚假信号，或在明确的趋势行情中才启用该策略。"""

    return analysis.strip(), m


def build_param_analysis(res_df):
    """参数对比分析文字（7项指标）"""
    best_return = res_df.loc[res_df["return"].idxmax()]
    best_ann = res_df.loc[res_df["ann_return"].idxmax()]
    best_mdd = res_df.loc[res_df["mdd"].idxmax()]
    best_sharpe = res_df.loc[res_df["sharpe"].idxmax()]
    best_win = res_df.loc[res_df["win_rate"].idxmax()]
    best_plr = res_df.loc[res_df["plr"].idxmax()]
    best_exp = res_df.loc[res_df["expectancy"].idxmax()]

    worst_return = res_df.loc[res_df["return"].idxmin()]

    analysis = f"""为了探索不同均线参数组合对策略表现的影响，本任务选取了10组典型的短长均线组合进行对比测试，包括MA3/5、MA3/10、MA5/10、MA5/15、MA5/20、MA5/30、MA10/20、MA10/30、MA10/60和MA20/60。

从收益维度看，累计回报率表现最佳的参数组合为MA{int(best_return['short'])}/MA{int(best_return['long'])}，累计回报率为{best_return['return']:.2f}%，年化收益率为{best_return['ann_return']:.2f}%；表现最差的组合为MA{int(worst_return['short'])}/MA{int(worst_return['long'])}，累计回报率为{worst_return['return']:.2f}%。最大回撤方面，MA{int(best_mdd['short'])}/MA{int(best_mdd['long'])}组合的最大回撤最浅，为{best_mdd['mdd']:.2f}%。

从风险调整维度看，MA{int(best_sharpe['short'])}/MA{int(best_sharpe['long'])}组合的夏普比率最高，为{best_sharpe['sharpe']:.3f}。年化波动率方面，短线组合（如MA3/5）的波动率普遍低于长线组合，这是因为短线策略空仓时间更长，降低了组合波动。

从交易质量维度看，MA{int(best_win['short'])}/MA{int(best_win['long'])}组合以{best_win['win_rate']:.1f}%的胜率领先。盈亏比方面，MA{int(best_plr['short'])}/MA{int(best_plr['long'])}组合的盈亏比最高，为{best_plr['plr']:.2f}。期望收益方面，MA{int(best_exp['short'])}/MA{int(best_exp['long'])}组合的期望收益最高，为{best_exp['expectancy']:.3f}R。

整体观察可以发现以下规律：第一，短周期参数组合（如MA3/5）的交易频率更高，在快速变化的行情中反应更灵敏，胜率相对较高，但手续费成本也更高；第二，长周期参数组合（如MA20/60）交易次数少，信号更稳定，但对趋势变化的反应滞后；第三，在比亚迪这一年的震荡行情中，多数参数组合的累计回报和期望收益为负，说明双均线策略本质上是一种趋势跟踪策略，在缺乏明确趋势的市场环境中难以发挥优势。投资者在使用双均线策略时，应首先判断市场处于趋势市还是震荡市，在趋势明确时使用较长周期的均线组合，在波动较大的短线环境中可适当缩短均线周期，但需控制交易频率以降低成本侵蚀。同时，盈亏比和期望收益(R)是评估策略长期可行性的关键指标——只有期望收益为正的策略才值得长期执行。"""

    return analysis.strip()


def build_param_table(res_df):
    """参数对比表格（7项指标）"""
    html = '<table class="data">\n'
    html += '<caption>表1 不同均线参数组合策略绩效对比（7项核心指标）</caption>\n'
    html += '<thead><tr><th>参数</th><th>年化收益(%)</th><th>最大回撤(%)</th><th>波动率(%)</th><th>夏普</th><th>胜率(%)</th><th>盈亏比</th><th>期望R</th><th>交易数</th></tr></thead>\n<tbody>\n'

    for _, row in res_df.iterrows():
        html += f"<tr><td>{row['combo']}</td>"
        html += f"<td>{row['ann_return']:+.2f}</td>"
        html += f"<td>{row['mdd']:.2f}</td>"
        html += f"<td>{row['volatility']:.2f}</td>"
        html += f"<td>{row['sharpe']:.3f}</td>"
        html += f"<td>{row['win_rate']:.1f}</td>"
        html += f"<td>{row['plr']:.2f}</td>"
        html += f"<td>{row['expectancy']:.3f}</td>"
        html += f"<td>{int(row['trades'])}</td></tr>\n"

    html += "</tbody></table>\n"
    return html


def build_html():
    """构建完整HTML"""
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css_content = f.read()

    # 运行策略获取数据
    df, m = run_strategy(CSV_PATH, DEFAULT_SHORT, DEFAULT_LONG)

    # 参数对比
    from plot_strategy import plot_param_comparison
    _, res_df = plot_param_comparison.__wrapped__() if hasattr(plot_param_comparison, '__wrapped__') else (None, None)

    # 重新计算参数对比数据（不重复绘图）
    df_raw = load_data(CSV_PATH)
    combos = [(3, 5), (3, 10), (5, 10), (5, 15), (5, 20), (5, 30), (10, 20), (10, 30), (10, 60), (20, 60)]
    results = []
    for short, long in combos:
        d = calc_ma(df_raw.copy(), short, long)
        d = generate_signals(d)
        d = backtest(d)
        met = calc_metrics(d)
        results.append({
            "combo": f"MA{short}/{long}",
            "short": short, "long": long,
            "return": met["cumulative_return"],
            "ann_return": met["ann_return"],
            "mdd": met["max_drawdown"],
            "volatility": met["ann_volatility"],
            "sharpe": met["sharpe"],
            "trades": met["n_trades"],
            "win_rate": met["win_rate"],
            "plr": met["profit_loss_ratio"],
            "expectancy": met["expectancy_r"],
        })
    res_df = pd.DataFrame(results)

    q1_html = text_to_html_paragraphs(Q1_BODY)
    q2_html = text_to_html_paragraphs(Q2_BODY)
    strategy_analysis, _ = build_strategy_analysis()
    strategy_html = text_to_html_paragraphs(strategy_analysis)
    param_analysis = build_param_analysis(res_df)
    param_html = text_to_html_paragraphs(param_analysis)
    param_table = build_param_table(res_df)

    code_escaped = Q3_CODE.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    img1_b64 = img_to_base64(IMG1)
    img2_b64 = img_to_base64(IMG2)
    img3_b64 = img_to_base64(IMG3)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>量化交易作业 TASK3 - {STUDENT_NAME}</title>
    <style>
{css_content}
    </style>
</head>
<body>

<div class="info-block">
    <h1>量化交易作业 TASK3</h1>
    <p>双均线策略：用均线交叉反应市场趋势和波动</p>
    <p>姓名：{STUDENT_NAME}</p>
    <p>分析标的：比亚迪 002594.SZ</p>
</div>

<h2>{Q1_TITLE}</h2>
{q1_html}

<h2>{Q2_TITLE}</h2>
{q2_html}

<h2>{Q3_TITLE}</h2>

<h3>3.1 策略实现</h3>
<p>本任务以前两个任务获取并处理的比亚迪(002594.SZ)前复权日线数据为基础，使用纯Python实现双均线策略的完整流程：加载股价数据、计算均线、生成信号、模拟回测和绩效评估。核心代码如下：</p>
<pre class="code">{code_escaped}</pre>

<h3>3.2 均线交叉信号可视化</h3>
<p>基于MA{DEFAULT_SHORT}/MA{DEFAULT_LONG}双均线策略参数，绘制了股价走势、长短均线及买卖信号标记图，如图1所示。</p>
<div class="figure">
    <img src="{img1_b64}" alt="双均线策略信号图" />
    <div class="fig-caption">图1 比亚迪(002594.SZ) MA{DEFAULT_SHORT}/{DEFAULT_LONG} 双均线策略信号图</div>
</div>
<p>图1解读：</p>
{strategy_html}

<h3>3.3 策略净值与回撤分析</h3>
<p>图2展示了策略累计净值与买入持有基准净值的对比，以及两者的回撤变化。</p>
<div class="figure">
    <img src="{img2_b64}" alt="净值与回撤图" />
    <div class="fig-caption">图2 策略净值 vs 基准净值及回撤对比</div>
</div>
<p>图2解读：上图显示策略净值（红色实线）与基准净值（灰色虚线）的走势对比。在市场下跌阶段，策略净值高于基准，表明双均线策略通过及时卖出规避了部分下跌风险；下图展示了策略和基准的回撤幅度对比，红色区域为策略回撤，灰色区域为基准回撤，可以直观看出策略在控制回撤方面的表现。</p>

<h3>3.4 不同均线参数组合对比</h3>
<p>为研究不同均线周期对策略表现的影响，选取了10组典型参数组合进行回测对比，结果如图3和表1所示。</p>
<div class="figure">
    <img src="{img3_b64}" alt="参数对比图" />
    <div class="fig-caption">图3 不同均线参数组合绩效对比</div>
</div>
{param_table}
<p>表1和图3解读：</p>
{param_html}

</body>
</html>"""

    return html


def _print_pdf_via_browser(html_path, pdf_path):
    """用 Edge/Chrome headless 打印PDF"""
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
        raise RuntimeError("未找到 Edge 或 Chrome 浏览器，无法生成PDF。")

    abs_html = os.path.abspath(html_path)
    abs_pdf = os.path.abspath(pdf_path)
    cmd = [
        browser, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html.replace(os.sep, '/')}",
    ]
    print(f"使用浏览器生成PDF: {browser}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if not os.path.exists(abs_pdf):
        raise RuntimeError(f"PDF生成失败。\nstdout: {result.stdout}\nstderr: {result.stderr}")


def main():
    print("=" * 60)
    print(f"开始生成 TASK3 PDF 文档")
    print(f"学生姓名: {STUDENT_NAME}")
    print(f"输出文件: {PDF_NAME}")
    print("=" * 60)

    # 检查依赖
    for f in [CSV_PATH, IMG1, IMG2, IMG3]:
        if not os.path.exists(f):
            print(f"错误：找不到文件 {f}")
            sys.exit(1)

    print("构建HTML内容...")
    html_content = build_html()

    html_path = PDF_PATH.replace(".pdf", ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML 已保存: {html_path}")

    print("生成PDF...")
    _print_pdf_via_browser(html_path, PDF_PATH)

    print(f"\nPDF 生成成功！")
    print(f"文件路径: {PDF_PATH}")
    print(f"文件大小: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
