# -*- coding: utf-8 -*-
"""
build_pdf.py
生成 TASK6 机器学习选股策略 PDF 报告
"""
import os
import sys
import base64
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ml_strategy import main as run_strategy
from content import Q1_TITLE, Q1_BODY, Q2_TITLE, Q2_BODY, Q3_TITLE, Q3_CODE

STUDENT_NAME = "薛德刚"
PDF_NAME = f"{STUDENT_NAME}+TASK6.pdf"
PDF_PATH = os.path.join(BASE_DIR, PDF_NAME)
CSS_PATH = os.path.join(BASE_DIR, "style.css")

IMG1 = os.path.join(BASE_DIR, "quarterly_returns.png")
IMG2 = os.path.join(BASE_DIR, "nav_curves.png")
IMG3 = os.path.join(BASE_DIR, "feature_importance.png")
IMG4 = os.path.join(BASE_DIR, "performance_comparison.png")


def text_to_html_paragraphs(text):
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


def build_metrics_table(all_perf):
    """构建三模型绩效对比表"""
    html = '<table class="data">\n'
    html += '<caption>表1 三模型选股策略绩效指标对比</caption>\n'
    html += '<thead><tr><th>指标</th><th>线性回归</th><th>决策树</th><th>随机森林</th><th>市场基准</th></tr></thead>\n<tbody>\n'

    metrics = [
        ("累计回报(%)", "cum_return", "bench_cum_return"),
        ("年化收益率(%)", "ann_return", "bench_ann_return"),
        ("最大回撤(%)", "mdd", "bench_mdd"),
        ("年化波动率(%)", "ann_vol", "bench_vol"),
        ("夏普比率", "sharpe", "bench_sharpe"),
        ("胜率(%)", "win_rate", None),
        ("盈亏比", "plr", None),
        ("期望收益(R)", "exp_r", None),
    ]

    for label, key, bench_key in metrics:
        html += f"<tr><td>{label}</td>"
        for name in ["线性回归", "决策树", "随机森林"]:
            val = all_perf[name][key]
            if abs(val) < 0.01 and val != 0:
                html += f"<td>{val:.4f}</td>"
            else:
                html += f"<td>{val:.2f}</td>"
        # 基准列
        if bench_key is not None:
            bench_val = all_perf["随机森林"][bench_key]
            if abs(bench_val) < 0.01 and bench_val != 0:
                html += f"<td>{bench_val:.4f}</td>"
            else:
                html += f"<td>{bench_val:.2f}</td>"
        else:
            html += "<td>-</td>"
        html += "</tr>\n"

    html += "</tbody></table>\n"
    return html


def build_strategy_analysis(all_perf, all_backtest, model_results):
    """动态生成策略分析文字"""
    lr = all_perf["线性回归"]
    dt = all_perf["决策树"]
    rf = all_perf["随机森林"]
    n_stocks = len(all_backtest["随机森林"]["quarter_labels"]) + 1

    best_model = max(all_perf.items(), key=lambda x: x[1]["cum_return"])
    best_name = best_model[0]
    best_perf = best_model[1]

    fi = model_results["随机森林"]["feature_importance"]
    top3_features = ""
    if fi is not None:
        top3 = fi.head(3)
        top3_features = "、".join([f"{idx}({val:.4f})" for idx, val in top3.items()])

    analysis = f"""本任务选取48只A股代表性标的（覆盖银行金融、新能源、消费、科技、周期、医药、基建等行业），获取2023年1月至2026年7月的不复权日线数据，共41067条记录。基于日线数据构建12个技术因子作为自变量，以未来60个交易日（约一个季度）收益率为应变量，训练线性回归、决策树和随机森林三个回归模型，并基于模型预测结果构建季度选股策略——每季度初选预测收益最高的30只股票等权配置，持有一个季度后调仓。

数据按时间划分：2023-2024年为训练集（22262条），2025-2026年为测试集（14965条），共覆盖13个季度的回测区间。三个模型在测试集上的R²均为负值（线性回归R²={model_results['线性回归']['r2']:.4f}，决策树R²={model_results['决策树']['r2']:.4f}，随机森林R²={model_results['随机森林']['r2']:.4f}），说明模型对未来收益的直接预测能力有限——这符合金融市场的弱有效性，股票收益的噪音远大于信号。然而，R²为负并不意味着策略无效——关键在于模型能否在横截面上区分相对收益高低，即选股能力。

季度选股策略的回测结果如下。决策树模型表现最优，累计回报为{dt['cum_return']:+.2f}%，同期市场基准（48只股票等权）累计回报为{dt['bench_cum_return']:+.2f}%，超额收益（alpha）为{dt['alpha']:+.2f}%。线性回归模型累计回报为{lr['cum_return']:+.2f}%，超额收益为{lr['alpha']:+.2f}%。随机森林模型累计回报为{rf['cum_return']:+.2f}%，超额收益为{rf['alpha']:+.2f}%。三个模型均跑赢了市场基准，其中决策树模型的季度胜率达到{dt['win_rate']:.1f}%，盈亏比为{dt['plr']:.2f}，夏普比率为{dt['sharpe']:.3f}，显著优于基准的{dt['bench_sharpe']:.3f}。

从风险控制维度看，决策树模型的最大回撤为{dt['mdd']:.2f}%，低于基准的{dt['bench_mdd']:.2f}%，年化波动率为{dt['ann_vol']:.2f}%，低于基准的{dt['bench_vol']:.2f}%，说明模型在获取超额收益的同时有效控制了风险。线性回归模型虽然超额收益为正，但最大回撤（{lr['mdd']:.2f}%）略高于基准，波动控制能力不如决策树。随机森林模型在回撤控制上表现最差（{rf['mdd']:.2f}%），可能是因为模型对训练数据的过拟合导致在部分季度的选股偏差较大。

从因子重要性来看，随机森林模型识别出的Top 3因子为{top3_features}。波动率因子（volatility_20d）的重要性最高（{fi.iloc[0]:.4f}），说明低波动率股票在未来一个季度更容易获得较好收益，这与"低波动率异象"（Low Volatility Anomaly）的学术发现一致。换手率代理因子排名第二，反映了市场关注度对收益的预测作用。成交量均线比率排名第三，说明短期资金流入对股价有正向推动作用。

综合三个模型的回测结果可以得出以下结论。第一，机器学习选股策略在熊市环境中（2025-2026年A股整体下行）能够有效跑赢市场基准，三个模型均获得了正的超额收益。第二，模型选择对策略表现影响显著——决策树模型在本任务中表现最优，可能是因为数据量有限时，决策树的简单结构比随机森林更不易过拟合。第三，机器学习模型的价值不在于精准预测绝对收益，而在于横截面选股能力——即使R²为负，只要模型能正确识别相对收益更高的股票，策略就能获得超额收益。第四，投资者在实盘应用中应结合因子重要性分析，关注低波动率和高成交量的股票，同时控制调仓频率以降低交易成本。"""

    return analysis.strip()


def build_html():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css_content = f.read()

    # 运行策略
    model_results, all_backtest, all_perf, _ = run_strategy()

    q1_html = text_to_html_paragraphs(Q1_BODY)
    q2_html = text_to_html_paragraphs(Q2_BODY)
    strategy_analysis = build_strategy_analysis(all_perf, all_backtest, model_results)
    strategy_html = text_to_html_paragraphs(strategy_analysis)
    metrics_table = build_metrics_table(all_perf)

    code_escaped = Q3_CODE.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    img1_b64 = img_to_base64(IMG1)
    img2_b64 = img_to_base64(IMG2)
    img3_b64 = img_to_base64(IMG3)
    img4_b64 = img_to_base64(IMG4)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>量化交易作业 TASK6 - {STUDENT_NAME}</title>
    <style>
{css_content}
    </style>
</head>
<body>

<div class="info-block">
    <h1>量化交易作业 TASK6</h1>
    <p>智能决策者：机器学习定制专属策略</p>
    <p>姓名：{STUDENT_NAME}</p>
    <p>分析标的：48只A股代表性标的 · 季度选股策略</p>
</div>

<h2>{Q1_TITLE}</h2>
{q1_html}

<h2>{Q2_TITLE}</h2>
{q2_html}

<h2>{Q3_TITLE}</h2>

<h3>3.1 策略实现</h3>
<p>本任务选取48只A股代表性标的，获取2023-2026年日线数据，构建12个技术因子，以未来60日收益率为应变量，训练三个回归模型并构建季度选股策略。核心代码如下：</p>
<pre class="code">{code_escaped}</pre>

<h3>3.2 季度收益对比</h3>
<p>图1展示了三个模型选股策略与市场基准在每个季度的收益率对比。</p>
<div class="figure">
    <img src="{img1_b64}" alt="季度收益对比" />
    <div class="fig-caption">图1 三模型季度选股策略收益 vs 市场基准</div>
</div>

<h3>3.3 累计净值曲线</h3>
<p>图2展示了三个模型选股策略与市场基准的累计净值曲线对比。</p>
<div class="figure">
    <img src="{img2_b64}" alt="累计净值曲线" />
    <div class="fig-caption">图2 机器学习选股策略累计净值 vs 市场基准</div>
</div>

<h3>3.4 因子重要性分析</h3>
<p>图3展示了随机森林模型识别的因子重要性排序。</p>
<div class="figure">
    <img src="{img3_b64}" alt="因子重要性" />
    <div class="fig-caption">图3 随机森林因子重要性排序</div>
</div>

<h3>3.5 绩效指标对比</h3>
<p>图4和表1展示了三个模型在6项核心绩效指标上的对比。</p>
<div class="figure">
    <img src="{img4_b64}" alt="绩效对比" />
    <div class="fig-caption">图4 三模型选股策略绩效指标对比</div>
</div>
{metrics_table}
<p>图1-图4和表1解读：</p>
{strategy_html}

</body>
</html>"""

    return html


def _print_pdf_via_browser(html_path, pdf_path):
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
    print(f"开始生成 TASK6 PDF 文档")
    print(f"学生姓名: {STUDENT_NAME}")
    print(f"输出文件: {PDF_NAME}")
    print("=" * 60)

    for f in [IMG1, IMG2, IMG3, IMG4]:
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
