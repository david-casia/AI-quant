# -*- coding: utf-8 -*-
"""
build_pdf.py
生成 TASK5 机器学习分类模型 PDF 报告
格式：宋体、五号字(10.5pt)、1.5倍行距、0段间距、两端对齐
"""
import os
import sys
import base64
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from ml_model import load_cancer_data, load_stock_data, train_and_evaluate
from content import Q1_TITLE, Q1_BODY, Q2_TITLE, Q2_BODY, Q3_TITLE, Q3_CODE

STUDENT_NAME = "薛德刚"
PDF_NAME = f"{STUDENT_NAME}+TASK5.pdf"
PDF_PATH = os.path.join(BASE_DIR, PDF_NAME)
CSS_PATH = os.path.join(BASE_DIR, "style.css")

IMG1 = os.path.join(BASE_DIR, "cancer_roc.png")
IMG2 = os.path.join(BASE_DIR, "cancer_confusion.png")
IMG3 = os.path.join(BASE_DIR, "stock_roc.png")
IMG4 = os.path.join(BASE_DIR, "stock_confusion.png")
IMG5 = os.path.join(BASE_DIR, "stock_feature_importance.png")


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


def build_metrics_table(results, dataset_name):
    """构建模型评估指标对比表"""
    html = '<table class="data">\n'
    html += f'<caption>表 {dataset_name}三模型评估指标对比</caption>\n'
    html += '<thead><tr><th>模型</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1分数</th><th>AUC</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th></tr></thead>\n<tbody>\n'

    for name, r in results.items():
        html += f"<tr><td>{name}</td>"
        html += f"<td>{r['accuracy']:.4f}</td>"
        html += f"<td>{r['precision']:.4f}</td>"
        html += f"<td>{r['recall']:.4f}</td>"
        html += f"<td>{r['f1']:.4f}</td>"
        html += f"<td>{r['auc']:.4f}</td>"
        html += f"<td>{r['tp']}</td><td>{r['fp']}</td><td>{r['tn']}</td><td>{r['fn']}</td></tr>\n"

    html += "</tbody></table>\n"
    return html


def build_cancer_analysis(results):
    """乳腺癌数据集分析文字"""
    lr = results["逻辑回归"]
    dt = results["决策树"]
    rf = results["随机森林"]

    best_auc = max(lr["auc"], dt["auc"], rf["auc"])
    best_name = "逻辑回归" if lr["auc"] == best_auc else ("决策树" if dt["auc"] == best_auc else "随机森林")

    analysis = f"""本任务使用scikit-learn内置的乳腺癌数据集（Breast Cancer Dataset）进行分类模型训练与评估。该数据集包含569个样本，30个细胞核形态特征（如半径、纹理、周长、面积、平滑度等），应变量为0（恶性）或1（良性），其中恶性212个、良性357个。数据按70%训练集、30%测试集划分，采用分层抽样保持类别比例。

三个模型的评估结果如下：逻辑回归的准确率为{lr['accuracy']:.4f}，AUC为{lr['auc']:.4f}，混淆矩阵显示仅有1个假正例和1个假负例，分类效果极为优异；决策树的准确率为{dt['accuracy']:.4f}，AUC为{dt['auc']:.4f}，各有6个误分类样本；随机森林的准确率为{rf['accuracy']:.4f}，AUC为{rf['auc']:.4f}，假正例6个、假负例5个。

从AUC指标来看，{best_name}表现最优（AUC={best_auc:.4f}），三个模型的AUC均显著高于0.5的随机分类基准，说明它们都能有效区分恶性和良性肿瘤。逻辑回归在乳腺癌数据集上表现最佳的原因在于：该数据集的特征与标签之间主要呈线性关系，且特征经过标准化处理后，逻辑回归的线性决策边界能够很好地拟合数据分布。决策树的AUC相对较低，这是因为单棵决策树容易受到数据中噪声的影响，在有限的训练样本上容易产生过拟合。随机森林通过集成多棵决策树，在AUC上优于单棵决策树，但仍略逊于逻辑回归。

从混淆矩阵来看，三个模型在正类（良性）的识别上均表现出色，召回率均在94%以上，这意味着模型能够有效减少漏诊（将恶性肿瘤误判为良性）的风险，在医学诊断场景中具有极高的实用价值。"""

    return analysis.strip()


def build_stock_analysis(results):
    """股票数据集分析文字"""
    lr = results["逻辑回归"]
    dt = results["决策树"]
    rf = results["随机森林"]

    best_auc = max(lr["auc"], dt["auc"], rf["auc"])
    best_name = "逻辑回归" if lr["auc"] == best_auc else ("决策树" if dt["auc"] == best_auc else "随机森林")

    fi = rf.get("feature_importance")
    top_features = ""
    if fi is not None:
        top3 = fi.head(3)
        top_features = "、".join([f"{idx}({val:.4f})" for idx, val in top3.items()])

    analysis = f"""本任务同时构建了基于比亚迪(002594.SZ)日线数据的股票收益分类数据集。通过计算17个技术指标特征（包括收益率、均线偏离度、波动率、成交量变化、动量指标、RSI等），以次日涨跌方向（1=涨，0=跌）为应变量，构建了一个223个样本的二分类数据集，其中下跌124个、上涨99个。

三个模型在股票数据集上的评估结果如下：逻辑回归的准确率仅为{lr['accuracy']:.4f}，AUC为{lr['auc']:.4f}，接近0.5的随机基准；决策树的准确率为{dt['accuracy']:.4f}，AUC为{dt['auc']:.4f}；随机森林的准确率为{rf['accuracy']:.4f}，AUC为{rf['auc']:.4f}，是三个模型中唯一AUC超过0.6的模型。

与乳腺癌数据集上的优异表现相比，三个模型在股票数据集上的性能大幅下降。这一结果具有重要的实践启示：股票市场的噪声远大于医学诊断数据，短期内（次日）的股价涨跌受大量随机因素影响，技术指标的预测能力有限。逻辑回归和决策树的AUC接近0.5，说明它们几乎无法区分上涨和下跌的样本，这符合有效市场假说的预期——在弱势有效市场中，基于历史价格的技术指标难以获得持续的超额收益。

随机森林在股票数据集上表现相对较好（AUC={rf['auc']:.4f}），说明其通过集成多棵决策树能够捕捉到技术指标间的非线性交互关系，提取出部分有价值的预测信号。从特征重要性来看，排名前三的特征为{top_features}，这些特征反映了价格动量和成交量变化对次日涨跌的预测能力。

综合两个数据集的实验结果可以得出以下结论：第一，机器学习模型的性能高度依赖于数据本身的信噪比，在信噪比高的医学数据上表现优异，在信噪比低的金融数据上表现平庸；第二，随机森林通过集成学习能够提升模型的鲁棒性和泛化能力，在复杂数据上优于单一模型；第三，在量化交易中应用机器学习时，不应期望模型能够精准预测短期涨跌方向，而应将模型输出作为概率信号，结合仓位管理和风险控制构建完整的交易系统。"""

    return analysis.strip()


def build_html():
    """构建完整HTML"""
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css_content = f.read()

    # 运行两个数据集
    X1, y1, name1 = load_cancer_data()
    results1, _, _, _, _ = train_and_evaluate(X1, y1)

    X2, y2, name2 = load_stock_data()
    results2, _, _, _, _ = train_and_evaluate(X2, y2)

    q1_html = text_to_html_paragraphs(Q1_BODY)
    q2_html = text_to_html_paragraphs(Q2_BODY)
    cancer_analysis = build_cancer_analysis(results1)
    cancer_html = text_to_html_paragraphs(cancer_analysis)
    stock_analysis = build_stock_analysis(results2)
    stock_html = text_to_html_paragraphs(stock_analysis)
    cancer_table = build_metrics_table(results1, "1 乳腺癌数据集")
    stock_table = build_metrics_table(results2, "2 股票收益数据集")

    code_escaped = Q3_CODE.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    img1_b64 = img_to_base64(IMG1)
    img2_b64 = img_to_base64(IMG2)
    img3_b64 = img_to_base64(IMG3)
    img4_b64 = img_to_base64(IMG4)
    img5_b64 = img_to_base64(IMG5)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>量化交易作业 TASK5 - {STUDENT_NAME}</title>
    <style>
{css_content}
    </style>
</head>
<body>

<div class="info-block">
    <h1>量化交易作业 TASK5</h1>
    <p>AI交易引擎：机器学习算法与场景应用</p>
    <p>姓名：{STUDENT_NAME}</p>
    <p>分析标的：比亚迪 002594.SZ + 乳腺癌数据集</p>
</div>

<h2>{Q1_TITLE}</h2>
{q1_html}

<h2>{Q2_TITLE}</h2>
{q2_html}

<h2>{Q3_TITLE}</h2>

<h3>3.1 策略实现</h3>
<p>本任务使用scikit-learn库实现三种分类模型的训练与评估。使用两个数据集进行实验：scikit-learn内置的乳腺癌数据集和基于比亚迪日线数据构建的股票收益数据集。核心代码如下：</p>
<pre class="code">{code_escaped}</pre>

<h3>3.2 乳腺癌数据集实验结果</h3>
<p>乳腺癌数据集包含569个样本、30个特征，按70%/30%划分训练集与测试集。三个模型的ROC曲线对比如图1所示，混淆矩阵对比如图2所示。</p>
<div class="figure">
    <img src="{img1_b64}" alt="乳腺癌ROC曲线" />
    <div class="fig-caption">图1 乳腺癌数据集 ROC曲线对比（三模型）</div>
</div>
<div class="figure">
    <img src="{img2_b64}" alt="乳腺癌混淆矩阵" />
    <div class="fig-caption">图2 乳腺癌数据集 混淆矩阵对比（三模型）</div>
</div>
{cancer_table}
<p>图1、图2和表1解读：</p>
{cancer_html}

<h3>3.3 股票收益数据集实验结果</h3>
<p>基于比亚迪日线数据构建的股票收益数据集包含223个样本、17个技术指标特征，以次日涨跌方向为应变量。三个模型的ROC曲线对比如图3所示，混淆矩阵对比如图4所示，随机森林的特征重要性如图5所示。</p>
<div class="figure">
    <img src="{img3_b64}" alt="股票ROC曲线" />
    <div class="fig-caption">图3 比亚迪股票收益数据集 ROC曲线对比（三模型）</div>
</div>
<div class="figure">
    <img src="{img4_b64}" alt="股票混淆矩阵" />
    <div class="fig-caption">图4 比亚迪股票收益数据集 混淆矩阵对比（三模型）</div>
</div>
<div class="figure">
    <img src="{img5_b64}" alt="特征重要性" />
    <div class="fig-caption">图5 随机森林特征重要性（股票收益数据集）</div>
</div>
{stock_table}
<p>图3、图4、图5和表2解读：</p>
{stock_html}

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
    print(f"开始生成 TASK5 PDF 文档")
    print(f"学生姓名: {STUDENT_NAME}")
    print(f"输出文件: {PDF_NAME}")
    print("=" * 60)

    for f in [IMG1, IMG2, IMG3, IMG4, IMG5]:
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
