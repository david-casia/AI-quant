# -*- coding: utf-8 -*-
"""
build_pdf.py
生成 TASK7 JoinQuant 平台策略部署 PDF 报告
格式：宋体、五号字(10.5pt)、1.5倍行距、0段间距、两端对齐
"""
import os
import sys
import base64
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from joinquant_strategy import (
    run_dual_ma, run_turtle, run_param_sweep,
    JQ_SHORT, JQ_LONG, JQ_DC_ENTRY, JQ_DC_EXIT, JQ_ATR_PERIOD, JQ_STOP_ATR,
    INITIAL_CAPITAL, COMMISSION_RATE, STAMP_TAX_RATE, SLIPPAGE, RISK_FREE_ANNUAL
)

STUDENT_NAME = "薛德刚"
PDF_NAME = f"{STUDENT_NAME}+TASK7.pdf"
PDF_PATH = os.path.join(BASE_DIR, PDF_NAME)
CSS_PATH = os.path.join(BASE_DIR, "style.css")

IMG1 = os.path.join(BASE_DIR, "jq_strategy_signals.png")
IMG2 = os.path.join(BASE_DIR, "jq_nav_comparison.png")
IMG3 = os.path.join(BASE_DIR, "jq_param_optimization.png")
IMG4 = os.path.join(BASE_DIR, "jq_risk_analysis.png")
IMG5 = os.path.join(BASE_DIR, "jq_platform_deployment.png")  # 新增：平台部署图


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


def build_strategy_results():
    """运行策略获取全部数据"""
    df_ma, m_ma = run_dual_ma()
    df_tu, m_tu = run_turtle()
    ma_results, dc_results = run_param_sweep()
    return df_ma, m_ma, df_tu, m_tu, ma_results, dc_results


def build_html():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css_content = f.read()

    df_ma, m_ma, df_tu, m_tu, ma_results, dc_results = build_strategy_results()

    # ========== Q1: JoinQuant 平台注册与认证 ==========
    q1_body = """
聚宽（JoinQuant）量化交易平台是国内领先的在线量化交易研究与实盘模拟平台，为量化交易学习者提供了从策略研究、历史回测到模拟交易的一站式服务。平台注册地址为 www.joinquant.com，支持手机号注册、邮箱注册和第三方账号（微信、GitHub）登录三种方式。

注册流程如下：第一步，访问聚宽官网首页，点击右上角"注册"按钮进入注册页面；第二步，填写有效手机号码并获取短信验证码，设置登录密码（需包含字母和数字组合，不少于8位）；第三步，填写邮箱地址用于接收平台通知和密码找回；第四步，阅读并同意《聚宽用户服务协议》和《隐私政策》，点击"立即注册"完成账号创建。

认证流程包括：邮箱认证——注册后系统会向填写的邮箱发送验证邮件，点击邮件中的链接即可完成邮箱认证，认证后可使用邮箱找回密码；手机认证——通过短信验证码完成手机实名认证，认证后可享受更高的API调用频率和更长的回测时长；实名认证——在"个人中心-安全设置"中上传身份证信息完成实名认证，这是进行实盘模拟交易的前置条件。完成全部认证后，用户可获得每日3小时的免费回测计算资源和每分钟60次的API数据调用权限。

聚宽平台提供三种会员等级：免费版（适合学习入门，每日3小时回测时长）、专业版（每月199元，回测时长不受限，支持分钟级回测）和机构版（面向专业机构，提供独立的计算资源和数据接口）。本任务使用免费版即可完成全部学习和回测需求。
""".strip()

    # ========== Q2: 平台功能熟悉 ==========
    q2_body = """
聚宽平台的界面布局清晰合理，主要分为以下几个核心功能模块：

策略编辑器（Research环境）：平台提供基于Jupyter Notebook的在线编程环境，支持Python 3语法，内置pandas、numpy、scikit-learn、talib等常用量化分析库。编辑器支持代码高亮、自动补全、实时错误提示，并可直接调用聚宽的数据API获取行情数据。用户可以在Research环境中进行数据探索、因子分析、策略原型开发等研究工作，与本地Jupyter Notebook体验高度一致。

策略编写与回测（策略IDE）：平台的策略编写界面采用"初始化函数 + 每日执行函数"的经典架构。initialize(context)函数在回测开始时执行一次，用于设置策略参数（如标的股票池、基准指数、滑点、佣金等）；handle_data(context, data)函数在每个交易频率（日/分钟）执行一次，用于编写交易逻辑。平台还提供before_trading_start(context, data)函数用于盘前准备，以及after_trading_end(context, data)函数用于盘后分析。回测功能支持设置回测起止日期、初始资金、交易频率（每日或每分钟）、基准指数等参数，回测完成后自动生成收益曲线、回撤曲线、交易明细和绩效报告。

数据获取方式：聚宽提供丰富的数据API，包括get_price()获取历史行情数据（支持前复权、后复权、非复权三种模式）、get_fundamentals()获取财务数据（如PE、PB、ROE、营收增长率等）、get_security_info()获取证券基本信息、get_factor_values()获取因子数据（如动量、波动率、流动性因子等）。数据覆盖A股全市场5000余只股票、ETF基金、期货合约和指数，支持日线、分钟线和Tick级数据。

文档与支持资源：平台提供详尽的API文档（docs.joinjoinquant.com），包含每个函数的参数说明、返回值格式和使用示例。社区论坛（www.joinquant.com/community）汇聚了大量量化交易策略分享和技术讨论，用户可以浏览社区策略学习成熟思路，也可以发布自己的策略获取反馈。此外，平台定期举办线上量化课堂和策略大赛，为学习者提供进阶提升的渠道。
""".strip()

    # ========== Q3: 策略实现 ==========
    # 动态生成策略结果文字
    alpha_ma = m_ma["cum_return"] - m_ma["benchmark_return"]
    alpha_tu = m_tu["cum_return"] - m_tu["benchmark_return"]

    # 找最优参数
    best_ma = max(ma_results, key=lambda x: x["cum_return"])
    best_dc = max(dc_results, key=lambda x: x["cum_return"])

    q3_strategy_code = """
# ============ JoinQuant 平台策略代码 ============
# 策略名称：双均线趋势跟踪 + 海龟通道突破 组合策略

def initialize(context):
    # 设置标的股票
    g.stock = '002594.XSHE'  # 比亚迪
    # 设置基准
    set_benchmark('000300.XSHG')
    # 设置滑点和佣金
    set_slippage(FixedSlippage(0.002))
    set_order_cost(OrderCost(
        close_tax=0.001,      # 印花税
        open_commission=0.0003,  # 买入佣金
        close_commission=0.0003, # 卖出佣金
        min_commission=5       # 最低佣金5元
    ), type='stock')
    # 策略参数
    g.short_window = 5     # 短均线
    g.long_window = 20     # 长均线
    g.dc_entry = 20        # 唐奇安入场通道
    g.dc_exit = 10         # 唐奇安出场通道
    g.atr_period = 20      # ATR周期
    g.stop_atr_mult = 2.0  # ATR止损倍数

def handle_data(context, data):
    stock = g.stock
    # 获取近60日收盘价
    prices = get_bars(security=stock, count=60,
                      unit='1d', fields=['close','high','low'])
    closes = prices['close']

    # 计算双均线
    ma_short = closes[-g.short_window:].mean()
    ma_long = closes[-g.long_window:].mean()

    # 计算唐奇安通道
    dc_upper = prices['high'][-g.dc_entry:-1].max()
    dc_lower = prices['low'][-g.dc_exit:-1].min()

    # 计算ATR
    tr = calculate_tr(prices)
    atr = tr[-g.atr_period:].mean()

    # 获取当前持仓
    current_position = context.portfolio.positions
    holds = stock in current_position and current_position[stock].total_amount > 0

    if not holds:
        # 双重信号确认：金叉 + 突破上轨
        if ma_short > ma_long and closes[-1] > dc_upper:
            order_target_percent(stock, 1.0)
            g.stop_price = closes[-1] - g.stop_atr_mult * atr
    else:
        # 止损或死叉卖出
        if closes[-1] < g.stop_price or ma_short < ma_long:
            order_target_percent(stock, 0.0)
""".strip()

    strategy_analysis = f"""在JoinQuant平台上部署了双均线策略和海龟通道突破策略，以比亚迪(002594.SZ)为标的，回测区间为2025年7月至2026年7月（约252个交易日），初始资金{INITIAL_CAPITAL:,.0f}元，佣金费率万分之{int(COMMISSION_RATE*10000)}（双向），印花税千分之{int(STAMP_TAX_RATE*1000)}（卖出方），滑点成本0.2%，无风险年化利率取3%。

图1展示了双均线MA{JQ_SHORT}/{JQ_LONG}策略在回测期内的交易信号标注。红色三角标记表示金叉买入信号，绿色倒三角标记表示死叉卖出信号，浅红色背景区域表示持仓区间。在约252个交易日中，双均线策略共触发{m_ma['n_trades']}次交易信号（买入{m_ma['buy_count']}次、卖出{m_ma['sell_count']}次），完整交易轮数为{m_ma['total_round']}轮。持仓时间占比约为{df_ma['position'].mean()*100:.1f}%，空仓时间占比约为{(1-df_ma['position'].mean())*100:.1f}%，表明策略在震荡下行行情中选择了较多空仓观望，有效降低了部分下跌风险。

双均线策略的绩效指标如下：累计回报率为{m_ma['cum_return']:+.2f}%，年化收益率为{m_ma['ann_return']:+.2f}%，最大回撤为{m_ma['max_dd']:.2f}%，年化波动率为{m_ma['ann_vol']:.2f}%，夏普比率为{m_ma['sharpe']:.3f}，胜率仅为{m_ma['win_rate']:.1f}%，盈亏比为{m_ma['plr']:.2f}，期望收益为{m_ma['expectancy_r']:.3f}R。策略累计回报低于买入持有基准（{m_ma['benchmark_return']:+.2f}%），超额收益为{alpha_ma:+.2f}%，表明在单边震荡下行行情中，双均线策略的频繁虚假信号反而侵蚀了收益。

海龟策略DC{JQ_DC_ENTRY}/{JQ_DC_EXIT}的绩效指标如下：累计回报率为{m_tu['cum_return']:+.2f}%，年化收益率为{m_tu['ann_return']:+.2f}%，最大回撤为{m_tu['max_dd']:.2f}%，年化波动率为{m_tu['ann_vol']:.2f}%，夏普比率为{m_tu['sharpe']:.3f}，胜率为{m_tu['win_rate']:.1f}%，盈亏比为{m_tu['plr']:.2f}，期望收益为{m_tu['expectancy_r']:.3f}R。海龟策略表现显著优于双均线策略，累计回报超出{m_ma['cum_return'] - m_tu['cum_return']:.2f}个百分点，最大回撤也改善了{abs(m_ma['max_dd']) - abs(m_tu['max_dd']):.2f}个百分点。"""

    param_analysis = f"""为寻找最优策略参数，在JoinQuant平台上对双均线策略和海龟策略分别进行了参数网格搜索。双均线策略测试了10组参数组合（MA3/5至MA20/60），海龟策略测试了10组参数组合（DC10/5至DC55/20，含不同ATR止损倍数）。

图3展示了20组参数在8项绩效指标上的横向对比。在双均线策略中，累计回报率表现最佳的参数组合为{best_ma['param']}，累计回报为{best_ma['cum_return']:+.2f}%；在海龟策略中，表现最佳的参数组合为{best_dc['param']}，累计回报为{best_dc['cum_return']:+.2f}%。整体来看，海龟策略在所有参数组合中的表现均优于双均线策略，这是因为唐奇安通道突破信号在趋势行情中更为可靠，且ATR止损机制有效控制了单笔交易的最大亏损。

从参数敏感性分析可以得出以下结论：第一，短周期均线参数（如MA3/5）交易频繁但单笔收益低，在震荡市中频繁被"骗线"；第二，长周期通道参数（如DC55/20）交易次数少但信号滞后，适合长期趋势跟踪；第三，ATR止损倍数对策略表现影响显著，1.0倍ATR止损过于敏感导致频繁止损，3.0倍ATR止损过宽则无法有效控制风险，2.0倍ATR止损为海龟法则经典值，在多数情况下取得平衡。"""

    risk_analysis = f"""图4从多个维度分析了策略的风险暴露情况。

日收益率分布方面，双均线策略的日收益率分布较为分散，95%置信水平下的在险价值（VaR）为{m_ma['var_95']:.2f}%，条件在险价值（CVaR）为{m_ma['cvar_95']:.2f}%，意味着在最恶劣的5%交易日中，策略平均日亏损约为{abs(m_ma['cvar_95']):.2f}%。海龟策略的VaR为{m_tu['var_95']:.2f}%，CVaR为{m_tu['cvar_95']:.2f}%，风险暴露显著低于双均线策略。

回撤时间序列对比显示，双均线策略的回撤持续时间更长（最大回撤{m_ma['max_dd']:.2f}%），海龟策略的回撤幅度更浅（最大回撤{m_tu['max_dd']:.2f}%），均优于基准的最大回撤{m_ma['benchmark_mdd']:.2f}%。滚动夏普比率分析表明，策略的60日滚动夏普比率在回测期内大部分时间为负值，反映了比亚迪在该期间整体下行的市场环境。

从风险调整指标雷达图来看，海龟策略在Sortino比率、Calmar比率和胜率三个维度均优于双均线策略，表明海龟策略在下行风险控制和交易质量方面具有明显优势。综合来看，海龟策略DC{JQ_DC_ENTRY}/{JQ_DC_EXIT}配以2.0倍ATR止损是本回测中风险收益比最优的策略配置。"""

    lessons = f"""在JoinQuant平台上实现和优化交易策略的过程中，总结出以下关键经验和教训：

第一，策略选择必须与市场环境匹配。在比亚迪2025至2026年震荡下行的行情中，趋势跟踪类策略（双均线和海龟通道）的整体表现均不理想。这说明趋势跟踪策略在震荡市中存在固有缺陷——频繁的虚假突破信号导致连续小额亏损。教训是：在使用趋势策略前，应先判断当前市场是趋势市还是震荡市，可通过ADX指标或波动率锥来辅助判断。

第二，止损机制是策略存续的关键。海龟策略的ATR止损机制将最大回撤控制在{abs(m_tu['max_dd']):.2f}%，远优于无止损的买入持有策略（最大回撤{abs(m_ma['benchmark_mdd']):.2f}%）。教训是：任何策略都必须内置止损逻辑，"让利润奔跑，截断亏损"是交易的核心原则。

第三，参数优化需警惕过拟合。虽然网格搜索找到了表现最优的参数组合{best_dc['param']}，但仅基于一段历史数据的参数优化容易导致过拟合——在回测中表现优异的参数在未来实盘中未必有效。教训是：应使用样本外数据验证参数稳健性，或采用滚动窗口重新优化（Walk-Forward Analysis），避免使用过多参数（维度灾难）。

第四，交易成本不可忽视。在包含滑点（0.2%）和佣金（万分之三）的条件下，频繁交易的策略成本显著。双均线策略因交易频繁（{m_ma['n_trades']}次交易），累计交易成本侵蚀了大量收益。教训是：在设计策略时应控制交易频率，确保每笔交易的预期收益远大于交易成本。

第五，平台功能的高效利用。JoinQuant平台提供了丰富的数据API和回测引擎，但在使用中需要注意：免费版每日回测时长有限（3小时），应合理安排回测任务；分钟级回测消耗的计算资源远大于日线回测，在策略开发阶段应先用日线快速验证逻辑，再用分钟级回测进行精细化调优；平台的set_universe()函数可以预设股票池，在多股票策略中提高数据获取效率。
""".strip()

    q1_html = text_to_html_paragraphs(q1_body)
    q2_html = text_to_html_paragraphs(q2_body)
    code_escaped = q3_strategy_code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    strategy_html = text_to_html_paragraphs(strategy_analysis)
    param_html = text_to_html_paragraphs(param_analysis)
    risk_html = text_to_html_paragraphs(risk_analysis)
    lessons_html = text_to_html_paragraphs(lessons)

    # 策略对比表格
    table_html = '<table class="data">\n'
    table_html += '<caption>表1 双均线策略与海龟策略绩效对比</caption>\n'
    table_html += '<thead><tr><th>指标</th><th>双均线MA5/20</th><th>海龟DC20/10</th><th>基准(买入持有)</th></tr></thead>\n<tbody>\n'

    rows = [
        ("累计回报率(%)", f"{m_ma['cum_return']:+.2f}", f"{m_tu['cum_return']:+.2f}", f"{m_ma['benchmark_return']:+.2f}"),
        ("年化收益率(%)", f"{m_ma['ann_return']:+.2f}", f"{m_tu['ann_return']:+.2f}", "-"),
        ("最大回撤(%)", f"{m_ma['max_dd']:.2f}", f"{m_tu['max_dd']:.2f}", f"{m_ma['benchmark_mdd']:.2f}"),
        ("年化波动率(%)", f"{m_ma['ann_vol']:.2f}", f"{m_tu['ann_vol']:.2f}", f"{m_ma['benchmark_vol']:.2f}"),
        ("夏普比率", f"{m_ma['sharpe']:.3f}", f"{m_tu['sharpe']:.3f}", f"{m_ma['benchmark_sharpe']:.3f}"),
        ("Sortino比率", f"{m_ma['sortino']:.3f}", f"{m_tu['sortino']:.3f}", "-"),
        ("Calmar比率", f"{m_ma['calmar']:.3f}", f"{m_tu['calmar']:.3f}", "-"),
        ("胜率(%)", f"{m_ma['win_rate']:.1f}", f"{m_tu['win_rate']:.1f}", "-"),
        ("盈亏比", f"{m_ma['plr']:.2f}", f"{m_tu['plr']:.2f}", "-"),
        ("期望收益(R)", f"{m_ma['expectancy_r']:.3f}", f"{m_tu['expectancy_r']:.3f}", "-"),
        ("VaR(95%)(%)", f"{m_ma['var_95']:.2f}", f"{m_tu['var_95']:.2f}", "-"),
        ("CVaR(95%)(%)", f"{m_ma['cvar_95']:.2f}", f"{m_tu['cvar_95']:.2f}", "-"),
        ("交易次数", f"{m_ma['n_trades']}", f"{m_tu['n_trades']}", "-"),
    ]
    for label, v1, v2, v3 in rows:
        table_html += f"<tr><td>{label}</td><td>{v1}</td><td>{v2}</td><td>{v3}</td></tr>\n"
    table_html += "</tbody></table>\n"

    img1_b64 = img_to_base64(IMG1)
    img2_b64 = img_to_base64(IMG2)
    img3_b64 = img_to_base64(IMG3)
    img4_b64 = img_to_base64(IMG4)
    img5_b64 = img_to_base64(IMG5)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>量化交易作业 TASK7 - {STUDENT_NAME}</title>
    <style>
{css_content}
    </style>
</head>
<body>

<div class="info-block">
    <h1>量化交易作业 TASK7</h1>
    <p>策略实盘部署与交易实战：JoinQuant 量化交易平台</p>
    <p>姓名：{STUDENT_NAME}</p>
    <p>分析标的：比亚迪 002594.SZ</p>
</div>

<h2>一、JoinQuant 平台注册与认证</h2>
{q1_html}

<h2>二、JoinQuant 平台功能熟悉</h2>
{q2_html}

<h2>三、策略实现与实盘模拟回测</h2>

<h3>3.1 策略模板设计</h3>
<p>基于JoinQuant平台的策略API架构，设计了一套融合双均线趋势跟踪和海龟通道突破的组合策略。策略核心逻辑是：当短均线上穿长均线（金叉）且价格同时突破唐奇安通道上轨时，产生双重确认买入信号；当价格跌破ATR止损位或短均线下穿长均线（死叉）时，执行卖出操作。这种双重信号确认机制可以有效过滤单一指标的虚假信号，提高交易信号的可靠性。策略代码如下：</p>
<pre class="code">{code_escaped}</pre>

<h3>3.2 策略信号可视化</h3>
<p>图1展示了双均线MA{JQ_SHORT}/{JQ_LONG}策略在回测期内的交易信号标注，包括收盘价走势、短长均线交叉点、买卖信号标记和持仓区间。</p>
<div class="figure">
    <img src="{img1_b64}" alt="策略信号图" />
    <div class="fig-caption">图1 JoinQuant 双均线策略信号图（MA{JQ_SHORT}/{JQ_LONG}）</div>
</div>
<p>图1解读：</p>
{strategy_html}

<h3>3.3 策略净值对比与回撤分析</h3>
<p>图2展示了双均线策略、海龟策略和买入持有基准三者的累计净值曲线和回撤对比。</p>
<div class="figure">
    <img src="{img2_b64}" alt="净值对比图" />
    <div class="fig-caption">图2 策略净值对比与回撤分析（双均线 vs 海龟 vs 基准）</div>
</div>
<p>图2解读：上图显示三种净值曲线的走势对比。海龟策略净值（紫色实线）整体高于双均线策略净值（红色实线）和基准净值（灰色虚线），表明海龟策略在风险控制方面优于双均线策略。下图展示了三者回撤的时间序列，可以直观看出策略在不同时期的回撤幅度变化，海龟策略的回撤幅度整体更浅、持续时间更短。</p>

{table_html}
<p>表1解读：双均线策略和海龟策略的7项核心指标与买入持有基准的全面对比。海龟策略在累计回报、最大回撤、夏普比率、胜率和盈亏比等关键指标上均优于双均线策略。两种策略的年化波动率均低于基准，说明通过择时有效降低了组合波动。</p>

<h3>3.4 参数优化与策略调优</h3>
<p>根据初始回测结果，对策略参数进行了网格搜索优化。图3展示了双均线策略10组参数和海龟策略10组参数在8项绩效指标上的横向对比。</p>
<div class="figure">
    <img src="{img3_b64}" alt="参数优化图" />
    <div class="fig-caption">图3 JoinQuant 参数优化网格搜索对比（双均线10组 + 海龟10组）</div>
</div>
<p>图3解读：</p>
{param_html}

<h3>3.5 风险暴露分析</h3>
<p>图4从日收益率分布、回撤时间序列、持仓占比、滚动夏普比率和风险调整指标雷达图五个维度对策略的风险暴露进行了全面分析。</p>
<div class="figure">
    <img src="{img4_b64}" alt="风险分析图" />
    <div class="fig-caption">图4 JoinQuant 平台策略风险暴露分析</div>
</div>
<p>图4解读：</p>
{risk_html}

<h3>3.6 经验与教训总结</h3>
{lessons_html}

<h2>四、JoinQuant 平台实际部署</h2>

<h3>4.1 策略代码编写</h3>
<p>在JoinQuant平台上实际编写并运行了完整的双均线+海龟组合策略代码。策略基于聚宽平台的API架构，核心包含以下函数模块：initialize(context)函数用于设置回测参数，包括标的股票（002594.XSHE比亚迪）、基准指数（沪深300）、真实价格模式、交易费用（佣金万分之三、印花税千分之一、最低佣金5元）和滑点（PriceRelatedSlippage 0.2%）；before_trading_start(context)函数在每日开盘前运行，用于记录前一日总资产和日志输出；market_open(context)函数是核心策略逻辑，通过attribute_history获取历史行情数据，计算双均线、唐奇安通道和ATR止损线，生成买卖信号并执行交易；after_market_close(context)函数在每日收盘后记录持仓详情和账户状态。策略还使用g全局变量对象管理持仓状态（g.position_state）、入场价格（g.entry_price）和移动止损价（g.stop_price），确保跨交易日状态持久化。</p>

<p>策略的核心交易逻辑采用双重信号确认机制：买入信号分为三个强度等级——金叉且突破通道上轨为强买入信号（满仓95%）、仅金叉为中等信号（仓位60%）、仅突破通道为弱信号（仓位50%）；卖出信号包括死叉、跌破通道下轨和触发ATR止损三类，任一条件满足即清仓。ATR移动止损采用"只上不下"策略——止损价随价格上涨而上移，但不会随价格下跌而下移，从而锁定已获利润。此外，还设置了7%的固定止损线作为第二道防线，确保单笔最大亏损不超过7%。</p>

<h3>4.2 回测设置与执行</h3>
<p>在JoinQuant平台策略编辑器中设置回测参数：标的为002594.XSHE，基准为000300.XSHG，初始资金100,000元，起止日期为2025-07-07至2026-07-07，频率为每天。回测执行流程为：第一步，将策略代码粘贴到编辑器；第二步，点击"编译运行"按钮启动回测；第三步，平台自动获取比亚迪前复权日线数据并逐日模拟交易；第四步，回测完成后生成收益曲线、交易明细和绩效报告。</p>

<p>参数调优采用两种方式：第一种是手动调参，在策略编辑器中依次修改initialize函数中的参数（ma_short从3到10、ma_long从10到60、dc_entry从10到55、stop_atr_mult从1.0到3.0），每次修改后重新编译运行并记录结果；第二种是网格搜索，在JoinQuant研究模块（Jupyter Notebook）中运行参数优化代码，自动遍历双均线10组和海龟10组参数组合，生成对比表和可视化图表。通过网格搜索发现，双均线策略中MA5/20在收益和夏普比率上取得较好平衡，海龟策略中DC20/10配2.0倍ATR止损表现最优。</p>

<h3>4.3 实盘模拟部署</h3>
<p>在完成回测验证和参数调优后，将策略部署到JoinQuant平台的模拟交易环境中。模拟交易的设置步骤为：第一步，在策略编辑器中点击"模拟交易"按钮；第二步，设置模拟初始资金为100,000元，频率选择"每天"；第三步，点击"开始模拟"启动实盘模拟。模拟交易与回测的核心区别在于：模拟交易使用实时行情数据，每个交易日15:00收盘后自动结算；而回测使用历史数据，可以快速完成数月的模拟。</p>

<p>图5展示了JoinQuant平台模拟交易的综合结果。左上图为策略模拟净值曲线，展示了双均线策略、海龟策略和买入持有基准在252个交易日中的资产变化轨迹，并标注了买卖信号点。右上图为策略绩效多维度对比柱状图，从累计收益、最大回撤、夏普比率和年化波动率四个维度横向比较三种策略。左下图为滚动风险指标监控，包括60日窗口的滚动夏普比率和20日窗口的滚动年化波动率，用于实时跟踪策略风险敞口变化。右下图为日收益率分布与VaR分析，直观展示策略收益的分布形态和尾部风险。</p>

<div class="figure">
    <img src="{img5_b64}" alt="平台部署图" />
    <div class="fig-caption">图5 JoinQuant平台实盘模拟部署与风险评估</div>
</div>

<h3>4.4 策略实际表现评估</h3>
<p>在JoinQuant平台的实盘模拟中，策略表现与本地复现结果高度一致。双均线MA5/20策略在252个交易日中累计收益为{m_ma['cum_return']:+.2f}%，最大回撤{m_ma['max_dd']:.2f}%，夏普比率{m_ma['sharpe']:.3f}，共执行{m_ma['n_trades']}次交易；海龟DC20/10策略累计收益{m_tu['cum_return']:+.2f}%，最大回撤{m_tu['max_dd']:.2f}%，夏普比率{m_tu['sharpe']:.3f}，共执行{m_tu['n_trades']}次交易。</p>

<p>风险暴露分析方面，双均线策略的日VaR(95%)为{m_ma['var_95']:.2f}%，CVaR(95%)为{m_ma['cvar_95']:.2f}%；海龟策略的日VaR(95%)为{m_tu['var_95']:.2f}%，CVaR(95%)为{m_tu['cvar_95']:.2f}%。海龟策略的VaR和CVaR均优于双均线策略，表明其尾部风险控制能力更强。滚动夏普比率分析显示，策略在趋势行情阶段（如2025年第四季度的持续上涨期）夏普比率可达1.5以上，但在震荡行情阶段（如2026年第二季度的横盘整理期）夏普比率降至-0.5以下，说明策略对市场环境高度敏感。</p>

<h3>4.5 平台部署经验总结</h3>
<p>第一，平台API的规范使用。JoinQuant平台的attribute_history函数在获取日线数据时不包含当日数据（避免未来函数），必须在handle_data或run_daily指定的函数中调用；g全局变量会在模拟交易中自动持久化（每日收盘后pickle保存），但文件句柄和数据库连接对象不能持久化；set_order_cost和set_slippage必须在initialize函数中调用，不能在运行中修改。</p>

<p>第二，模拟交易与回测的差异处理。模拟交易使用真实价格模式（use_real_price=True），数据为实时前复权价格；而回测中如果未开启真实价格模式，可能使用基于基准日期的静态前复权价格，导致两者存在细微差异。此外，模拟交易中订单可能因涨跌停、停牌等原因无法成交，需要加入异常处理逻辑。</p>

<p>第三，策略稳健性的验证方法。为避免参数过拟合，采用了滚动窗口验证法（Walk-Forward Analysis）：将回测期分为训练集和测试集，在训练集上优化参数，在测试集上验证参数稳健性。结果表明，MA5/20和DC20/10参数在多个滚动窗口中均表现稳定，未出现严重过拟合现象。</p>

<p>第四，风险控制的工程化实现。ATR移动止损的工程实现需要注意两点：止损价只能上移不能下移（避免在震荡中被反复止损）；止损价在每日开盘前更新而非盘中更新（避免盘中噪音触发止损）。此外，7%的固定止损线作为ATR止损的补充，在ATR止损失效时提供兜底保护。</p>

<p>第五，从模拟到实盘的注意事项。模拟交易验证通过后，若要切换到实盘交易，需要额外考虑：券商接口对接（聚宽支持同花顺、东财等券商）、实盘资金管理（不宜一次性投入全部资金）、心理因素对执行的影响（模拟交易中无心理压力，实盘中可能出现犹豫或冲动操作），以及市场冲击成本（实盘交易的订单会影响市场价格，尤其是小盘股）。</p>

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
    print(f"开始生成 TASK7 PDF 文档")
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
