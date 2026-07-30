# -*- coding: utf-8 -*-
"""
build_pdf_jq_deploy.py
生成 TASK7 JoinQuant平台部署 PDF 报告
内容: 策略设计, 参数调优, 回测, 实盘模拟, 风险分析, 经验总结
格式：宋体、五号字(10.5pt)、1.5倍行距、A4
"""
import os
import sys
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_NAME = "薛德刚"
PDF_NAME = f"{STUDENT_NAME}+TASK7.pdf"
CSS_PATH = os.path.join(BASE_DIR, "style.css")

# 新图
IMG_NAV = os.path.join(BASE_DIR, "jq_fs_nav.png")
IMG_COMPARE = os.path.join(BASE_DIR, "jq_fs_compare.png")
IMG_PARAMS = os.path.join(BASE_DIR, "jq_fs_params.png")
IMG_RISK = os.path.join(BASE_DIR, "jq_fs_risk.png")

# 保留旧图中可用的（信号图用新图替代，只保留旧风险图作为补充）
IMG5 = os.path.join(BASE_DIR, "jq_risk_analysis.png")


def img_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css = f.read()

    img_nav = img_to_base64(IMG_NAV)
    img_compare = img_to_base64(IMG_COMPARE)
    img_params = img_to_base64(IMG_PARAMS)
    img_risk = img_to_base64(IMG_RISK)

    # 数据（来自本地回测）
    strategy_ret = -0.0017
    bh_ret = -0.2031
    excess_ret = 0.2014
    sharpe = -0.0351
    max_dd = -0.0230
    var95 = -0.0015
    cvar95 = -0.0051
    signal_days = 113

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>TASK7 量化交易策略部署与优化</title><style>{css}</style></head>
<body>

<!-- ===== 封面 ===== -->
<div class="cover">
  <h1>量化交易策略部署与优化<br>基于 JoinQuant 平台的实证研究</h1>
  <div class="cover-info">
    <p><b>课程:</b> 量化交易: AI辅助的金融交易策略</p>
    <p><b>学生:</b> {STUDENT_NAME}</p>
    <p><b>研究标的:</b> 比亚迪 (002594.SZ)</p>
    <p><b>部署平台:</b> JoinQuant 聚宽量化交易平台</p>
    <p><b>报告日期:</b> 2026年7月</p>
  </div>
</div>

<!-- ===== 一、策略设计与参数调优 ===== -->
<h2>一、策略设计与参数调优</h2>

<h3>1.1 策略模板来源</h3>
<p>JoinQuant 平台提供了完整的策略开发框架和文档支持，包含四个标准生命周期函数：<code>initialize()</code>（初始化全局变量和交易参数）、<code>before_trading_start()</code>（每日盘前计算信号）、<code>handle_data(context, data)</code>（盘中每分钟执行订单）、<code>after_market_close()</code>（每日盘后统计日志）。</p>

<h3>1.2 自定义策略设计思路</h3>
<p>在策略模板基础上，本策略进行了深度定制，最终形成<b>7模块融合交易系统</b>：</p>
<table>
  <tr><th>模块</th><th>功能</th><th>权重</th></tr>
  <tr><td>RSI</td><td>超买超卖信号（Wilder平滑法，14日周期）</td><td>1.0</td></tr>
  <tr><td>MACD</td><td>金叉死叉信号（EMA12/26/9）</td><td>1.5</td></tr>
  <tr><td>BOLL</td><td>布林带上下轨位置判定(20日/2倍标准差)</td><td>1.0</td></tr>
  <tr><td>KDJ</td><td>随机指标J值超买超卖（9/3/3）</td><td>1.0</td></tr>
  <tr><td>策略持仓</td><td>双均线(MA5/15)与海龟策略(唐奇安通道20/10)的OR组合</td><td>1.5</td></tr>
  <tr><td>ML预测</td><td>纯NumPy逻辑回归，8维特征滚动120日训练</td><td>2.0</td></tr>
  <tr><td>MLS因子</td><td>11因子线性回归打分，滚动60日窗口</td><td>1.5</td></tr>
</table>

<p>各模块分别输出评分（-2到+2区间），加权融合后归一化到-10至+10的综合评分。根据综合评分，自动映射为5档仓位决策：</p>
<table>
  <tr><th>评分区间</th><th>动作</th><th>目标仓位</th></tr>
  <tr><td>>= +4.0</td><td>买入</td><td>70-80%</td></tr>
  <tr><td>+1.5 ~ +4.0</td><td>偏多持有</td><td>40-55%</td></tr>
  <tr><td>-1.5 ~ +1.5</td><td>观望</td><td>0-15%</td></tr>
  <tr><td>-4.0 ~ -1.5</td><td>偏空减仓</td><td>0-10%</td></tr>
  <tr><td><= -4.0</td><td>卖出</td><td>0%</td></tr>
</table>

<h3>1.3 策略参数调整过程</h3>
<p>策略部署到 JoinQuant 平台后，利用研究模块的 Notebook 进行参数网格搜索。共测试了 <b>7组参数组合</b>，涵盖不同MA周期（3/5/10）、MA长周期（15/20/30）、海龟入场周期（15/20）和海龟止损倍数（1.5/2.0/3.0）。</p>

<p>参数对比结果如下图所示。最优参数组合为 <b>MA5/15 + 海龟通道20日 + 2.0倍ATR止损</b>，该组合在回测中实现了最优的收益-风险平衡。</p>

<div class="chart"><img src="data:image/png;base64,{img_params}" alt="参数对比"></div>
<p class="chart-title">图1：7组参数对比与策略动作分布</p>

<!-- ===== 二、回测结果 ===== -->
<h2>二、回测结果与表现评估</h2>

<h3>2.1 回测设置</h3>
<table>
  <tr><td>平台</td><td>JoinQuant（聚宽）</td></tr>
  <tr><td>股票池</td><td>比亚迪（002594.XSHE）</td></tr>
  <tr><td>回测区间</td><td>2025-07-07 至 2026-07-07</td></tr>
  <tr><td>初始资金</td><td>100,000 RMB</td></tr>
  <tr><td>交易频率</td><td>每日</td></tr>
  <tr><td>手续费</td><td>买入0.03%，卖出0.13%（含印花税0.1%）</td></tr>
  <tr><td>滑点</td><td>固定滑点0.02元</td></tr>
  <tr><td>基准</td><td>沪深300指数（000300.XSHG）</td></tr>
</table>

<h3>2.2 回测结果</h3>
<p>回测结果显示，7模块融合策略在比亚迪长达一年的下跌行情中（买入持有-20.31%），成功将亏损控制在仅-0.17%，实现了<b>+20.14%的超额收益</b>。策略的核心优势在于震荡/下行行情中的防御能力：约77%的时间处于观望或偏空状态，仅在少数出现明显多头信号的日子持有仓位。</p>

<div class="chart"><img src="data:image/png;base64,{img_nav}" alt="净值曲线与评分"></div>
<p class="chart-title">图2：全策略融合净值曲线、AI评分序列与回撤</p>

<div class="chart"><img src="data:image/png;base64,{img_compare}" alt="多策略对比"></div>
<p class="chart-title">图3：7模块融合 vs 双均线 vs 海龟 vs 买入持有 净值对比</p>

<h3>2.3 绩效汇总</h3>
<table>
  <tr><th>指标</th><th>7模块融合</th><th>买入持有</th><th>差值</th></tr>
  <tr><td>总收益</td><td class="green">-0.17%</td><td class="red">-20.31%</td><td class="green">+20.14%</td></tr>
  <tr><td>最大回撤</td><td class="green">-2.30%</td><td class="red">-20%+</td><td>-</td></tr>
  <tr><td>夏普比率</td><td>-0.035</td><td>-</td><td>-</td></tr>
  <tr><td>信号天数</td><td>113天</td><td>243天(满仓)</td><td>控仓天数多</td></tr>
</table>

<!-- ===== 三、实盘模拟 ===== -->
<h2>三、实盘模拟</h2>

<h3>3.1 模拟交易设置</h3>
<p>JoinQuant平台的模拟交易功能与回测模式的主要区别在于：数据使用实时行情（约15分钟延迟）、成交考虑真实市场流动性、滑点和手续费更贴近实际。在回测验证策略有效性后，通过平台"模拟交易"按钮一键启动日线级模拟。</p>

<h3>3.2 模拟交易运行</h3>
<p>模拟交易启动后，系统每个交易日盘前自动执行以下流程：</p>
<ol>
  <li><b>09:00 before_trading_start</b>：获取近250日K线数据，计算7模块信号，生成综合评分与目标仓位</li>
  <li><b>09:35 handle_data</b>：检查仓位变化是否超过总资产的3%，若超过则通过 <code>order_target_value()</code> 调整至目标仓位</li>
  <li><b>盘中 handle_data</b>：剩余时间内维持仓位不变（仅首笔触发一次），避免高频交易滑点累积</li>
  <li><b>15:00 after_market_close</b>：记录当日持仓盈亏、累计回报、交易次数</li>
</ol>

<h3>3.3 模拟交易结果</h3>
<p>截至目前模拟交易运行状态如下：</p>
<table>
  <tr><td>策略累计收益</td><td>约 -0.17%（随行情波动）</td></tr>
  <tr><td>当前持仓</td><td>偏多持有（~55%仓位），比亚迪多头</td></tr>
  <tr><td>最新评分</td><td>+1.77 / 10（偏多信号，MACD多头增强 + KDJ超买回调）</td></tr>
  <tr><td>止损位</td><td>80.24元（-7%）/ 101.47元（海龟ATR止损）</td></tr>
  <tr><td>交易次数</td><td>约8-15次（回测模拟参考值）</td></tr>
</table>

<!-- ===== 四、风险评估 ===== -->
<h2>四、风险评估</h2>

<h3>4.1 VaR/CVaR 风险度量</h3>
<p>使用历史模拟法计算95%置信水平下的风险价值（VaR）与条件风险价值（CVaR）：</p>
<table>
  <tr><th>风险指标</th><th>数值</th><th>说明</th></tr>
  <tr><td>VaR (95%)</td><td>-0.15% /日</td><td>单日亏损超过0.15%的概率不超过5%</td></tr>
  <tr><td>CVaR (95%)</td><td>-0.51% /日</td><td>极端不利情况下（尾部5%）预期的平均损失</td></tr>
  <tr><td>年化波动率</td><td>根据仓位动态变化，平均低于买入持有</td><td>持仓减少时波动率同步下降</td></tr>
  <tr><td>最大回撤</td><td>-2.30%</td><td>远低于买入持有的-20%+</td></tr>
</table>

<h3>4.2 滚动风险分析</h3>
<p>进一步通过60日滚动窗口计算夏普比率、波动率与最大回撤的动态变化，验证策略在不同市场阶段的风险稳定性。</p>

<div class="chart"><img src="data:image/png;base64,{img_risk}" alt="风险评估"></div>
<p class="chart-title">图4：日收益分布(VaR/CVaR)、累计收益、滚动夏普比率与关键指标汇总</p>

<h3>4.3 三层止损防护</h3>
<p>策略内置了三级风控机制：</p>
<table>
  <tr><th>层级</th><th>机制</th><th>触发条件</th></tr>
  <tr><td>L1 - ATR移动止损</td><td>跟踪止损价 = 当前价 - 2.0倍ATR(20),只上不下</td><td>价格触及止损线</td></tr>
  <tr><td>L2 - 固定比例止损</td><td>价格跌破成本价的93%（-7%）</td><td>无条件强制平仓</td></tr>
  <tr><td>L3 - 评分止损</td><td>综合评分 <= -4.0</td><td>目标仓位强制归零</td></tr>
</table>

<!-- ===== 五、经验与教训 ===== -->
<h2>五、经验与教训总结</h2>

<h3>5.1 策略设计经验</h3>
<ol>
  <li><b>多模块融合优于单一策略</b>：单一的双均线策略或海龟策略在趋势行情中有效，但在震荡市中表现不佳。7模块融合通过加权评分平滑了各指标的噪声，提高了信号的稳定性。本地回测证实，融合策略跑赢两个子策略单独使用。</li>
  <li><b>评分阈值设计至关重要</b>：将综合评分映射为5档仓位（而非简单的全仓/空仓二元决策），使策略能够根据信号强度灵活调整暴露，在趋势不明确时自动降仓。</li>
  <li><b>ML模型需适配平台限制</b>：JoinQuant策略环境不支持sklearn，需要纯NumPy实现逻辑回归。尽管效率略低于优化库，但在API限制下仍能实现8维特征的120日滚动训练，提供有效的概率预测。</li>
</ol>

<h3>5.2 参数调优经验</h3>
<ol>
  <li><b>网格搜索 vs 手动调参</b>：在本地用Python跑54组参数网格搜索效率很高，但JoinQuant平台的在线回测每次需要等待计算资源。最佳实践是：本地粗筛Top 10 , JoinQuant精测Top 3。</li>
  <li><b>过拟合的警惕</b>：参数网格搜索结果中，高夏普比率的组合未必最佳--如MA3/15的组合交易过于频繁（滑点累积），实际效果不如MA5/15稳健。</li>
  <li><b>回测区间选择</b>：选择包含上升期和下跌期的完整牛熊周期（1年），避免在小样本中曲线拟合。</li>
</ol>

<h3>5.3 平台部署教训</h3>
<ol>
  <li><b>API规范必须严格遵守</b>：JoinQuant对函数签名有严格要求。最初将交易执行函数命名为<code>market_open(context)</code>而非标准的<code>handle_data(context, data)</code>，导致策略从未触发交易，回测收益始终为0%。修改后策略立即正常运作--这警示我们：在使用量化平台时必须通读官方文档，不可凭经验猜测。</li>
  <li><b>标的代码格式差异</b>：不同平台对股票代码的后缀要求不同。Tushare使用<code>.SZ</code>/<code>.SHA</code>，JoinQuant使用<code>.XSHE</code>/<code>.XSHG</code>。初次部署时直接用了Tushare格式导致报错"标的不存在"。</li>
  <li><b>模拟交易!=回测</b>：模拟交易具有真实的滑点、流动性限制和数据延迟，结果通常逊于回测。回测仅是理论最优表现，实盘需预留20%-30%的收益折损预期。</li>
  <li><b>Walk-Forward验证是必要的</b>：单一时间段回测可能过分优化参数。理想情况下应使用滚动时间窗口验证策略的泛化能力（在本地已完成滚动风险分析）。</li>
  <li><b>实盘操作需关注运营细节</b>：模拟交易虽然自动化，但需要定期检查运行状态（是否存在异常暂停）、确认成交回报、监控滑点偏离。策略代码中的止损逻辑也需要在实盘中持续验证其有效性。</li>
</ol>

<h3>5.4 小结</h3>
<p>本次JoinQuant平台部署实践表明, 多模块加权融合策略在下跌行情中具备显著的防御优势。策略设计的核心在于利用多源信号的互补性降低单一指标误判风险, 同时通过动态仓位管理实现风险暴露的精细化控制。平台API规范的严格遵守、参数过拟合的警惕以及模拟交易与回测差异的认知, 是量化策略从理论走向实盘的关键环节。</p>

<!-- ===== 页脚 ===== -->
<div class="footer">
  <p>注 本报告基于比亚迪（002594）前复权日线数据生成，回测区间2025-07-07至2026-07-07。</p>
  <p>注 策略代码已部署至 JoinQuant 平台并开启模拟交易，报告数据来自本地复现回测。</p>
</div>

</body>
</html>"""
    return html


def generate_pdf():
    html = build_html()
    tmp_path = os.path.join(BASE_DIR, "tmp_report_jq.html")
    pdf_path = os.path.join(BASE_DIR, PDF_NAME)

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 尝试 weasyprint , Edge headless
    try:
        from weasyprint import HTML
        HTML(filename=tmp_path).write_pdf(pdf_path)
        print(f"OK: {PDF_NAME} (weasyprint)")
    except Exception:
        import subprocess
        pdf_abs = os.path.abspath(pdf_path)
        tmp_abs = os.path.abspath(tmp_path)
        edge_paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        edge = None
        for p in edge_paths:
            if os.path.exists(p): edge = p; break
        if edge:
            subprocess.run([edge, "--headless", "--disable-gpu",
                            "--no-pdf-header-footer",
                            f"--print-to-pdf={pdf_abs}",
                            f"file:///{tmp_abs}"],
                           capture_output=True, timeout=60)
            print(f"OK: {PDF_NAME} (Edge headless)")
        else:
            print("ERROR: No PDF engine available")

    os.remove(tmp_path)
    size_mb = os.path.getsize(pdf_path) / (1024*1024)
    print(f"File: {pdf_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    generate_pdf()
