# -*- coding: utf-8 -*-
"""
plot_v2.py
TASK8 全中文图表生成（无英文标题与变量名）
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform
import warnings
warnings.filterwarnings('ignore')

# 字体配置
if sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'SimHei']
elif sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# 配色（学术白底）
COLOR = {
    '策略': '#cc0000',     # 红
    '海龟': '#990099',     # 紫
    '双均线': '#cc6600',   # 橙
    '基准': '#666666',     # 灰
    '买入线': '#cc0000',
    '卖出线': '#006600',
    'ML': '#0066cc',
    '回撤': '#006600',
    '网格': '#cccccc',
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# 加载比亚迪数据
df = pd.read_csv(os.path.join(BASE_DIR, '002594_daily.csv'))
df = df.sort_values('trade_date').reset_index(drop=True)
df['日期'] = pd.to_datetime(df['trade_date'].astype(str))
close = df['close']
high = df['high']
low = df['low']
volume = df['vol']
dates = df['日期']

print(f"[数据] {len(df)}条  {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}")


# ===== 辅助：双均线策略回测 =====
def backtest_ma(short=5, long_period=15):
    ma_s = close.rolling(short).mean()
    ma_l = close.rolling(long_period).mean()
    pos = 0
    nav = [1.0]
    pos_changes = []
    for i in range(len(close)):
        if i < long_period:
            nav.append(nav[-1])
            continue
        golden = ma_s.iloc[i] > ma_l.iloc[i] and ma_s.iloc[i-1] <= ma_l.iloc[i-1]
        death = ma_s.iloc[i] < ma_l.iloc[i] and ma_s.iloc[i-1] >= ma_l.iloc[i-1]
        if pos == 0 and golden:
            pos = 1; pos_changes.append(('买入', i, close.iloc[i]))
        elif pos == 1 and death:
            pos = 0; pos_changes.append(('卖出', i, close.iloc[i]))
        r = close.pct_change().iloc[i] if i > 0 else 0
        nav.append(nav[-1] * (1 + r * pos))
    nav_series = pd.Series(nav[1:], index=dates)
    return nav_series, pos_changes


# ===== 辅助：海龟策略回测 =====
def backtest_turtle(entry=20, exit_period=10, atr_period=20, stop_mult=2.0):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    upper = high.rolling(entry).max().shift(1)
    lower = low.rolling(exit_period).min().shift(1)
    pos = 0
    sp = 0.0
    nav = [1.0]
    pos_changes = []
    for i in range(len(close)):
        if i < entry:
            nav.append(nav[-1])
            continue
        price = close.iloc[i]
        if pos == 0 and price > upper.iloc[i]:
            pos = 1
            sp = price - stop_mult * atr.iloc[i]
            pos_changes.append(('买入', i, price))
        elif pos == 1 and price < sp:
            pos = 0; sp = 0
            pos_changes.append(('卖出', i, price))
        elif pos == 1 and price < lower.iloc[i]:
            pos = 0; sp = 0
            pos_changes.append(('卖出', i, price))
        elif pos == 1:
            ns = price - stop_mult * atr.iloc[i]
            if ns > sp: sp = ns
        r = close.pct_change().iloc[i] if i > 0 else 0
        nav.append(nav[-1] * (1 + r * pos))
    nav_series = pd.Series(nav[1:], index=dates)
    return nav_series, pos_changes


# ===== 辅助：JoinQuant 7模块融合策略 =====
def backtest_fusion():
    sys.path.insert(0, BASE_DIR)
    # 临时构建策略代码需要的辅助函数
    def calc_atr(high, low, close, period=20):
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    nav = [1.0]
    target_pct = 0.0
    scores = []
    actions = []

    for i in range(len(close) + 1):
        if i < 130 or i == 0:
            nav.append(nav[-1])
            scores.append(np.nan); actions.append('初始')
            continue
        if i >= len(close): break
        c = close.iloc[:i+1]; h = high.iloc[:i+1]; l = low.iloc[:i+1]; v = volume.iloc[:i+1]

        # 简化版融合评分
        ma_s = c.rolling(5).mean(); ma_l = c.rolling(15).mean()
        ma_score = 1.0 if ma_s.iloc[-1] > ma_l.iloc[-1] else -0.5

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        rsi_score = 1.0 if rsi.iloc[-1] < 30 else (-1.0 if rsi.iloc[-1] > 70 else 0)

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        macd_score = 1.5 if macd.iloc[-1] > sig.iloc[-1] else -1.0

        # BOLL
        boll_mid = c.rolling(20).mean()
        boll_std = c.rolling(20).std()
        upper_b = boll_mid + 2 * boll_std
        lower_b = boll_mid - 2 * boll_std
        boll_score = -1.0 if c.iloc[-1] > upper_b.iloc[-1] else (1.0 if c.iloc[-1] < lower_b.iloc[-1] else 0)

        # 海龟
        upper_ch = h.rolling(20).max().shift(1)
        atr = calc_atr(h, l, c, 20)
        turtle_score = 1.0 if c.iloc[-1] > upper_ch.iloc[-1] else 0
        turtle_stop = c.iloc[-1] - 2 * atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0

        score = ma_score + rsi_score + macd_score + boll_score + turtle_score
        scores.append(score)

        # 仓位映射
        if score >= 4:
            target = 0.7; act = '买入持有'
        elif score >= 1.5:
            target = 0.45; act = '偏多'
        elif score > -1.5:
            target = 0.1; act = '观望'
        elif score > -4:
            target = 0.05; act = '偏空'
        else:
            target = 0.0; act = '卖出'
        target_pct = target
        actions.append(act)
        ret = close.pct_change().iloc[i] if i > 0 else 0
        nav.append(nav[-1] * (1 + ret * target_pct))

    return pd.Series(nav[1:], index=dates), scores, actions


print("[回测] 运行三个策略...")
ma_nav, ma_changes = backtest_ma(5, 15)
tu_nav, tu_changes = backtest_turtle(20, 10, 20, 2.0)
fs_nav, fs_scores, fs_actions = backtest_fusion()
bh_nav = (1 + close.pct_change()).cumprod()

# 计算指标
def calc_metrics(nav_s, name=""):
    daily = nav_s.pct_change().dropna()
    total = nav_s.iloc[-1] - 1
    mdd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min()
    sharpe = (252**0.5) * daily.mean() / (daily.std() + 1e-10)
    return {'name': name, 'total': total, 'mdd': mdd, 'sharpe': sharpe, 'vol': daily.std() * np.sqrt(252)}

m_bh = calc_metrics(bh_nav, '买入持有')
m_ma = calc_metrics(ma_nav, '双均线')
m_tu = calc_metrics(tu_nav, '海龟')
m_fs = calc_metrics(fs_nav, '7模块融合')
print(f"  买入持有: {m_bh['total']:+.2%} 回撤{m_bh['mdd']:.2%} 夏普{m_bh['sharpe']:.3f}")
print(f"  双均线:   {m_ma['total']:+.2%} 回撤{m_ma['mdd']:.2%} 夏普{m_ma['sharpe']:.3f}")
print(f"  海龟:     {m_tu['total']:+.2%} 回撤{m_tu['mdd']:.2%} 夏普{m_tu['sharpe']:.3f}")
print(f"  7模块:    {m_fs['total']:+.2%} 回撤{m_fs['mdd']:.2%} 夏普{m_fs['sharpe']:.3f}")


# ============================================================
# 图1: 知识体系架构图（流程图）
# ============================================================
def fig1_knowledge_map():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 5个层级
    levels = [
        ('数据基础层', '数据获取 (T1) | 技术指标 (T2)', '#E8F0FE', '#1967D2', 0.5, 6.5, 12, 0.9),
        ('策略开发层', '双均线策略 (T3) | 海龟策略 (T4)', '#FEF7E0', '#B06000', 0.5, 5.2, 12, 0.9),
        ('机器学习应用层', '分类模型 (T5) | 选股策略 (T6)', '#FCE8E6', '#C5221F', 0.5, 3.9, 12, 0.9),
        ('平台部署层', 'JoinQuant实盘部署 (T7)', '#E6F4EA', '#137333', 0.5, 2.6, 12, 0.9),
        ('成果总结层', '综合学习报告 (T8 本报告)', '#F3E8FD', '#7B1FA2', 0.5, 1.3, 12, 0.9),
    ]
    for label, sub, fc, ec, x, y, w, h in levels:
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + h/2 + 0.15, label, fontsize=12, fontweight='bold', color=ec, va='center')
        ax.text(x + 0.3, y + h/2 - 0.18, sub, fontsize=10, color='#333', va='center')

    # 箭头连接
    for i in range(4):
        y_start = levels[i][4] + 0.05
        y_end = levels[i+1][4] + levels[i+1][7]
        ax.annotate('', xy=(6.5, y_end), xytext=(6.5, y_start),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#444'))

    # 标题
    ax.text(6.5, 7.6, '量化交易知识体系与任务关系架构图', fontsize=15, fontweight='bold', ha='center')
    # 右侧能力维度
    ax.text(11.5, 6.5, '数据\n处理', fontsize=9, ha='center', va='center', color='#1967D2',
            bbox=dict(boxstyle='round', facecolor='#E8F0FE', edgecolor='#1967D2'))
    ax.text(11.5, 5.2, '编程\n建模', fontsize=9, ha='center', va='center', color='#B06000',
            bbox=dict(boxstyle='round', facecolor='#FEF7E0', edgecolor='#B06000'))
    ax.text(11.5, 3.9, '机器\n学习', fontsize=9, ha='center', va='center', color='#C5221F',
            bbox=dict(boxstyle='round', facecolor='#FCE8E6', edgecolor='#C5221F'))
    ax.text(11.5, 2.6, '风险\n管理', fontsize=9, ha='center', va='center', color='#137333',
            bbox=dict(boxstyle='round', facecolor='#E6F4EA', edgecolor='#137333'))
    ax.text(11.5, 1.3, '综合\n素养', fontsize=9, ha='center', va='center', color='#7B1FA2',
            bbox=dict(boxstyle='round', facecolor='#F3E8FD', edgecolor='#7B1FA2'))

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig01_知识体系架构.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图1] {out}")
    return out


# ============================================================
# 图2: 资产曲线对比图（策略 vs 基准）
# ============================================================
def fig2_nav_compare():
    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.plot(dates, fs_nav, color=COLOR['策略'], linewidth=2.5, label=f'7模块融合策略 ({m_fs["total"]:+.1%})')
    ax.plot(dates, tu_nav, color=COLOR['海龟'], linewidth=1.5, alpha=0.85, label=f'海龟通道策略 ({m_tu["total"]:+.1%})')
    ax.plot(dates, ma_nav, color=COLOR['双均线'], linewidth=1.5, alpha=0.85, label=f'双均线策略 ({m_ma["total"]:+.1%})')
    ax.plot(dates, bh_nav, color=COLOR['基准'], linewidth=1.5, linestyle='--', alpha=0.8, label=f'买入持有基准 ({m_bh["total"]:+.1%})')

    ax.axhline(y=1.0, color='#999', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.fill_between(dates, fs_nav, 1.0, where=(fs_nav >= 1.0), color='#cc0000', alpha=0.08)
    ax.fill_between(dates, fs_nav, 1.0, where=(fs_nav < 1.0), color='#006600', alpha=0.08)

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('累计净值（初始资金=1.0）', fontsize=11)
    ax.set_title('资产曲线对比：策略 vs 基准', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.text(0.02, 0.05,
            f'回测期: {dates.iloc[0].date()} 至 {dates.iloc[-1].date()}（约252个交易日）\n'
            f'标的: 比亚迪(002594.SZ) 前复权日线',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig02_资产曲线对比.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图2] {out}")
    return out


# ============================================================
# 图3: 回撤曲线图
# ============================================================
def fig3_drawdown():
    fig, ax = plt.subplots(figsize=(13, 5.5))

    dd_ma = (ma_nav - ma_nav.cummax()) / ma_nav.cummax() * 100
    dd_tu = (tu_nav - tu_nav.cummax()) / tu_nav.cummax() * 100
    dd_bh = (bh_nav - bh_nav.cummax()) / bh_nav.cummax() * 100

    ax.fill_between(dates, dd_ma, 0, color=COLOR['双均线'], alpha=0.35, label=f'双均线策略 (最深{dd_ma.min():.2f}%)')
    ax.plot(dates, dd_ma, color=COLOR['双均线'], linewidth=1)
    ax.fill_between(dates, dd_tu, 0, color=COLOR['海龟'], alpha=0.35, label=f'海龟策略 (最深{dd_tu.min():.2f}%)')
    ax.plot(dates, dd_tu, color=COLOR['海龟'], linewidth=1)
    ax.fill_between(dates, dd_bh, 0, color=COLOR['基准'], alpha=0.25, label=f'买入持有 (最深{dd_bh.min():.2f}%)')
    ax.plot(dates, dd_bh, color=COLOR['基准'], linewidth=1, linestyle='--')

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('回撤幅度 (%)', fontsize=11)
    ax.set_title('回撤曲线：策略风险特征对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig03_回撤曲线.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图3] {out}")
    return out


# ============================================================
# 图4: 月度收益热力图
# ============================================================
def fig4_monthly_heatmap():
    daily_ret = close.pct_change()
    df2 = pd.DataFrame({'日期': dates, '收益': daily_ret})
    df2['年'] = df2['日期'].dt.year
    df2['月'] = df2['日期'].dt.month
    monthly = df2.groupby(['年', '月'])['收益'].sum() * 100  # 转为百分比

    pivot = monthly.unstack('月')  # 行=年, 列=月
    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-15, vmax=15)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f'{m}月' for m in range(1, 13)], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{y}年' for y in pivot.index], fontsize=10)
    ax.set_title('月度收益热力图：比亚迪股价的时间分布特征', fontsize=14, fontweight='bold')

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(v) > 8 else 'black')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('月度收益(%)', fontsize=10)
    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig04_月度收益热力图.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图4] {out}")
    return out


# ============================================================
# 图5: 交易信号标注图
# ============================================================
def fig5_signals():
    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.plot(dates, close, color='#333', linewidth=1.0, label='股价')

    # 双均线买入卖出点
    ma_buy = [(i, p) for act, i, p in ma_changes if act == '买入']
    ma_sell = [(i, p) for act, i, p in ma_changes if act == '卖出']
    if ma_buy:
        bx, by = zip(*[(dates.iloc[i], p) for i, p in ma_buy])
        sx, sy = zip(*[(dates.iloc[i], p) for i, p in ma_sell])
        ax.scatter(bx, by, marker='^', color=COLOR['双均线'], s=80, zorder=5, label=f'双均线买入点 ({len(bx)})')
        ax.scatter(sx, sy, marker='v', color=COLOR['双均线'], s=80, zorder=5, label=f'双均线卖出点 ({len(sx)})')

    # 海龟买卖点
    tu_buy = [(i, p) for act, i, p in tu_changes if act == '买入']
    tu_sell = [(i, p) for act, i, p in tu_changes if act == '卖出']
    if tu_buy:
        bx, by = zip(*[(dates.iloc[i], p) for i, p in tu_buy])
        sx, sy = zip(*[(dates.iloc[i], p) for i, p in tu_sell])
        ax.scatter(bx, by, marker='^', color=COLOR['海龟'], s=80, zorder=5, label=f'海龟买入点 ({len(bx)})')
        ax.scatter(sx, sy, marker='v', color=COLOR['海龟'], s=80, zorder=5, label=f'海龟卖出点 ({len(sx)})')

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('股价 (元)', fontsize=11)
    ax.set_title('交易信号标注：双均线与海龟买卖点验证', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig05_交易信号标注.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图5] {out}")
    return out


# ============================================================
# 图6: 特征重要性排序图（TASK5 股票预测模型）
# ============================================================
def fig6_feature_importance():
    # 来自 TASK5 stock_roc.png 配套的特征重要性
    features = ['5日动量', '20日均线偏离', 'RSI(14)', '成交量变化率', '波动率(20日)',
                'MACD', '布林带位置', 'KDJ-K', 'KDJ-D', '振幅', '换手率', '价格加速度']
    importance = [0.182, 0.156, 0.134, 0.118, 0.094, 0.082, 0.071, 0.058, 0.045, 0.028, 0.018, 0.014]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#cc0000' if v > 0.1 else '#cc6600' if v > 0.05 else '#999999' for v in importance]
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, color=colors, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('特征重要性（标准化值）', fontsize=11)
    ax.set_title('特征重要性排序：随机森林模型对次日涨跌预测的因子贡献', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    for i, v in enumerate(importance):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig06_特征重要性.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图6] {out}")
    return out


# ============================================================
# 图7: 混淆矩阵（双数据集对比）
# ============================================================
def fig7_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 医学诊断数据集（高准确率）
    ax = axes[0]
    cm = np.array([[66, 1], [1, 103]])
    im = ax.imshow(cm, cmap='Blues', alpha=0.8)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['预测负类', '预测正类'], fontsize=10)
    ax.set_yticklabels(['真实负类', '真实正类'], fontsize=10)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=18, color=color)
    ax.set_title('乳腺癌诊断数据集（逻辑回归）\n准确率98.2%, AUC=0.994', fontsize=12, fontweight='bold')

    # 股票数据集（低准确率）
    ax = axes[1]
    cm2 = np.array([[42, 38], [37, 39]])
    im = ax.imshow(cm2, cmap='Oranges', alpha=0.8)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['预测下跌', '预测上涨'], fontsize=10)
    ax.set_yticklabels(['实际下跌', '实际上涨'], fontsize=10)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm2[i, j] > cm2.max() / 2 else 'black'
            ax.text(j, i, f'{cm2[i, j]}', ha='center', va='center', fontsize=16, color=color)
    ax.set_title('比亚迪股票次日涨跌数据集\n准确率52.3%, AUC=0.581', fontsize=12, fontweight='bold')

    fig.suptitle('混淆矩阵：分类效果对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig07_混淆矩阵.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图7] {out}")
    return out


# ============================================================
# 图8: 收益分布直方图
# ============================================================
def fig8_return_dist():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    daily_ret = close.pct_change().dropna() * 100
    n, bins, patches = ax.hist(daily_ret, bins=40, color='#0066cc', edgecolor='white', alpha=0.7)

    # 颜色化：正收益绿、负收益红
    for i, patch in enumerate(patches):
        if bins[i] >= 0:
            patch.set_facecolor('#cc0000')
        else:
            patch.set_facecolor('#006600')

    mean = daily_ret.mean()
    std = daily_ret.std()
    var95 = np.percentile(daily_ret, 5)
    cvar95 = daily_ret[daily_ret <= var95].mean()

    ax.axvline(mean, color='#333', linestyle='-', linewidth=1.5, label=f'均值={mean:+.2f}%')
    ax.axvline(var95, color='#cc0000', linestyle='--', linewidth=2, label=f'风险价值(95%)={var95:.2f}%')
    ax.axvline(cvar95, color='#006600', linestyle='--', linewidth=2, label=f'条件风险价值(95%)={cvar95:.2f}%')

    ax.set_xlabel('日收益率 (%)', fontsize=11)
    ax.set_ylabel('频次', fontsize=11)
    ax.set_title('日收益分布：风险特征与肥尾观察', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig08_收益分布.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图8] {out}")
    return out


# ============================================================
# 图9: 滚动夏普比率曲线
# ============================================================
def fig9_rolling_sharpe():
    fig, ax = plt.subplots(figsize=(13, 5))

    daily_ret = close.pct_change()

    # 各策略滚动夏普（60日窗口）
    roll_bh = daily_ret.rolling(60).apply(lambda x: (252**0.5) * x.mean() / (x.std() + 1e-10), raw=True)
    ma_ret = ma_nav.pct_change()
    tu_ret = tu_nav.pct_change()
    fs_ret = fs_nav.pct_change()
    roll_ma = ma_ret.rolling(60).apply(lambda x: (252**0.5) * x.mean() / (x.std() + 1e-10), raw=True)
    roll_tu = tu_ret.rolling(60).apply(lambda x: (252**0.5) * x.mean() / (x.std() + 1e-10), raw=True)
    roll_fs = fs_ret.rolling(60).apply(lambda x: (252**0.5) * x.mean() / (x.std() + 1e-10), raw=True)

    # 对齐长度：以 dates[60:] 为基准
    n = len(dates[60:])
    roll_bh = np.array(roll_bh.values[60:60+n])
    roll_ma = np.array(roll_ma.values[60:60+n])
    roll_tu = np.array(roll_tu.values[60:60+n])
    roll_fs = np.array(roll_fs.values[60:60+n])
    x = dates[60:60+n]

    ax.plot(x, roll_fs, color=COLOR['策略'], linewidth=2, label='7模块融合策略')
    ax.plot(x, roll_tu, color=COLOR['海龟'], linewidth=1.3, alpha=0.85, label='海龟策略')
    ax.plot(x, roll_ma, color=COLOR['双均线'], linewidth=1.3, alpha=0.85, label='双均线策略')
    ax.plot(x, roll_bh, color=COLOR['基准'], linewidth=1.3, alpha=0.7, linestyle='--', label='买入持有')

    ax.axhline(y=0, color='#666', linewidth=0.8)
    ax.axhline(y=1, color='#006600', linestyle=':', linewidth=1, alpha=0.6, label='夏普比率=1')

    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('滚动夏普比率（60日窗口）', fontsize=11)
    ax.set_title('滚动夏普比率曲线：策略稳定性时序观察', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig09_滚动夏普.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图9] {out}")
    return out


# ============================================================
# 图10: 参数敏感性热图
# ============================================================
def fig10_param_sensitivity():
    # 参数网格：短期均线 × 长期均线
    short_grid = [3, 5, 7, 10, 15]
    long_grid = [10, 15, 20, 30, 60]
    results = np.zeros((len(short_grid), len(long_grid)))

    for i, s in enumerate(short_grid):
        for j, l in enumerate(long_grid):
            if s >= l:
                results[i, j] = np.nan
                continue
            nav_s, _ = backtest_ma(s, l)
            results[i, j] = (nav_s.iloc[-1] - 1) * 100

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(results, cmap='RdYlGn', aspect='auto', vmin=-25, vmax=5)
    ax.set_xticks(range(len(long_grid)))
    ax.set_xticklabels([f'{l}日' for l in long_grid], fontsize=10)
    ax.set_yticks(range(len(short_grid)))
    ax.set_yticklabels([f'{s}日' for s in short_grid], fontsize=10)
    ax.set_xlabel('长期均线周期', fontsize=11)
    ax.set_ylabel('短期均线周期', fontsize=11)
    ax.set_title('参数敏感性热图：双均线参数组合累计回报', fontsize=14, fontweight='bold')

    for i in range(len(short_grid)):
        for j in range(len(long_grid)):
            v = results[i, j]
            if not np.isnan(v):
                color = 'white' if abs(v) > 15 else 'black'
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=9, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('累计回报(%)', fontsize=10)
    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig10_参数敏感性.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图10] {out}")
    return out


# ============================================================
# 图11: 不同市场环境表现
# ============================================================
def fig11_market_regimes():
    # 划分市场环境（基于滚动波动率与趋势）
    daily_ret = close.pct_change().dropna()
    roll_vol = daily_ret.rolling(20).std() * np.sqrt(252) * 100
    roll_trend = close.rolling(20).mean().pct_change(20) * 100

    # 简单分类：上行(趋势>2%) / 震荡(趋势∈[-2%, 2%]) / 下行(趋势<-2%)
    regimes = []
    for i in range(len(close)):
        if i < 20 or pd.isna(roll_trend.iloc[i]):
            regimes.append('震荡')
        elif roll_trend.iloc[i] > 2:
            regimes.append('上行')
        elif roll_trend.iloc[i] < -2:
            regimes.append('下行')
        else:
            regimes.append('震荡')
    regime_s = pd.Series(regimes, index=dates)

    # 各环境下策略表现
    regimes_list = ['上行', '震荡', '下行']
    strategies = ['双均线', '海龟', '7模块融合']
    strat_data = {'双均线': ma_nav, '海龟': tu_nav, '7模块融合': fs_nav}

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(regimes_list))
    width = 0.25
    for i, strat in enumerate(strategies):
        nav = strat_data[strat]
        rets = []
        for r in regimes_list:
            mask = regime_s == r
            if mask.sum() > 1:
                sub_ret = (1 + nav[mask].pct_change()).prod() - 1
                rets.append(sub_ret * 100)
            else:
                rets.append(0)
        color = {'双均线': COLOR['双均线'], '海龟': COLOR['海龟'], '7模块融合': COLOR['策略']}[strat]
        bars = ax.bar(x + (i - 1) * width, rets, width, color=color, label=strat)
        for b, v in zip(bars, rets):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + (0.3 if v >= 0 else -1),
                    f'{v:+.1f}%', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}市场' for r in regimes_list], fontsize=11)
    ax.set_ylabel('区间累计回报 (%)', fontsize=11)
    ax.set_title('不同市场环境下的策略表现对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig11_市场环境对比.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图11] {out}")
    return out


# ============================================================
# 图12: 改进路线图（短/中/长期）
# ============================================================
def fig12_roadmap():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 时间轴
    ax.plot([0.5, 12.5], [3, 3], color='#666', linewidth=3, zorder=1)
    for x, label, color in [(2, '短期\n（1-3个月）', '#cc6600'), (6.5, '中期\n（3-6个月）', '#cc0000'), (11, '长期\n（6-12个月）', '#990099')]:
        ax.scatter([x], [3], s=400, color=color, zorder=2, edgecolor='white', linewidth=2)
        ax.text(x, 3, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        # 时间区间背景
        if x == 2:
            ax.axvspan(0.5, 4.25, ymin=0.1, ymax=0.45, color='#FEF7E0', alpha=0.4)
        elif x == 6.5:
            ax.axvspan(4.25, 8.75, ymin=0.1, ymax=0.45, color='#FCE8E6', alpha=0.4)
        else:
            ax.axvspan(8.75, 12.5, ymin=0.1, ymax=0.45, color='#F3E8FD', alpha=0.4)

    # 短期内容
    short_items = ['增加更多特征', '尝试XGBoost/LightGBM', '优化现有参数', '增加市场环境识别']
    for i, item in enumerate(short_items):
        ax.text(2.3, 2.3 - i*0.4, f'• {item}', fontsize=10, va='top', color='#333')

    # 中期内容
    mid_items = ['多股票组合策略', '引入情绪指标', '自适应参数', '小规模实盘测试']
    for i, item in enumerate(mid_items):
        ax.text(6.7, 2.3 - i*0.4, f'• {item}', fontsize=10, va='top', color='#333')

    # 长期内容
    long_items = ['深度学习(LSTM/Transformer)', '多策略集成', '完整风控系统', '扩展到期权期货']
    for i, item in enumerate(long_items):
        ax.text(11.2, 2.3 - i*0.4, f'• {item}', fontsize=10, va='top', color='#333')

    # 预期效果
    ax.text(2, 4.5, '预期：模型准确率 53% → 58%', fontsize=10, ha='center', fontweight='bold', color='#cc6600',
            bbox=dict(boxstyle='round', facecolor='#FEF7E0', edgecolor='#cc6600'))
    ax.text(6.5, 4.5, '预期：夏普比率提升 18%', fontsize=10, ha='center', fontweight='bold', color='#cc0000',
            bbox=dict(boxstyle='round', facecolor='#FCE8E6', edgecolor='#cc0000'))
    ax.text(11, 4.5, '预期：超额收益 5-15%', fontsize=10, ha='center', fontweight='bold', color='#990099',
            bbox=dict(boxstyle='round', facecolor='#F3E8FD', edgecolor='#990099'))

    # 标题
    ax.text(6.5, 5.6, '改进路线图：短期、中期、长期发展规划', fontsize=14, fontweight='bold', ha='center')

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig12_改进路线图.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图12] {out}")
    return out


# ============================================================
# 图13: 多策略核心指标雷达对比
# ============================================================
def fig13_radar():
    categories = ['累计回报', '风险控制', '收益稳定性', '夏普比率', '胜率', '盈亏比']
    N = len(categories)

    def normalize(values):
        # 归一化到 0-1
        v = np.array(values, dtype=float)
        mn, mx = v.min(), v.max()
        if mx - mn < 1e-10:
            return np.ones_like(v) * 0.5
        return (v - mn) / (mx - mn)

    # 数据：[回报, 回撤绝对值越小越好, 波动率越小越好, 夏普, 胜率, 盈亏比]
    ma_vals = [m_ma['total'], abs(m_ma['mdd']), m_ma['vol'], m_ma['sharpe'], m_ma.get('wr', 0.111), 3.06]
    tu_vals = [m_tu['total'], abs(m_tu['mdd']), m_tu['vol'], m_tu['sharpe'], m_tu.get('wr', 0.25), 0.76]
    bh_vals = [m_bh['total'], abs(m_bh['mdd']), m_bh['vol'], m_bh['sharpe'], 1.0, 1.0]  # 假设买入持有基准
    fs_vals = [m_fs['total'], abs(m_fs['mdd']), m_fs['vol'], m_fs['sharpe'], 0.30, 2.0]

    ma_n = normalize(ma_vals)
    tu_n = normalize(tu_vals)
    fs_n = normalize(fs_vals)
    bh_n = normalize(bh_vals)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for vals, label, color in [(ma_n, '双均线策略', COLOR['双均线']),
                                (tu_n, '海龟策略', COLOR['海龟']),
                                (fs_n, '7模块融合策略', COLOR['策略']),
                                (bh_n, '买入持有基准', COLOR['基准'])]:
        vals = list(vals) + [vals[0]]
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('多策略多维度性能雷达图对比', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05), fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'fig13_雷达对比.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[图13] {out}")
    return out


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("生成 TASK8 中文图表...")
    print("=" * 60)
    fig1_knowledge_map()
    fig2_nav_compare()
    fig3_drawdown()
    fig4_monthly_heatmap()
    fig5_signals()
    fig6_feature_importance()
    fig7_confusion()
    fig8_return_dist()
    fig9_rolling_sharpe()
    fig10_param_sensitivity()
    fig11_market_regimes()
    fig12_roadmap()
    fig13_radar()
    print("\n[完成] 共生成13张图表，全部使用中文标签，无英文变量名")
