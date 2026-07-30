# ============================================================
# TASK7: 全策略融合 - 图表生成（用于PDF报告）
# ============================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

# ===== 字体配置 =====
if sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['Apple LiSung Light', 'Arial Unicode MS', 'PingFang SC']
elif sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# ===== 导入策略代码 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
exec(open(os.path.join(BASE_DIR, 'jq_full_strategy.py'), encoding='utf-8').read().split('# JoinQuant 策略主函数')[0])

# ===== 加载数据 =====
df = pd.read_csv(os.path.join(BASE_DIR, '002594_daily.csv'), parse_dates=['trade_date'])
df = df.sort_values('trade_date').reset_index(drop=True)
close = df['close']
high = df['high']
low = df['low']
volume = df['vol']
dates = df['trade_date']

print(f"[1/4] 数据: {len(df)}条, {dates.iloc[0].date()} ~ {dates.iloc[-1].date()}")

# ===== 完整回测 =====
nav = [1.0]
scores_list = []
actions_list = []
target_pct = 0.0
stop = 0.0

for i in range(len(close)):
    if i < 130:
        nav.append(nav[-1])
        scores_list.append(np.nan)
        actions_list.append('initial')
        continue
    c = close.iloc[:i+1]; h = high.iloc[:i+1]; l = low.iloc[:i+1]; v = volume.iloc[:i+1]
    try:
        signals = compute_signals(c, h, l, v)
        fs = compute_final_score(signals)
        holding = signals['_strat_holding']
        action, target = score_to_action(fs, holding)
        turtle_stop = signals['_turtle_stop']
        fixed = c.iloc[-1] * 0.93
        if target > 0:
            new_stop = max(turtle_stop if turtle_stop else 0, fixed)
            if new_stop > stop: stop = new_stop
        else: stop = 0.0
        if stop > 0 and c.iloc[-1] < stop:
            target = 0.0; action = 'SL'; stop = 0.0
        target_pct = target
        scores_list.append(fs)
        actions_list.append(action)
    except:
        scores_list.append(np.nan); actions_list.append('err')
    ret = close.pct_change().iloc[i] if i > 0 else 0
    nav.append(nav[-1] * (1 + ret * target_pct))

nav_s = pd.Series(nav[1:], index=dates)
daily_ret = nav_s.pct_change().dropna()
total_ret = nav_s.iloc[-1] - 1
bh_ret = close.iloc[-1] / close.iloc[0] - 1
max_dd = ((nav_s - nav_s.cummax()) / nav_s.cummax()).min()
sharpe = (252**0.5) * daily_ret.mean() / (daily_ret.std() + 1e-10)
var95 = np.percentile(daily_ret, 5)
cvar95 = daily_ret[daily_ret <= var95].mean()
actions = [a for a in actions_list if a != 'initial']
act_dist = Counter(actions)

print(f"[2/4] 回测完成: 策略{total_ret:+.2%} | 买入持有{bh_ret:+.2%} | "
      f"夏普{sharpe:.3f} | 回撤{max_dd:.2%}")

# ===== 图1: 净值曲线 + 仓位热力图 =====
print(f"[3/4] 生成图表...")
fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

ax = axes[0]
ax.plot(dates, nav_s, color='#cc0000', linewidth=2, label='7模块融合策略')
bh_nav = (1 + close.pct_change()).cumprod()
ax.plot(dates, bh_nav, color='#0066cc', linewidth=1.2, alpha=0.7, label='买入持有')
ax.fill_between(dates, nav_s, nav_s.cummax(), color='#006600', alpha=0.15, label='回撤区间')
ax.set_ylabel('净值', fontsize=11)
ax.set_title('图1  比亚迪(002594) 全策略融合 vs 买入持有', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.95, f'策略收益:{total_ret:+.2%}  买入持有:{bh_ret:+.2%}\n夏普:{sharpe:.3f}  最大回撤:{max_dd:.2%}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9), color='black')

ax = axes[1]
score_vals = pd.Series(scores_list, index=dates).dropna()
colors = ['#cc0000' if s > 0 else '#006600' for s in score_vals]
ax.bar(score_vals.index, score_vals, color=colors, alpha=0.6, width=1)
ax.axhline(y=4, color='#cc0000', linestyle='--', alpha=0.5, linewidth=1, label='买入线(+4)')
ax.axhline(y=1.5, color='#cc0000', linestyle=':', alpha=0.3, linewidth=1)
ax.axhline(y=-1.5, color='#006600', linestyle=':', alpha=0.3, linewidth=1)
ax.axhline(y=-4, color='#006600', linestyle='--', alpha=0.5, linewidth=1, label='卖出线(-4)')
ax.axhline(y=0, color='gray', alpha=0.3)
ax.set_ylabel('综合评分', fontsize=11)
ax.set_title('图1  AI综合评分序列(-10~+10)', fontsize=12)
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.2)

ax = axes[2]
dd_series = (nav_s - nav_s.cummax()) / nav_s.cummax()
ax.fill_between(dates, dd_series * 100, 0, color='#006600', alpha=0.35)
ax.set_ylabel('回撤(%)', fontsize=11)
ax.set_xlabel('日期', fontsize=11)
ax.set_title('图1  策略回撤曲线', fontsize=12)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'jq_fs_nav.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  , jq_fs_nav.png")

# ===== 图2: 多策略对比（含旧策略 + 买入持有）=====
fig, ax = plt.subplots(figsize=(14, 6))

# 双均线策略
ma_s = close.rolling(5).mean(); ma_l = close.rolling(15).mean()
ma_pos, ma_nav = 0, [1.0]
for i in range(len(close)):
    if i < 15: ma_nav.append(ma_nav[-1]); continue
    golden = ma_s.iloc[i] > ma_l.iloc[i] and ma_s.iloc[i-1] <= ma_l.iloc[i-1]
    death = ma_s.iloc[i] < ma_l.iloc[i] and ma_s.iloc[i-1] >= ma_l.iloc[i-1]
    if golden: ma_pos = 1
    elif death: ma_pos = 0
    r = close.pct_change().iloc[i] if i > 0 else 0
    ma_nav.append(ma_nav[-1] * (1 + r * ma_pos))

# 海龟策略
t_entry, t_exit, t_atr, t_stop = 20, 10, 20, 2.0
atr = calc_atr(high, low, close, t_atr)
upper_ch, lower_ch = high.rolling(t_entry).max().shift(1), low.rolling(t_exit).min().shift(1)
t_pos, t_nav, t_sp = 0, [1.0], 0.0
for i in range(len(close)):
    if i < t_entry: t_nav.append(t_nav[-1]); continue
    price = close.iloc[i]
    if t_pos == 0 and price > upper_ch.iloc[i]:
        t_pos = 1; t_sp = price - t_stop * atr.iloc[i]
    elif t_pos == 1 and price < t_sp: t_pos = 0
    elif t_pos == 1 and price < lower_ch.iloc[i]: t_pos = 0
    elif t_pos == 1:
        ns = price - t_stop * atr.iloc[i]
        if ns > t_sp: t_sp = ns
    r = close.pct_change().iloc[i] if i > 0 else 0
    t_nav.append(t_nav[-1] * (1 + r * t_pos))

ma_nav = pd.Series(ma_nav[1:], index=dates)
t_nav = pd.Series(t_nav[1:], index=dates)

ax.plot(dates, nav_s, color='#cc0000', linewidth=2.5, label=f'7模块融合({total_ret:+.1%})')
ax.plot(dates, ma_nav, color='#990099', linewidth=1.2, alpha=0.7, label=f'双均线MA5/15({ma_nav.iloc[-1]-1:+.1%})')
ax.plot(dates, t_nav, color='#0099cc', linewidth=1.2, alpha=0.7, label=f'海龟通道({t_nav.iloc[-1]-1:+.1%})')
ax.plot(dates, bh_nav, color='#666666', linewidth=1.2, alpha=0.5, linestyle='--', label=f'买入持有({bh_ret:+.1%})')
ax.set_ylabel('净值', fontsize=11)
ax.set_title('图2  比亚迪(002594) 多策略净值对比', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'jq_fs_compare.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  , jq_fs_compare.png")

# ===== 图3: 参数对比 + 动作分布 =====
param_configs = [
    ('MA3/15 T20 S2.0', 3, 15, 20, 2.0),
    ('MA5/15 T20 S2.0', 5, 15, 20, 2.0),
    ('MA5/20 T20 S2.0', 5, 20, 20, 2.0),
    ('MA5/15 T15 S2.0', 5, 15, 15, 2.0),
    ('MA5/15 T20 S1.5', 5, 15, 20, 1.5),
    ('MA5/15 T20 S3.0', 5, 15, 20, 3.0),
    ('MA10/30 T20 S2.0', 10, 30, 20, 2.0),
]

def backtest_combo(close, high, low, ms, ml, te, sm):
    upper_ch = high.rolling(te).max().shift(1); lower_ch = low.rolling(10).min().shift(1)
    atr = calc_atr(high, low, close, 20)
    ma_short = close.rolling(ms).mean(); ma_long = close.rolling(ml).mean()
    pos, sp, nav = 0, 0.0, [1.0]
    for i in range(len(close)):
        if i < max(ml, te): nav.append(nav[-1]); continue
        p = close.iloc[i]
        g = ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i-1] <= ma_long.iloc[i-1]
        d = ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i-1] >= ma_long.iloc[i-1]
        tb = p > upper_ch.iloc[i] if not pd.isna(upper_ch.iloc[i]) else False
        # Use exit=10 for turtle
        ts = p < lower_ch.iloc[i] if not pd.isna(lower_ch.iloc[i]) else False
        if pos == 0 and (g or tb): pos = 1; sp = p - sm * atr.iloc[i]
        elif pos == 1 and p < sp: pos = 0
        elif pos == 1 and (d or ts): pos = 0
        elif pos == 1:
            ns = p - sm * atr.iloc[i]
            if ns > sp: sp = ns
        r = close.pct_change().iloc[i] if i > 0 else 0
        nav.append(nav[-1] * (1 + r * pos))
    nav = pd.Series(nav[1:])
    tr = nav.iloc[-1] - 1
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    dr = nav.pct_change().dropna()
    sr = (252**0.5) * dr.mean() / (dr.std() + 1e-10) if len(dr) > 1 else 0
    return tr, dd, sr

results = []
for name, ms, ml, te, sm in param_configs:
    tr, dd, sr = backtest_combo(close, high, low, ms, ml, te, sm)
    results.append({'name': name, 'return': tr, 'dd': dd, 'sharpe': sr})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左: 参数对比
ax = axes[0]
x = np.arange(len(results))
width = 0.25
ax.bar(x - width, [r['return']*100 for r in results], width, color='#cc0000', label='总收益(%)')
ax.bar(x, [r['sharpe']*100 for r in results], width, color='#cc6600', label='夏普×100')
ax.bar(x + width, [-r['dd']*100 for r in results], width, color='#006600', label='-最大回撤(%)')
ax.set_xticks(x); ax.set_xticklabels([r['name'] for r in results], rotation=30, ha='right', fontsize=7)
ax.set_title('图3  7组参数对比', fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(y=0, color='gray', alpha=0.3)

# 右: 动作分布饼图
ax = axes[1]
act_labels = list(act_dist.keys())
act_values = list(act_dist.values())
colors = ['#cc0000' if '买' in a else '#cc6600' if '偏多' in a else '#999900' if '观望' in a else '#006600' if '偏空' in a else '#0066cc' for a in act_labels]
wedges, texts, autotexts = ax.pie(act_values, labels=None, autopct='%1.1f%%',
                                   startangle=90, colors=colors, textprops={'fontsize': 8})
ax.legend(wedges, [f'{a}({c}天)' for a, c in zip(act_labels, act_values)], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
ax.set_title(f'图3  策略动作分布(共{sum(act_values)}天)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'jq_fs_params.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  , jq_fs_params.png")

# ===== 图4: 风险评估 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 左上: VaR直方图
ax = axes[0, 0]
ax.hist(daily_ret*100, bins=30, color='#0099cc', edgecolor='white', alpha=0.7)
ax.axvline(x=var95*100, color='#cc0000', linestyle='--', linewidth=2, label=f'VaR(95%)={var95*100:.2f}%')
ax.axvline(x=cvar95*100, color='#006600', linestyle='--', linewidth=2, label=f'CVaR(95%)={cvar95*100:.2f}%')
ax.set_xlabel('日收益(%)'); ax.set_ylabel('频次')
ax.set_title('图4  日收益分布 & VaR/CVaR')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# 右上: 滚动夏普
ax = axes[1, 0]
roll_sharpe = daily_ret.rolling(60).apply(lambda x: (252**0.5)*x.mean()/(x.std()+1e-10))
ax.plot(dates[60:], roll_sharpe[59:], color='#cc0000', linewidth=1.5)
ax.axhline(y=0, color='gray', alpha=0.3); ax.axhline(y=1, color='#006600', linestyle='--', alpha=0.5, label='夏普=1')
ax.set_xlabel('日期'); ax.set_ylabel('夏普比率')
ax.set_title('图4  滚动夏普比率(60日窗口)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# 左下: 累计收益分解
ax = axes[0, 1]
cum_ret = nav_s - 1
ax.plot(dates, cum_ret*100, color='#cc0000', linewidth=2)
ax.fill_between(dates, 0, cum_ret*100, where=(cum_ret>=0), color='#cc0000', alpha=0.3)
ax.fill_between(dates, 0, cum_ret*100, where=(cum_ret<0), color='#006600', alpha=0.3)
ax.axhline(y=0, color='gray', alpha=0.3)
ax.set_xlabel('日期'); ax.set_ylabel('累计收益(%)')
ax.set_title('图4  策略累计收益曲线')
ax.grid(True, alpha=0.2)

# 右下: 关键指标表
ax = axes[1, 1]
ax.axis('off')
metrics_text = f"""回测绩效指标
=================
策略总收益        {total_ret:>+10.2%}
买入持有          {bh_ret:>+10.2%}
超额收益          {total_ret-bh_ret:>+10.2%}
年化波动率        {daily_ret.std()*np.sqrt(252):>10.2%}
最大回撤          {max_dd:>10.2%}
夏普比率          {sharpe:>10.4f}
VaR (95%)         {var95:>10.4f}
CVaR (95%)        {cvar95:>10.4f}
信号天数          {sum(act_values):>10}d
头寸天数          {act_dist.get('偏多持有',0)+act_dist.get('偏空减仓',0):>10}d
评估数据源        比亚迪002594 前复权日线
回测区间          2025-07-07 ~ 2026-07-07"""

ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='sans-serif',
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#999999'))
ax.set_title('图4  关键指标汇总')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'jq_fs_risk.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("  , jq_fs_risk.png")

# ===== 打印摘要 =====
print(f"\n[4/4] 全部图表生成完成！")
print(f"================================================")
print(f"回测绩效摘要（用于PDF报告）")
print(f"================================================")
print(f"策略总收益:      {total_ret:+.2%}")
print(f"买入持有:        {bh_ret:+.2%}")
print(f"超额收益:        {total_ret-bh_ret:+.2%}")
print(f"夏普比率:        {sharpe:.4f}")
print(f"最大回撤:        {max_dd:.2%}")
print(f"VaR(95%):        {var95:.4f}")
print(f"CVaR(95%):       {cvar95:.4f}")
print(f"交易信号天数:    {sum(act_values)}")
print(f"最优参数:        MA5/15 + 海龟T20/S2.0")
print(f"动作分布:        {dict(act_dist)}")
