# ============================================================
# JoinQuant 研究模块：ML训练 + 参数调优 + 回测分析
# 使用方法：
#   1. 登录 joinquant.com → 研究 → 新建 → Notebook (Python 3)
#   2. 粘贴本文件全部代码
#   3. 逐段运行（Shift+Enter）
# ============================================================

# === Cell 1: 导入依赖 ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 聚宽API
from jqdata import *

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# === Cell 2: 技术指标计算（与主策略一致）===
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist

def calc_boll(close, period=20, std=2.0):
    mid = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std

def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    lowest_low = low.rolling(n, min_periods=1).min()
    highest_high = high.rolling(n, min_periods=1).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    k = rsv.ewm(alpha=1.0/m1, adjust=False).mean()
    d = k.ewm(alpha=1.0/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def calc_atr(high, low, close, period=20):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False).mean()


# === Cell 3: 纯NumPy逻辑回归 ===
class LogisticRegressionNP:
    def __init__(self, lr=0.05, epochs=800, l2=0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        for _ in range(self.epochs):
            preds = self._sigmoid(X @ self.weights + self.bias)
            error = preds - y
            self.weights -= self.lr * ((X.T @ error) / n + self.l2 * self.weights)
            self.bias -= self.lr * np.mean(error)

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)


# === Cell 4: 获取数据 ===
stock = '002594.SZ'
start_date = '2025-07-07'
end_date = '2026-07-07'

# 获取前复权日线
df = get_price(stock, start_date=start_date, end_date=end_date,
               frequency='daily', fields=['open', 'high', 'low', 'close', 'volume'],
               fq='qfq', panel=False)

print(f"数据范围: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"总交易日: {len(df)}")
print(f"最新收盘: {df['close'].iloc[-1]:.2f}")
df.head()


# === Cell 5: 构建ML特征与标签 ===
def prepare_features(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    features = pd.DataFrame(index=df.index)
    features['ret_1'] = close.pct_change(1)
    features['ret_3'] = close.pct_change(3)
    features['rsi_dev'] = (calc_rsi(close, 14) - 50) / 50
    features['macd_hist'] = calc_macd(close)[2] / close * 100
    _, mid, _ = calc_boll(close)
    upper, _, lower = calc_boll(close)
    features['boll_pos'] = (close - mid) / (upper - lower + 1e-10)
    k, d, j = calc_kdj(high, low, close)
    features['kdj_j'] = (j - 50) / 50
    features['vol_ratio'] = volume.pct_change(5).clip(-2, 2)
    features['ma_dev'] = (close - close.rolling(15).mean()) / close.rolling(15).std()

    # 标签：下一日涨(1)/跌(0)
    features['label'] = (close.shift(-1) > close).astype(int)
    return features.dropna()

data = prepare_features(df)
feature_cols = ['ret_1', 'ret_3', 'rsi_dev', 'macd_hist',
                'boll_pos', 'kdj_j', 'vol_ratio', 'ma_dev']

print(f"特征样本数: {len(data)}")
print(f"正样本比例: {data['label'].mean():.2%}")
data[feature_cols].describe()


# === Cell 6: ML模型训练与评估 ===
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression as SKLR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 划分训练/测试集（时间序列，不能随机打乱）
split_idx = int(len(data) * 0.7)
train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

X_train = train[feature_cols].values
y_train = train['label'].values
X_test = test[feature_cols].values
y_test = test['label'].values

# 标准化
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8
X_train_s = (X_train - mean) / std
X_test_s = (X_test - mean) / std

results = {}

# 1. 纯NumPy逻辑回归（与策略一致）
model_np = LogisticRegressionNP(lr=0.05, epochs=800, l2=0.01)
model_np.fit(X_train_s, y_train)
pred_np = model_np.predict_proba(X_test_s)
pred_label_np = (pred_np >= 0.5).astype(int)
acc_np = accuracy_score(y_test, pred_label_np)
auc_np = roc_auc_score(y_test, pred_np)
results['NumPy逻辑回归'] = {'acc': acc_np, 'auc': auc_np}

# 2. sklearn逻辑回归
model_lr = SKLR(max_iter=1000)
model_lr.fit(X_train_s, y_train)
pred_lr = model_lr.predict_proba(X_test_s)[:, 1]
results['sklearn逻辑回归'] = {'acc': accuracy_score(y_test, model_lr.predict(X_test_s)), 'auc': roc_auc_score(y_test, pred_lr)}

# 3. 决策树
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)
pred_dt = model_dt.predict_proba(X_test)[:, 1]
results['决策树'] = {'acc': accuracy_score(y_test, model_dt.predict(X_test)), 'auc': roc_auc_score(y_test, pred_dt)}

# 4. 随机森林
model_rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict_proba(X_test)[:, 1]
results['随机森林'] = {'acc': accuracy_score(y_test, model_rf.predict(X_test)), 'auc': roc_auc_score(y_test, pred_rf)}

# 打印结果
print("=" * 50)
print("ML模型对比结果")
print("=" * 50)
print(f"{'模型':<20} {'准确率':>8} {'AUC':>8}")
print("-" * 36)
for name, r in results.items():
    print(f"{name:<20} {r['acc']:>8.2%} {r['auc']:>8.4f}")

# ROC曲线
plt.figure(figsize=(10, 6))
for name, preds in [('NumPy逻辑回归', pred_np), ('sklearn逻辑回归', pred_lr),
                    ('决策树', pred_dt), ('随机森林', pred_rf)]:
    fpr, tpr, _ = roc_curve(y_test, preds)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, preds):.3f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('假正率(FPR)')
plt.ylabel('真正率(TPR)')
plt.title(f'{stock} ML模型ROC曲线对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_roc_comparison.png', dpi=150)
plt.show()


# === Cell 7: 参数网格搜索（MA双均线 + 海龟）===
def backtest_ma_turtle(df, ma_short, ma_long, t_entry, t_exit, atr_p, stop_mult):
    """本地回测双均线+海龟组合"""
    close = df['close']
    high = df['high']
    low = df['low']

    ma_s = close.rolling(ma_short).mean()
    ma_l = close.rolling(ma_long).mean()
    atr = calc_atr(high, low, close, atr_p)
    upper_ch = high.rolling(t_entry).max().shift(1)
    lower_ch = low.rolling(t_exit).min().shift(1)

    position = 0
    stop = 0.0
    nav = [1.0]
    trades = 0
    wins = 0

    for i in range(len(close)):
        if i < max(ma_long, t_entry, atr_p):
            nav.append(nav[-1])
            continue

        price = close.iloc[i]

        # 信号
        golden = ma_s.iloc[i] > ma_l.iloc[i] and ma_s.iloc[i-1] <= ma_l.iloc[i-1]
        turtle_buy = price > upper_ch.iloc[i] and not np.isnan(upper_ch.iloc[i])
        turtle_sell = price < lower_ch.iloc[i] and not np.isnan(lower_ch.iloc[i])
        death = ma_s.iloc[i] < ma_l.iloc[i] and ma_s.iloc[i-1] >= ma_l.iloc[i-1]

        # 入场
        if position == 0 and (golden or turtle_buy):
            position = 1
            stop = price - stop_mult * atr.iloc[i]
            entry_price = price
            trades += 1
        # 止损
        elif position == 1 and price < stop:
            position = 0
            if price > entry_price:
                wins += 1
        # 离场
        elif position == 1 and (turtle_sell or death):
            position = 0
            if price > entry_price:
                wins += 1
        # 移动止损
        elif position == 1:
            new_stop = price - stop_mult * atr.iloc[i]
            if new_stop > stop:
                stop = new_stop

        ret = close.pct_change().iloc[i] if i > 0 else 0
        nav.append(nav[-1] * (1 + ret * position))

    nav = pd.Series(nav[1:], index=close.index)
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    max_dd = ((nav - nav.cummax()) / nav.cummax()).min()
    ann_ret = (1 + total_ret) ** (252 / len(nav)) - 1
    daily_ret = nav.pct_change().dropna()
    sharpe = (252 ** 0.5) * daily_ret.mean() / (daily_ret.std() + 1e-10) if len(daily_ret) > 1 else 0
    win_rate = wins / max(trades, 1)

    return {
        'total_return': total_ret, 'annual_return': ann_ret,
        'max_drawdown': max_dd, 'sharpe': sharpe,
        'trades': trades, 'win_rate': win_rate
    }

# 参数组合
param_grid = []
for ms in [3, 5, 10]:
    for ml_p in [15, 20, 30]:
        for te in [15, 20]:
            for sl in [1.5, 2.0, 3.0]:
                if ms < ml_p:
                    param_grid.append((ms, ml_p, te, 10, 20, sl))

print(f"参数组合数: {len(param_grid)}")

# 执行网格搜索
results_grid = []
for params in param_grid:
    r = backtest_ma_turtle(df, *params)
    r['params'] = params
    results_grid.append(r)

grid_df = pd.DataFrame(results_grid)
grid_df = grid_df.sort_values('sharpe', ascending=False)

print("\n" + "=" * 80)
print("参数网格搜索 Top 10（按夏普比率排序）")
print("=" * 80)
print(f"{'排名':>3} {'MA短':>5} {'MA长':>5} {'入场':>5} {'止损':>5} {'总收益':>8} {'夏普':>7} {'回撤':>8} {'交易':>5} {'胜率':>7}")
for i, row in grid_df.head(10).iterrows():
    p = row['params']
    print(f"{i:>3} {p[0]:>5} {p[1]:>5} {p[2]:>5} {p[5]:>5} "
          f"{row['total_return']:>7.2%} {row['sharpe']:>7.3f} "
          f"{row['max_drawdown']:>7.2%} {row['trades']:>5} {row['win_rate']:>6.1%}")


# === Cell 8: 最优参数可视化 ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top10参数的总收益 vs 夏普
ax = axes[0, 0]
top10 = grid_df.head(10)
ax.barh(range(len(top10)), top10['total_return'] * 100, color='#f85149')
ax.set_yticks(range(len(top10)))
ax.set_yticklabels([f"MA{p[0]}/{p[1]} T{p[2]} S{p[5]}" for p in top10['params']], fontsize=8)
ax.set_xlabel('总收益率(%)')
ax.set_title('Top 10 参数组合总收益')
ax.invert_yaxis()

# 参数敏感性：MA短周期
ax = axes[0, 1]
for ms in [3, 5, 10]:
    sub = grid_df[grid_df['params'].apply(lambda x: x[0] == ms)]
    ax.scatter(sub['sharpe'], sub['total_return'] * 100, label=f'MA短={ms}', alpha=0.6)
ax.set_xlabel('夏普比率')
ax.set_ylabel('总收益率(%)')
ax.set_title('MA短周期参数敏感性')
ax.legend()
ax.grid(True, alpha=0.3)

# 止损倍数 vs 回撤
ax = axes[1, 0]
for sl in [1.5, 2.0, 3.0]:
    sub = grid_df[grid_df['params'].apply(lambda x: x[5] == sl)]
    ax.scatter(sub['sharpe'], sub['max_drawdown'] * 100, label=f'止损={sl}×ATR', alpha=0.6)
ax.set_xlabel('夏普比率')
ax.set_ylabel('最大回撤(%)')
ax.set_title('止损倍数 vs 回撤')
ax.legend()
ax.grid(True, alpha=0.3)

# 最优参数净值曲线
ax = axes[1, 1]
best_params = grid_df.iloc[0]['params']
worst_params = grid_df.iloc[-1]['params']

# 重新回测获取净值
for label, params in [('最优', best_params), ('最差', worst_params)]:
    close = df['close']
    high = df['high']
    low = df['low']
    ma_s = close.rolling(params[0]).mean()
    ma_l = close.rolling(params[1]).mean()
    atr = calc_atr(high, low, close, params[4])
    upper_ch = high.rolling(params[2]).max().shift(1)
    lower_ch = low.rolling(params[3]).min().shift(1)
    pos = 0
    stop = 0.0
    nav = [1.0]
    for i in range(len(close)):
        if i < max(params[1], params[2], params[4]):
            nav.append(nav[-1])
            continue
        price = close.iloc[i]
        golden = ma_s.iloc[i] > ma_l.iloc[i] and ma_s.iloc[i-1] <= ma_l.iloc[i-1]
        tb = price > upper_ch.iloc[i] and not np.isnan(upper_ch.iloc[i])
        ts = price < lower_ch.iloc[i] and not np.isnan(lower_ch.iloc[i])
        death = ma_s.iloc[i] < ma_l.iloc[i] and ma_s.iloc[i-1] >= ma_l.iloc[i-1]
        if pos == 0 and (golden or tb):
            pos = 1
            stop = price - params[5] * atr.iloc[i]
        elif pos == 1 and price < stop:
            pos = 0
        elif pos == 1 and (ts or death):
            pos = 0
        elif pos == 1:
            ns = price - params[5] * atr.iloc[i]
            if ns > stop:
                stop = ns
        ret = close.pct_change().iloc[i] if i > 0 else 0
        nav.append(nav[-1] * (1 + ret * pos))
    ax.plot(nav[1:], label=f'{label} MA{params[0]}/{params[1]}')

# 基准
ax.plot((1 + df['close'].pct_change()).cumprod(), '--', label='买入持有', alpha=0.5)
ax.set_xlabel('交易日')
ax.set_ylabel('净值')
ax.set_title('最优 vs 最差参数净值对比')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f'{stock} 全策略融合 - 参数优化分析', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('param_optimization.png', dpi=150)
plt.show()


# === Cell 9: 综合评分引擎回测 ===
def full_strategy_backtest(df, ml_window=120, mls_window=60):
    """全策略融合回测"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    nav = [1.0]
    position_pct = 0.0
    stop_price = 0.0
    scores = []
    actions = []

    for i in range(len(close)):
        if i < max(ml_window, mls_window, 30):
            nav.append(nav[-1])
            scores.append(0)
            actions.append('观望')
            continue

        # 截取窗口数据
        c = close.iloc[:i+1]
        h = high.iloc[:i+1]
        l = low.iloc[:i+1]
        v = volume.iloc[:i+1]

        # 计算信号
        rsi = calc_rsi(c, 14).iloc[-1]
        dif, dea, hist = calc_macd(c)
        upper, mid, lower = calc_boll(c)
        k, d, j = calc_kdj(h, l, c)

        # 信号评分
        sig_rsi = 2 if rsi < 30 else 1 if rsi < 40 else -2 if rsi > 70 else -1 if rsi > 60 else 0

        last_dif, last_dea = dif.iloc[-1], dea.iloc[-1]
        last_hist, prev_hist = hist.iloc[-1], hist.iloc[-2]
        if last_dif > last_dea and prev_hist <= 0 < last_hist:
            sig_macd = 2
        elif last_dif > last_dea and last_hist > prev_hist:
            sig_macd = 1
        elif last_dif < last_dea and prev_hist >= 0 > last_hist:
            sig_macd = -2
        elif last_dif < last_dea and last_hist < prev_hist:
            sig_macd = -1
        else:
            sig_macd = 1 if last_dif > last_dea else -1

        price = c.iloc[-1]
        sig_boll = -1 if price > upper.iloc[-1] else 1 if price < lower.iloc[-1] else 1 if price > mid.iloc[-1] else -1

        last_j = j.iloc[-1]
        last_k, last_d = k.iloc[-1], d.iloc[-1]
        prev_k, prev_d = k.iloc[-2], d.iloc[-2]
        sig_kdj = 2 if last_j < 0 else 1 if (last_k < 20 and last_d < 20) else -2 if last_j > 100 else -1 if (last_k > 80 and last_d > 80) else 1 if (prev_k < prev_d and last_k > last_d) else -1 if (prev_k > prev_d and last_k < last_d) else 0

        # 双均线持仓
        ma_s = c.rolling(5).mean()
        ma_l_p = c.rolling(15).mean()
        ma_hold = ma_s.iloc[-1] > ma_l_p.iloc[-1]
        sig_strat = 1 if ma_hold else -1

        # ML预测（简化：用滚动窗口）
        try:
            feat_df = pd.DataFrame(index=c.index)
            feat_df['ret_1'] = c.pct_change(1)
            feat_df['ret_3'] = c.pct_change(3)
            feat_df['rsi_dev'] = (calc_rsi(c, 14) - 50) / 50
            feat_df['macd_hist'] = calc_macd(c)[2] / c * 100
            feat_df['boll_pos'] = (c - mid) / (upper - lower + 1e-10)
            feat_df['kdj_j'] = (j - 50) / 50
            feat_df['vol_ratio'] = v.pct_change(5).clip(-2, 2)
            feat_df['ma_dev'] = (c - c.rolling(15).mean()) / c.rolling(15).std()
            feat_df['label'] = (c.shift(-1) > c).astype(int)
            feat_df = feat_df.dropna()

            train = feat_df.iloc[-ml_window:-1]
            if len(train) > 30:
                fcols = ['ret_1', 'ret_3', 'rsi_dev', 'macd_hist', 'boll_pos', 'kdj_j', 'vol_ratio', 'ma_dev']
                X_tr = train[fcols].values
                y_tr = train['label'].values
                m, s = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-8
                X_tr = (X_tr - m) / s
                model = LogisticRegressionNP(lr=0.05, epochs=500, l2=0.01)
                model.fit(X_tr, y_tr)
                latest = feat_df[fcols].iloc[-1].values
                latest = (latest - m) / s
                prob_up = model.predict_proba(latest.reshape(1, -1))[0]
                sig_ml = (prob_up - 0.5) * 4
            else:
                sig_ml = 0
        except:
            sig_ml = 0

        # 综合评分
        weights = {'rsi': 1.0, 'macd': 1.5, 'boll': 1.0, 'kdj': 1.0, 'strat': 1.5, 'ml': 2.0}
        scores_dict = {'rsi': sig_rsi, 'macd': sig_macd, 'boll': sig_boll, 'kdj': sig_kdj, 'strat': sig_strat, 'ml': sig_ml}
        total = sum(scores_dict[k] * weights[k] for k in weights)
        total_w = sum(weights[k] * 2 for k in weights)
        final = max(-10, min(10, total / total_w * 10))
        scores.append(final)

        # 仓位管理
        if final >= 4:
            action, target = '买入', 0.70
        elif final >= 1.5:
            action, target = '偏多持有', 0.40
        elif final > -1.5:
            action, target = '观望', 0.15 if ma_hold else 0.0
        elif final > -4:
            action, target = '偏空减仓', 0.10
        else:
            action, target = '卖出', 0.0
        actions.append(action)

        # 止损
        if target > 0:
            new_stop = price * 0.93
            if new_stop > stop_price:
                stop_price = new_stop
        else:
            stop_price = 0.0

        if stop_price > 0 and price < stop_price:
            target = 0.0
            stop_price = 0.0

        position_pct = target
        ret = c.pct_change().iloc[i] if i > 0 else 0
        nav.append(nav[-1] * (1 + ret * position_pct))

    nav = pd.Series(nav[1:], index=close.index)
    scores = pd.Series(scores, index=close.index)
    return nav, scores, actions

# 执行
nav, scores, actions = full_strategy_backtest(df)

# 绩效指标
total_ret = nav.iloc[-1] - 1
bh_ret = df['close'].iloc[-1] / df['close'].iloc[0] - 1
max_dd = ((nav - nav.cummax()) / nav.cummax()).min()
daily_ret = nav.pct_change().dropna()
sharpe = (252 ** 0.5) * daily_ret.mean() / (daily_ret.std() + 1e-10)

# VaR/CVaR
var_95 = np.percentile(daily_ret, 5)
cvar_95 = daily_ret[daily_ret <= var_95].mean()

print("=" * 60)
print("全策略融合回测绩效报告")
print("=" * 60)
print(f"回测区间:     {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"交易天数:     {len(df)}")
print(f"策略总收益:   {total_ret:>10.2%}")
print(f"买入持有:     {bh_ret:>10.2%}")
print(f"超额收益:     {total_ret - bh_ret:>10.2%}")
print(f"最大回撤:     {max_dd:>10.2%}")
print(f"夏普比率:     {sharpe:>10.4f}")
print(f"VaR(95%):     {var_95:>10.4f}")
print(f"CVaR(95%):    {cvar_95:>10.4f}")

# 动作分布
from collections import Counter
action_dist = Counter(actions)
print(f"\n信号分布:")
for a, cnt in sorted(action_dist.items(), key=lambda x: -x[1]):
    print(f"  {a}: {cnt}天 ({cnt/len(actions):.1%})")


# === Cell 10: 可视化 ===
fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

# 净值曲线
ax = axes[0]
ax.plot(nav.index, nav, label='全策略融合', linewidth=2, color='#f85149')
bh_nav = (1 + df['close'].pct_change()).cumprod()
ax.plot(bh_nav.index, bh_nav, label='买入持有', alpha=0.6, color='#58a6ff')
ax.fill_between(nav.index, nav, nav.cummax(), alpha=0.2, color='#3fb950', label='回撤区间')
ax.set_ylabel('净值')
ax.set_title(f'{stock} 全策略融合 vs 买入持有', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 评分时间序列
ax = axes[1]
colors = ['#f85149' if s > 0 else '#3fb950' for s in scores]
ax.bar(scores.index, scores, color=colors, alpha=0.6, width=1)
ax.axhline(y=4, color='#f85149', linestyle='--', alpha=0.5, label='买入线(+4)')
ax.axhline(y=-4, color='#3fb950', linestyle='--', alpha=0.5, label='卖出线(-4)')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.set_ylabel('综合评分')
ax.set_title('AI综合评分时间序列 (-10 ~ +10)')
ax.legend()
ax.grid(True, alpha=0.3)

# 回撤
ax = axes[2]
drawdown = (nav - nav.cummax()) / nav.cummax()
ax.fill_between(drawdown.index, drawdown * 100, 0, color='#3fb950', alpha=0.4)
ax.set_ylabel('回撤(%)')
ax.set_title('最大回撤')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('full_strategy_backtest.png', dpi=150)
plt.show()


# === Cell 11: 滚动风险分析 ===
window = 60
rolling_sharpe = daily_ret.rolling(window).apply(
    lambda x: (252 ** 0.5) * x.mean() / (x.std() + 1e-10))
rolling_vol = daily_ret.rolling(window).std() * np.sqrt(252)
rolling_dd = nav.rolling(window).apply(lambda x: ((x - x.cummax()) / x.cummax()).min())

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

ax = axes[0]
ax.plot(rolling_sharpe, color='#f85149', linewidth=1.5)
ax.axhline(y=0, color='gray', alpha=0.3)
ax.axhline(y=1, color='#3fb950', linestyle='--', alpha=0.5, label='夏普=1')
ax.set_title(f'滚动夏普比率 (窗口={window}天)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(rolling_vol * 100, color='#d29922', linewidth=1.5)
ax.set_title('滚动年化波动率')
ax.set_ylabel('%')
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.fill_between(rolling_dd.index, rolling_dd * 100, 0, color='#3fb950', alpha=0.4)
ax.set_title('滚动最大回撤')
ax.set_ylabel('%')
ax.grid(True, alpha=0.3)

plt.suptitle(f'{stock} 滚动风险分析', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rolling_risk_analysis.png', dpi=150)
plt.show()

print("\n✅ 研究模块分析完成！")
print(f"最优参数推荐: MA短={best_params[0]}, MA长={best_params[1]}, 海龟入场={best_params[2]}, 止损={best_params[5]}×ATR")
print(f"全策略融合夏普: {sharpe:.4f}, 总收益: {total_ret:.2%}, 回撤: {max_dd:.2%}")
print("\n下一步：将最优参数填入 jq_full_strategy.py 的 initialize() 函数中")
