# ============================================================
# JoinQuant 全策略融合交易系统
# 标的：比亚迪 002594.XSHE（深交所股票用XSHE后缀）
# 策略：7模块融合（RSI + MACD + BOLL + KDJ + 双均线/海龟 + ML预测 + MLS因子）
# 评分：加权综合评分 -10 ~ +10 → 5档仓位管理
# ============================================================
# 使用方法：
#   1. 登录 joinquant.com → 我的策略 → 新建策略
#   2. 删除模板代码，粘贴本文件全部内容
#   3. 回测设置：标的 002594.XSHE，起始资金 100000，频率 每天
#   4. 点击「编译运行」→ 查看回测结果
#   5. 点击「模拟交易」→ 实盘模拟开始
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# 模块1：技术指标计算引擎（与HTML完全一致）
# ============================================================

def calc_rsi(close, period=14):
    """RSI - Wilder平滑法"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi


def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD: DIF, DEA, Hist"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def calc_boll(close, period=20, std=2.0):
    """布林带: upper, mid, lower"""
    mid = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    """KDJ指标"""
    lowest_low = low.rolling(n, min_periods=1).min()
    highest_high = high.rolling(n, min_periods=1).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    k = rsv.ewm(alpha=1.0/m1, adjust=False).mean()
    d = k.ewm(alpha=1.0/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_atr(high, low, close, period=20):
    """ATR - 平均真实波幅"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
    return atr


def calc_donchian(high, low, entry_p=20, exit_p=10):
    """唐奇安通道"""
    upper = high.rolling(entry_p).max().shift(1)
    lower = low.rolling(exit_p).min().shift(1)
    return upper, lower


# ============================================================
# 模块2：双均线 + 海龟策略持仓状态
# ============================================================

def ma_strategy_signal(close, short_p=5, long_p=15):
    """双均线策略：返回当前应持仓(True/False)"""
    ma_short = close.rolling(short_p).mean()
    ma_long = close.rolling(long_p).mean()
    golden_cross = (ma_short.shift(1) < ma_long.shift(1)) & (ma_short > ma_long)
    death_cross = (ma_short.shift(1) > ma_long.shift(1)) & (ma_short < ma_long)
    # 持仓状态：上次金叉后未死叉
    signal = pd.Series(0, index=close.index)
    holding = False
    for i in range(len(close)):
        if i < long_p:
            signal.iloc[i] = 0
            continue
        if golden_cross.iloc[i]:
            holding = True
        elif death_cross.iloc[i]:
            holding = False
        signal.iloc[i] = 1 if holding else 0
    return signal.iloc[-1] == 1


def turtle_strategy_signal(high, low, close, entry_p=20, exit_p=10, atr_p=20, stop_mult=2.0):
    """海龟策略：返回(应持仓, 止损价)"""
    upper_ch, lower_ch = calc_donchian(high, low, entry_p, exit_p)
    atr = calc_atr(high, low, close, atr_p)

    holding = False
    entry_price = 0.0
    stop_price = 0.0

    for i in range(len(close)):
        if i < entry_p:
            continue
        # 入场：突破通道上轨
        if not holding and close.iloc[i] > upper_ch.iloc[i]:
            holding = True
            entry_price = close.iloc[i]
            stop_price = close.iloc[i] - stop_mult * atr.iloc[i]
        # 止损
        elif holding and close.iloc[i] < stop_price:
            holding = False
        # 离场：跌破通道下轨
        elif holding and close.iloc[i] < lower_ch.iloc[i]:
            holding = False
        # 移动止损（只上不下）
        elif holding:
            new_stop = close.iloc[i] - stop_mult * atr.iloc[i]
            if new_stop > stop_price:
                stop_price = new_stop

    return holding, stop_price


# ============================================================
# 模块3：机器学习预测（纯NumPy逻辑回归）
# ============================================================

class LogisticRegressionNP:
    """纯NumPy实现的逻辑回归（与HTML中JS逻辑一致）"""

    def __init__(self, lr=0.05, epochs=800, l2=0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for epoch in range(self.epochs):
            linear = X @ self.weights + self.bias
            preds = self._sigmoid(linear)
            error = preds - y
            grad_w = (X.T @ error) / n_samples + self.l2 * self.weights
            grad_b = np.mean(error)
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

    def predict_proba(self, X):
        linear = X @ self.weights + self.bias
        return self._sigmoid(linear)


def prepare_ml_features(close, high, low, volume):
    """构建ML特征矩阵（8维）"""
    df = pd.DataFrame(index=close.index)
    df['ret_1'] = close.pct_change(1)
    df['ret_3'] = close.pct_change(3)
    df['rsi_dev'] = (calc_rsi(close, 14) - 50) / 50
    df['macd_hist'] = calc_macd(close)[2] / close * 100
    df['boll_pos'] = (close - calc_boll(close)[1]) / (calc_boll(close)[0] - calc_boll(close)[2] + 1e-10)
    k, d, j = calc_kdj(high, low, close)
    df['kdj_j'] = (j - 50) / 50
    df['vol_ratio'] = volume.pct_change(5).clip(-2, 2)
    df['ma_dev'] = (close - close.rolling(15).mean()) / close.rolling(15).std()
    # 标签：下一日涨(1)/跌(0)
    df['label'] = (close.shift(-1) > close).astype(int)
    return df.dropna()


def ml_predict(close, high, low, volume, window=120):
    """滚动窗口训练ML并预测"""
    if len(close) < window + 10:
        return 0.5, 0.0  # 数据不足，返回中性

    df = prepare_ml_features(close, high, low, volume)
    feature_cols = ['ret_1', 'ret_3', 'rsi_dev', 'macd_hist',
                    'boll_pos', 'kdj_j', 'vol_ratio', 'ma_dev']

    train_df = df.iloc[-window:-1]
    if len(train_df) < 50:
        return 0.5, 0.0

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values

    # 标准化
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std

    model = LogisticRegressionNP(lr=0.05, epochs=800, l2=0.01)
    model.fit(X_train, y_train)

    # 预测最新一行
    latest = df[feature_cols].iloc[-1].values
    latest = (latest - mean) / std
    prob_up = model.predict_proba(latest.reshape(1, -1))[0]
    conf = abs(prob_up - 0.5) * 2  # 置信度 0~1
    return prob_up, conf


# ============================================================
# 模块4：MLS因子打分（简化版因子回归）
# ============================================================

def mls_factor_score(close, high, low, volume, window=60):
    """12因子线性回归打分 → 映射到-1.5~+1.5"""
    if len(close) < window + 10:
        return 0.0

    factors = pd.DataFrame(index=close.index)

    # 因子构建（12个）
    factors['momentum_5'] = close.pct_change(5)
    factors['momentum_20'] = close.pct_change(20)
    factors['reversal_3'] = -close.pct_change(3)
    factors['volatility'] = close.pct_change().rolling(20).std()
    factors['volume_trend'] = volume.pct_change(5)
    factors['rsi_strength'] = (calc_rsi(close, 14) - 50) / 50
    dif, dea, hist = calc_macd(close)
    factors['macd_strength'] = hist / close * 100
    upper, mid, lower = calc_boll(close)
    factors['boll_width'] = (upper - lower) / mid
    k, d, j = calc_kdj(high, low, close)
    factors['kdj_momentum'] = j.diff(3) / 100
    factors['ma_align'] = np.where(close > close.rolling(15).mean(), 1, -1)
    factors['atr_ratio'] = calc_atr(high, low, close) / close

    # 标签：未来5日收益率
    factors['future_ret'] = close.shift(-5) / close - 1
    factors = factors.dropna()

    if len(factors) < window:
        return 0.0

    train = factors.iloc[-window:]
    factor_cols = ['momentum_5', 'momentum_20', 'reversal_3', 'volatility',
                   'volume_trend', 'rsi_strength', 'macd_strength', 'boll_width',
                   'kdj_momentum', 'ma_align', 'atr_ratio']

    X = train[factor_cols].values
    y = train['future_ret'].values

    # 线性回归（正规方程）
    try:
        X_ext = np.column_stack([X, np.ones(len(X))])
        beta = np.linalg.lstsq(X_ext, y, rcond=None)[0]
    except Exception:
        return 0.0

    # 预测最新一行的未来收益
    latest = factors[factor_cols].iloc[-1].values
    latest_ext = np.append(latest, 1.0)
    pred_ret = latest_ext @ beta

    # 映射到-1.5~+1.5
    score = np.clip(pred_ret * 15, -1.5, 1.5)
    return float(score)


# ============================================================
# 模块5：AI综合评分引擎
# ============================================================

def compute_signals(close, high, low, volume):
    """计算7个维度的信号评分"""
    signals = {}

    # --- RSI (权重 1.0) ---
    rsi = calc_rsi(close, 14)
    last_rsi = rsi.iloc[-1]
    if last_rsi < 30:
        signals['rsi'] = (2, f"RSI={last_rsi:.1f} 超卖看涨")
    elif last_rsi < 40:
        signals['rsi'] = (1, f"RSI={last_rsi:.1f} 偏弱有反弹潜力")
    elif last_rsi > 70:
        signals['rsi'] = (-2, f"RSI={last_rsi:.1f} 超买看跌")
    elif last_rsi > 60:
        signals['rsi'] = (-1, f"RSI={last_rsi:.1f} 偏强追高风险")
    else:
        signals['rsi'] = (0, f"RSI={last_rsi:.1f} 中性")

    # --- MACD (权重 1.5) ---
    dif, dea, hist = calc_macd(close)
    last_dif, last_dea, last_hist = dif.iloc[-1], dea.iloc[-1], hist.iloc[-1]
    prev_hist = hist.iloc[-2] if len(hist) >= 2 else 0
    if last_dif > last_dea and prev_hist <= 0 < last_hist:
        signals['macd'] = (2, "MACD金叉看涨")
    elif last_dif > last_dea and last_hist > prev_hist:
        signals['macd'] = (1, f"MACD多头增强(H={last_hist:.4f})")
    elif last_dif < last_dea and prev_hist >= 0 > last_hist:
        signals['macd'] = (-2, "MACD死叉看跌")
    elif last_dif < last_dea and last_hist < prev_hist:
        signals['macd'] = (-1, f"MACD空头增强(H={last_hist:.4f})")
    elif last_dif > last_dea:
        signals['macd'] = (1, "MACD金叉区间")
    else:
        signals['macd'] = (-1, "MACD死叉区间")

    # --- BOLL (权重 1.0) ---
    upper, mid, lower = calc_boll(close)
    last_close = close.iloc[-1]
    last_upper, last_mid, last_lower = upper.iloc[-1], mid.iloc[-1], lower.iloc[-1]
    if last_close > last_upper:
        signals['boll'] = (-1, f"突破上轨{last_upper:.2f} 超买回调")
    elif last_close < last_lower:
        signals['boll'] = (1, f"跌破下轨{last_lower:.2f} 超卖反弹")
    elif last_close > last_mid:
        signals['boll'] = (1, "中轨上方多头偏强")
    else:
        signals['boll'] = (-1, "中轨下方空头偏强")

    # --- KDJ (权重 1.0) ---
    k, d, j = calc_kdj(high, low, close)
    last_k, last_d, last_j = k.iloc[-1], d.iloc[-1], j.iloc[-1]
    prev_k, prev_d = k.iloc[-2], d.iloc[-2]
    if last_j < 0:
        signals['kdj'] = (2, f"J={last_j:.1f} 严重超卖")
    elif last_k < 20 and last_d < 20:
        signals['kdj'] = (1, f"K={last_k:.1f} D={last_d:.1f} 低位超卖")
    elif last_j > 100:
        signals['kdj'] = (-2, f"J={last_j:.1f} 严重超买")
    elif last_k > 80 and last_d > 80:
        signals['kdj'] = (-1, f"K={last_k:.1f} D={last_d:.1f} 高位超买")
    elif prev_k < prev_d and last_k > last_d:
        signals['kdj'] = (1, "KDJ金叉买入信号")
    elif prev_k > prev_d and last_k < last_d:
        signals['kdj'] = (-1, "KDJ死叉卖出信号")
    else:
        signals['kdj'] = (0, f"K={last_k:.1f} D={last_d:.1f} J={last_j:.1f}")

    # --- 策略持仓 (权重 1.5) ---
    ma_holding = ma_strategy_signal(close, 5, 15)
    turtle_holding, turtle_stop = turtle_strategy_signal(
        high, low, close, 20, 10, 20, 2.0)
    strat_holding = ma_holding or turtle_holding  # 任一策略持仓即为持仓
    if strat_holding:
        signals['strat'] = (1, f"策略持仓中(MA:{'持仓' if ma_holding else '空仓'}/Turtle:{'持仓' if turtle_holding else '空仓'})")
    else:
        signals['strat'] = (-1, "双策略均空仓")
    signals['_turtle_stop'] = turtle_stop
    signals['_strat_holding'] = strat_holding

    # --- ML预测 (权重 2.0) ---
    prob_up, conf = ml_predict(close, high, low, volume, window=120)
    ml_score = (prob_up - 0.5) * 4  # -2 ~ +2
    signals['ml'] = (ml_score, f"ML预测涨概率{prob_up*100:.1f}%(置信{conf*100:.1f}%)")
    signals['_ml_prob'] = prob_up

    # --- MLS因子 (权重 1.5) ---
    mls_score = mls_factor_score(close, high, low, volume, window=60)
    direction = "看多" if mls_score > 0 else "看空" if mls_score < 0 else "中性"
    signals['mls'] = (mls_score, f"MLS因子打分{mls_score:+.2f}({direction})")

    return signals


def compute_final_score(signals):
    """加权综合评分 → -10 ~ +10"""
    weights = {
        'rsi': 1.0, 'macd': 1.5, 'boll': 1.0, 'kdj': 1.0,
        'strat': 1.5, 'ml': 2.0, 'mls': 1.5
    }
    total_score = 0.0
    total_weight = 0.0
    for key, w in weights.items():
        total_score += signals[key][0] * w
        total_weight += w * 2  # 最大绝对值

    raw_score = total_score / total_weight * 10 if total_weight > 0 else 0
    final_score = max(-10.0, min(10.0, raw_score))
    return final_score


def score_to_action(score, strat_holding):
    """评分 → (动作, 目标仓位比例)"""
    if score >= 4.0:
        return "买入", 0.70 if not strat_holding else 0.80
    elif score >= 1.5:
        return "偏多持有", 0.40 if not strat_holding else 0.55
    elif score > -1.5:
        return "观望", 0.15 if strat_holding else 0.0
    elif score > -4.0:
        return "偏空减仓", 0.10 if strat_holding else 0.0
    else:
        return "卖出", 0.0


# ============================================================
# JoinQuant 策略主函数
# ============================================================

def initialize(context):
    """初始化"""
    # 标的
    g.stock = '002594.XSHE'

    # 策略参数
    g.ma_short = 5          # 双均线短周期
    g.ma_long = 15          # 双均线长周期
    g.turtle_entry = 20     # 海龟入场通道
    g.turtle_exit = 10      # 海龟离场通道
    g.turtle_atr = 20       # ATR周期
    g.turtle_stop_mult = 2.0  # ATR止损倍数
    g.ml_window = 120       # ML训练窗口
    g.mls_window = 60       # MLS因子窗口
    g.fixed_stop_pct = 0.07  # 固定止损7%
    g.max_position = 0.80    # 最大仓位80%

    # 状态变量
    g.stop_price = 0.0       # 当前止损价
    g.last_score = 0.0       # 上次评分
    g.last_action = ''       # 上次动作
    g.trade_count = 0        # 交易次数
    g.win_count = 0          # 盈利次数
    g.log_enabled = True     # 日志开关

    # 设置滑点、手续费
    set_slippage(FixedSlippage(0.02))
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 设置基准
    set_benchmark('000300.XSHG')

    log.info("=" * 50)
    log.info("全策略融合交易系统初始化完成")
    log.info("标的: %s | 7模块融合评分引擎", g.stock)
    log.info("权重: RSI=1.0 MACD=1.5 BOLL=1.0 KDJ=1.0 策略=1.5 ML=2.0 MLS=1.5")
    log.info("=" * 50)


def before_trading_start(context):
    """盘前：获取数据、计算信号"""
    g.signals = None
    g.final_score = 0.0
    g.action = ''
    g.target_position = 0.0

    # 获取近250日数据（确保指标和ML窗口充足）
    bars = attribute_history(g.stock, 250, '1d',
                             ['open', 'high', 'low', 'close', 'volume'],
                             df=True)
    if bars is None or len(bars) < 60:
        log.warn("数据不足，跳过今日")
        return

    close = bars['close']
    high = bars['high']
    low = bars['low']
    volume = bars['volume']

    # ===== 计算7模块信号 =====
    signals = compute_signals(close, high, low, volume)

    # ===== 综合评分 =====
    final_score = compute_final_score(signals)

    # ===== 动作与仓位 =====
    strat_holding = signals['_strat_holding']
    action, target_pct = score_to_action(final_score, strat_holding)

    # ===== 止损管理 =====
    current_price = close.iloc[-1]
    turtle_stop = signals['_turtle_stop']
    fixed_stop = current_price * (1 - g.fixed_stop_pct)

    # 取海龟止损和固定止损的较高者
    if turtle_stop and turtle_stop > 0:
        g.stop_price = max(turtle_stop, fixed_stop)
    elif target_pct > 0:
        g.stop_price = fixed_stop
    else:
        g.stop_price = 0.0

    # 止损触发：当前价低于止损价 → 强制清仓
    current_position = context.portfolio.positions
    if g.stock in current_position and current_position[g.stock].total_amount > 0:
        avg_cost = current_position[g.stock].avg_cost
        if g.stop_price > 0 and current_price < g.stop_price:
            action = "止损卖出"
            target_pct = 0.0
            log.warn("⚠️ 止损触发！当前价%.2f < 止损价%.2f (成本%.2f，亏损%.1f%%)",
                     current_price, g.stop_price, avg_cost,
                     (current_price - avg_cost) / avg_cost * 100)

    g.signals = signals
    g.final_score = final_score
    g.action = action
    g.target_position = target_pct

    # 日志
    if g.log_enabled:
        log.info("-" * 50)
        log.info("📊 [%s] 综合评分: %+.1f/10 → %s (目标仓位%.0f%%)",
                 str(context.current_dt.date()), final_score, action, target_pct * 100)
        log.info("  RSI:  %s", signals['rsi'][1])
        log.info("  MACD: %s", signals['macd'][1])
        log.info("  BOLL: %s", signals['boll'][1])
        log.info("  KDJ:  %s", signals['kdj'][1])
        log.info("  策略: %s", signals['strat'][1])
        log.info("  ML:   %s", signals['ml'][1])
        log.info("  MLS:  %s", signals['mls'][1])
        if g.stop_price > 0:
            log.info("  止损价: %.2f (%.1f%%)", g.stop_price,
                     (g.stop_price - current_price) / current_price * 100)


def handle_data(context, data):
    """盘中交易（JoinQuant 每分钟调用一次）

    参数: context + data（两个参数都必须接收！data 包含当前 bar 数据）
    """
    # 9:35 之后才下单，避开开盘竞价波动
    if context.current_dt.hour == 9 and context.current_dt.minute < 35:
        return

    # 没有目标仓位或已清仓时不交易
    if g.action == '' or g.target_position <= 0:
        return

    # 当前价格（用attribute_history更稳健，回测+实时都能用）
    try:
        price_data = attribute_history(g.stock, 1, '1m', ['close'], df=False)
        current_price = price_data['close'][-1]
    except Exception:
        # 退而求其次：用日线数据
        price_data = attribute_history(g.stock, 1, '1d', ['close'], df=False)
        current_price = price_data['close'][-1]

    # 计算目标金额
    target_value = context.portfolio.total_value * g.target_position
    target_value = min(target_value, context.portfolio.total_value * g.max_position)

    # 查当前实际仓位
    current_position_value = 0.0
    if g.stock in context.portfolio.positions:
        pos = context.portfolio.positions[g.stock]
        if pos.total_amount > 0:
            current_position_value = pos.total_amount * current_price

    # 仓位变化超过5%才下单（避免微小调整导致频繁交易）
    if abs(target_value - current_position_value) < context.portfolio.total_value * 0.03:
        return

    # 执行下单
    order_target_value(g.stock, target_value)

    # 统计交易次数
    if g.last_action != g.action:
        g.trade_count += 1
        if g.log_enabled:
            log.info("🔄 [%s] 执行%s: 目标金额%.0f元 (当前价%.2f, 实际%.0f→%.0f)",
                     str(context.current_dt.time()), g.action, target_value,
                     current_price, current_position_value, target_value)

    g.last_action = g.action
    g.last_score = g.final_score


def after_market_close(context):
    """盘后：记录日志、统计"""
    if not g.log_enabled:
        return

    # 持仓信息
    positions = context.portfolio.positions
    if g.stock in positions and positions[g.stock].total_amount > 0:
        pos = positions[g.stock]
        pnl = (pos.price - pos.avg_cost) / pos.avg_cost * 100
        log.info("📈 持仓: %d股 @ %.2f (浮盈%.1f%%) | 总资产%.0f",
                 pos.total_amount, pos.price, pnl, context.portfolio.total_value)
    else:
        log.info("💰 空仓 | 总资产%.0f", context.portfolio.total_value)

    # 回报率
    returns = context.portfolio.returns
    log.info("📊 累计回报: %.2f%% | 交易次数: %d", returns * 100, g.trade_count)


# ============================================================
# 风险控制函数（可在研究模块中调用分析）
# ============================================================

def calc_var(returns, confidence=0.95):
    """VaR风险价值"""
    return np.percentile(returns, (1 - confidence) * 100)


def calc_cvar(returns, confidence=0.95):
    """CVaR条件风险价值"""
    var = calc_var(returns, confidence)
    return returns[returns <= var].mean()


def calc_max_drawdown(nav):
    """最大回撤"""
    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    return drawdown.min()


def calc_sharpe(returns, rf=0.03, freq=252):
    """夏普比率"""
    excess = returns - rf / freq
    if excess.std() == 0:
        return 0.0
    return np.sqrt(freq) * excess.mean() / excess.std()
