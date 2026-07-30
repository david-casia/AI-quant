# -*- coding: utf-8 -*-
"""
JoinQuant平台策略 - 双均线+海龟通道组合策略
标的：002594.XSHE（比亚迪）
直接复制到JoinQuant平台运行

使用方法：
1. 登录 joinquant.com → 策略 → 新建策略
2. 将本文件全部代码粘贴到编辑器
3. 设置回测周期：2025-07-07 ~ 2026-07-07
4. 设置初始资金：100000元，频率：每天
5. 点击"编译运行"开始回测
"""

import jqdata
import numpy as np
import pandas as pd
from jqdata import *

# ==================== 策略初始化 ====================

def initialize(context):
    """策略初始化 - 只在回测开始时运行一次"""

    # --- 标的设置 ---
    g.security = '002594.XSHE'      # 比亚迪
    set_benchmark('000300.XSHG')     # 沪深300基准

    # --- 回测参数 ---
    set_option('use_real_price', True)       # 使用真实价格(前复权)
    set_option('order_volume_ratio', 0.25)   # 成交量不超过25%
    set_option('avoid_future_data', True)    # 避免未来数据

    # --- 交易费用 ---
    set_order_cost(OrderCost(
        close_tax=0.001,             # 卖出印花税0.1%
        open_commission=0.0003,      # 买入佣金万三
        close_commission=0.0003,     # 卖出佣金万三
        close_today_commission=0,    # 平今仓佣金
        min_commission=5             # 最低佣金5元
    ), type='stock')

    # --- 滑点设置 ---
    set_slippage(PriceRelatedSlippage(0.002))  # 滑点0.2%

    # --- 策略参数（可调整） ---
    # 双均线参数
    g.ma_short = 5          # 短均线周期
    g.ma_long = 20          # 长均线周期

    # 海龟通道参数
    g.dc_entry = 20         # 唐奇安入场通道周期
    g.dc_exit = 10          # 唐奇安出场通道周期
    g.atr_period = 20       # ATR周期
    g.stop_atr_mult = 2.0   # ATR止损倍数

    # 仓位管理
    g.max_position = 0.95   # 最大仓位95%
    g.stop_loss_pct = 0.07  # 固定止损比例7%

    # 全局状态
    g.position_state = 0    # 持仓状态：0空仓 1持仓
    g.entry_price = 0.0     # 入场价格
    g.stop_price = 0.0      # 止损价格
    g.signal_log = []       # 信号记录

    # --- 定时运行 ---
    run_daily(market_open, time='open')        # 开盘时运行
    run_daily(after_market_close, time='close')  # 收盘后运行

    log.info('策略初始化完成 | 标的:%s MA%d/%d DC%d/%d ATR%d S%.1f' % (
        g.security, g.ma_short, g.ma_long,
        g.dc_entry, g.dc_exit, g.atr_period, g.stop_atr_mult))


# ==================== 开盘前准备 ====================

def before_trading_start(context):
    """每日开盘前运行（09:00）"""
    # 记录前一天的总资产
    g.prev_total_value = context.portfolio.total_value
    log.debug('开盘前 %s | 账户总资产: %.2f' % (context.current_dt, g.prev_total_value))


# ==================== 核心交易逻辑 ====================

def market_open(context):
    """每日开盘时运行 - 核心策略执行"""

    security = g.security

    # ===== 1. 获取历史数据 =====
    # 需要的最大窗口 = max(ma_long, dc_entry, atr_period) + 1
    max_window = max(g.ma_long, g.dc_entry, g.atr_period) + 5
    hist = attribute_history(
        security, max_window, '1d',
        ['open', 'high', 'low', 'close', 'volume'],
        df=True
    )

    if hist is None or len(hist) < g.ma_long + 2:
        log.warning('数据不足，跳过')
        return

    close = hist['close'].values
    high = hist['high'].values
    low = hist['low'].values

    # ===== 2. 计算技术指标 =====

    # --- 双均线 ---
    ma_s = np.mean(close[-g.ma_short:])
    ma_l = np.mean(close[-g.ma_long:])
    prev_ma_s = np.mean(close[-g.ma_short-1:-1])
    prev_ma_l = np.mean(close[-g.ma_long-1:-1])

    # 金叉死叉判断
    golden_cross = (prev_ma_s <= prev_ma_l) and (ma_s > ma_l)
    death_cross = (prev_ma_s >= prev_ma_l) and (ma_s < ma_l)

    # --- 唐奇安通道 ---
    # 入场通道：前dc_entry日的最高价
    entry_upper = np.max(high[-g.dc_entry-1:-1])
    exit_lower = np.min(low[-g.dc_exit-1:-1])

    # --- ATR止损 ---
    tr_list = []
    for i in range(1, len(close)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list[-g.atr_period:]) if len(tr_list) >= g.atr_period else 0

    current_price = close[-1]

    # ===== 3. 信号生成（双信号确认） =====

    # 买入信号：金叉 或 突破唐奇安通道上轨
    buy_signal = golden_cross or (current_price > entry_upper)

    # 卖出信号：死叉 或 跌破唐奇安通道下轨 或 触发止损
    sell_signal = death_cross or (current_price < exit_lower)

    # 止损判断
    if g.position_state == 1:
        # ATR移动止损
        new_atr_stop = current_price - g.stop_atr_mult * atr
        if new_atr_stop > g.stop_price:
            g.stop_price = new_atr_stop  # 止损线上移（只上不下）

        # 固定止损
        fixed_stop = g.entry_price * (1 - g.stop_loss_pct)

        stop_triggered = current_price < g.stop_price or current_price < fixed_stop
    else:
        stop_triggered = False

    # ===== 4. 执行交易 =====

    portfolio = context.portfolio
    available_cash = portfolio.available_cash
    total_value = portfolio.total_value

    if g.position_state == 0:
        # --- 空仓状态 ---
        if buy_signal:
            # 双信号确认：金叉 + 突破通道 → 满仓；单信号 → 半仓
            if golden_cross and (current_price > entry_upper):
                target_value = total_value * g.max_position
                signal_type = '强买入(金叉+突破) 满仓'
            elif golden_cross:
                target_value = total_value * 0.6
                signal_type = '买入(金叉) 60%仓'
            else:
                target_value = total_value * 0.5
                signal_type = '买入(突破通道) 50%仓'

            order_target_value(security, target_value)

            # 更新状态
            g.position_state = 1
            g.entry_price = current_price
            g.stop_price = current_price - g.stop_atr_mult * atr
            fixed_stop = g.entry_price * (1 - g.stop_loss_pct)
            g.stop_price = max(g.stop_price, fixed_stop)

            log.info('▲ %s | 价格%.2f MA%.2f/%.2f DC上轨%.2f ATR止损%.2f' % (
                signal_type, current_price, ma_s, ma_l, entry_upper, g.stop_price
            ))
            g.signal_log.append({
                'date': str(context.current_dt.date()),
                'type': 'buy',
                'price': current_price,
                'detail': signal_type
            })

    else:
        # --- 持仓状态 ---
        if sell_signal or stop_triggered:
            order_target_value(security, 0)  # 清仓

            if stop_triggered:
                reason = '止损卖出(价格%.2f < 止损%.2f)' % (current_price, g.stop_price)
            elif death_cross:
                reason = '死叉卖出(MA%.2f < MA%.2f)' % (ma_s, ma_l)
            else:
                reason = '跌破通道卖出(价格%.2f < 下轨%.2f)' % (current_price, exit_lower)

            # 计算单笔收益
            trade_return = (current_price - g.entry_price) / g.entry_price * 100

            log.info('✕ %s | 价格%.2f 收益%.2f%%' % (reason, current_price, trade_return))
            g.signal_log.append({
                'date': str(context.current_dt.date()),
                'type': 'sell',
                'price': current_price,
                'detail': reason,
                'return': trade_return
            })

            # 重置状态
            g.position_state = 0
            g.entry_price = 0.0
            g.stop_price = 0.0


# ==================== 收盘后处理 ====================

def after_market_close(context):
    """每日收盘后运行（15:30）"""
    portfolio = context.portfolio
    positions = portfolio.positions

    log.info('收盘 | 日期:%s 总资产:%.2f 现金:%.2f' % (
        context.current_dt.date(),
        portfolio.total_value,
        portfolio.available_cash
    ))

    # 记录持仓详情
    for stock, position in positions.items():
        log.info('  持仓 %s: 数量%d 成本%.2f 现价%.2f 盈亏%.2f%%' % (
            stock, position.total_amount, position.avg_cost,
            position.price, (position.price / position.avg_cost - 1) * 100
        ))


# ==================== 策略结束总结 ====================

def after_code_changed(context):
    """代码修改后重新初始化（模拟盘用）"""
    log.info('策略代码已更新，重新初始化参数')


# ==================== 风险分析函数 ====================

def analyze_risk(context):
    """
    在策略结束后调用，分析风险暴露
    在JoinQuant的"研究"模块中运行此函数
    """
    import matplotlib.pyplot as plt

    # 读取回测结果（JoinQuant自动保存）
    # 在研究模块中使用 get_backtest_results() 获取

    log.info('=== 风险分析 ===')
    portfolio = context.portfolio
    total_value = portfolio.total_value

    # 当前持仓风险
    for stock, position in portfolio.positions.items():
        invested = position.value
        weight = invested / total_value
        unrealized_pnl = (position.price - position.avg_cost) / position.avg_cost

        log.info('%s 风险:' % stock)
        log.info('  仓位占比: %.1f%%' % (weight * 100))
        log.info('  浮动盈亏: %.2f%%' % (unrealized_pnl * 100))
        log.info('  持仓市值: %.2f元' % invested)

        # VaR估算（简化版）
        hist = attribute_history(stock, 60, '1d', ['close'], df=True)
        if hist is not None and len(hist) > 20:
            daily_returns = hist['close'].pct_change().dropna()
            var_95 = daily_returns.quantile(0.05) * invested
            cvar_95 = daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean() * invested

            log.info('  日VaR(95%%): %.2f元 (%.2f%%)' % (abs(var_95), abs(var_95/invested*100)))
            log.info('  日CVaR(95%%): %.2f元 (%.2f%%)' % (abs(cvar_95), abs(cvar_95/invested*100)))
