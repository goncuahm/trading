import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import itertools

st.set_page_config(layout="wide")

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.title("Strategy Settings")

TICKER = st.sidebar.text_input("Ticker", value="ISDMR.IS")

START = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2024-01-01")
)

END = datetime.today().strftime('%Y-%m-%d')

RISK_FREE_RATE_ANNUAL = st.sidebar.number_input(
    "Annual Risk-Free Rate (e.g. 0.30 = 30%)",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.01
)

TRANSACTION_COST = 0.002

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Long / Short Strategy Parameters")

rsi_days_ls = st.sidebar.slider("RSI Period (Long/Short)", 5, 30, 11)
rsi_entry_long = st.sidebar.slider("Long Entry: RSI below", 10, 50, 33)
rsi_entry_short = st.sidebar.slider("Short Entry: RSI above", 50, 90, 67)
max_holding_days = st.sidebar.slider("Max Holding Days", 5, 60, 22)

st.sidebar.markdown("---")
st.sidebar.subheader("🧮 RSI + Cubic Forecast Strategy")
rsi_days_cf = st.sidebar.slider(
    "RSI Period (also the cubic-fit window)", 5, 30, 14,
    help="The RSI is computed over this many days, and the cubic polynomial is fitted "
         "to the last this-many RSI values.")
cf_oversold = st.sidebar.slider("Oversold: RSI below", 10, 45, 30)
cf_overbought = st.sidebar.slider("Overbought: RSI above", 55, 90, 70)
cf_horizon = st.sidebar.slider(
    "Forecast horizon (days ahead)", 1, 5, 1,
    help="The cubic fit is extrapolated this many days ahead. 'Turning up' means the "
         "forecast RSI is above today's RSI; 'turning down' means it is below.")
cf_execution = st.sidebar.selectbox(
    "Execute trades at",
    ["Next day's open (no look-ahead)", "Same day's close"],
    index=0,
    help="The signal uses today's close. Trading at the next open avoids look-ahead; "
         "trading at today's close assumes you can act in the closing auction.")

# =====================================================
# DOWNLOAD FUNCTION
# =====================================================

@st.cache_data
def download_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close']].dropna()
    return data

# =====================================================
# RSI FUNCTION
# =====================================================

def compute_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =====================================================
# RSI PIVOT BACKTEST (original strategy)
# =====================================================

def backtest_strategy(df, rsi_len, pivot_lb, bottom_th, peak_th, risk_free_rate):
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], rsi_len)
    df['Bottom'] = False
    df['Peak'] = False

    for i in range(pivot_lb, len(df)):
        window_past = df['RSI'].iloc[i - pivot_lb: i + 1]
        current = df['RSI'].iloc[i]
        if current == window_past.min() and current < bottom_th:
            df.iloc[i, df.columns.get_loc('Bottom')] = True
        if current == window_past.max() and current > peak_th:
            df.iloc[i, df.columns.get_loc('Peak')] = True

    initial_cash = 10000
    cash = initial_cash
    shares = 0.0
    position_size = 2000
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    portfolio = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        cash *= (1 + daily_rf)
        if df['Bottom'].iloc[i] and cash >= position_size:
            shares_to_buy = position_size / price
            cost = shares_to_buy * price * (1 + TRANSACTION_COST)
            if cash >= cost:
                shares += shares_to_buy
                cash -= cost
        if df['Peak'].iloc[i] and shares > 0:
            shares_to_sell = min(position_size / price, shares)
            proceeds = shares_to_sell * price * (1 - TRANSACTION_COST)
            shares -= shares_to_sell
            cash += proceeds
        portfolio.append(cash + shares * price)

    df['Portfolio'] = portfolio
    returns = df['Portfolio'].pct_change().fillna(0)
    total_return = df['Portfolio'].iloc[-1] / initial_cash - 1
    sharpe = 0 if returns.std() == 0 else np.sqrt(252) * returns.mean() / returns.std()
    max_dd = (df['Portfolio'] / df['Portfolio'].cummax() - 1).min()
    return df, total_return, sharpe, max_dd

# =====================================================
# LONG / SHORT BACKTEST CORE
# =====================================================

def run_ls_backtest(df, rsi_len, rsi_threshold, max_hold, tp_pct, sl_pct, direction="long"):
    """
    direction = 'long'  → enter when RSI < rsi_threshold
    direction = 'short' → enter when RSI > rsi_threshold
    Returns list of trade dicts and a pd.Series of cumulative equity (trade-by-trade).
    """
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], rsi_len)
    df = df.dropna(subset=['RSI'])

    trades = []
    i = 0
    prices = df['Close'].values
    rsi_vals = df['RSI'].values
    dates = df.index

    in_trade = False
    entry_i = None
    entry_price = None

    while i < len(df):
        if not in_trade:
            rsi = rsi_vals[i]
            # Entry condition
            if direction == "long" and rsi < rsi_threshold:
                in_trade = True
                entry_i = i
                entry_price = prices[i]
            elif direction == "short" and rsi > rsi_threshold:
                in_trade = True
                entry_i = i
                entry_price = prices[i]
        else:
            price = prices[i]
            held = i - entry_i

            if direction == "long":
                ret = (price - entry_price) / entry_price
                hit_tp = ret >= tp_pct
                hit_sl = ret <= -sl_pct
            else:
                ret = (entry_price - price) / entry_price
                hit_tp = ret >= tp_pct
                hit_sl = ret <= -sl_pct

            exit_trade = hit_tp or hit_sl or held >= max_hold

            if exit_trade:
                net_ret = ret - TRANSACTION_COST * 2
                trades.append({
                    "entry_date": dates[entry_i],
                    "exit_date": dates[i],
                    "entry_price": entry_price,
                    "exit_price": price,
                    "held_days": held,
                    "gross_ret": ret,
                    "net_ret": net_ret,
                    "exit_reason": "TP" if hit_tp else ("SL" if hit_sl else "MaxHold"),
                })
                in_trade = False
                entry_i = None
                entry_price = None
        i += 1

    if not trades:
        return [], pd.Series(dtype=float), pd.Series(dtype=float)

    trade_df = pd.DataFrame(trades)
    cum_returns = (1 + trade_df['net_ret']).cumprod() - 1
    equity_curve = (1 + trade_df['net_ret']).cumprod()

    return trade_df, cum_returns, equity_curve


def compute_calmar(trade_df):
    if trade_df is None or len(trade_df) == 0:
        return -999, 0, 0

    equity = (1 + trade_df['net_ret']).cumprod()
    total_ret = equity.iloc[-1] - 1
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    if max_dd == 0:
        calmar = total_ret / 1e-6
    else:
        calmar = total_ret / abs(max_dd)

    return calmar, total_ret, max_dd


def optimize_tp_sl(df, rsi_len, rsi_threshold, max_hold, direction="long"):
    """Grid search over TP and SL to maximise Calmar ratio."""
    tp_grid = [0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30]
    sl_grid = [0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20]

    best_calmar = -999
    best_tp = tp_grid[4]
    best_sl = sl_grid[3]
    best_trade_df = None
    best_cum = None
    best_equity = None

    for tp, sl in itertools.product(tp_grid, sl_grid):
        trade_df, cum_ret, equity = run_ls_backtest(
            df, rsi_len, rsi_threshold, max_hold, tp, sl, direction
        )
        if trade_df is None or len(trade_df) < 3:
            continue
        calmar, total_ret, max_dd = compute_calmar(trade_df)
        if calmar > best_calmar:
            best_calmar = calmar
            best_tp = tp
            best_sl = sl
            best_trade_df = trade_df
            best_cum = cum_ret
            best_equity = equity

    return best_tp, best_sl, best_calmar, best_trade_df, best_cum, best_equity


def performance_summary(trade_df, direction_label):
    if trade_df is None or len(trade_df) == 0:
        return {"Strategy": direction_label, "Trades": 0}

    equity = (1 + trade_df['net_ret']).cumprod()
    total_ret = equity.iloc[-1] - 1
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    calmar = total_ret / abs(max_dd) if max_dd != 0 else np.nan
    win_rate = (trade_df['net_ret'] > 0).mean()
    avg_ret = trade_df['net_ret'].mean()
    avg_hold = trade_df['held_days'].mean()

    exits = trade_df['exit_reason'].value_counts().to_dict()

    return {
        "Strategy": direction_label,
        "Trades": len(trade_df),
        "Win Rate": f"{win_rate*100:.1f}%",
        "Avg Return/Trade": f"{avg_ret*100:.2f}%",
        "Cumulative Return": f"{total_ret*100:.2f}%",
        "Max Drawdown": f"{max_dd*100:.2f}%",
        "Calmar Ratio": f"{calmar:.2f}" if not np.isnan(calmar) else "N/A",
        "Avg Hold (days)": f"{avg_hold:.1f}",
        "TP exits": exits.get("TP", 0),
        "SL exits": exits.get("SL", 0),
        "MaxHold exits": exits.get("MaxHold", 0),
    }


# =====================================================
# PLOT HELPERS
# =====================================================

DARK_BG    = "#0f172a"
PANEL_BG   = "#1e293b"
PANEL_BG2  = "#0d1b2a"
GRID_COLOR = "#334155"
TEXT_COLOR = "#94a3b8"
WHITE      = "#f1f5f9"
GREEN      = "#22c55e"
RED        = "#ef4444"
GOLD       = "#facc15"
CYAN       = "#22d3ee"
BLUE       = "#60a5fa"
ORANGE     = "#fb923c"


def style_ax(ax, bg=PANEL_BG):
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(color=GRID_COLOR, alpha=0.4, lw=0.5)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)


def plot_equity_curve(ax, equity_curve, trade_df, color, label, title):
    style_ax(ax, PANEL_BG2)
    if equity_curve is None or len(equity_curve) == 0:
        ax.text(0.5, 0.5, "No trades found", transform=ax.transAxes,
                color=TEXT_COLOR, ha='center', va='center', fontsize=10)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)
        return

    x = np.arange(len(equity_curve))
    vals = equity_curve.values

    ax.plot(x, vals, color=color, lw=1.8, label=label)
    ax.fill_between(x, 1, vals, where=(vals >= 1), color=color, alpha=0.12)
    ax.fill_between(x, 1, vals, where=(vals < 1),  color=RED,   alpha=0.12)
    ax.axhline(1.0, color=GRID_COLOR, lw=0.8, ls='--')

    # Mark TP / SL / MaxHold exits
    if trade_df is not None and len(trade_df) > 0:
        for j, row in trade_df.iterrows():
            idx_j = trade_df.index.get_loc(j)
            if row['exit_reason'] == 'TP':
                ax.axvline(idx_j, color=GREEN, lw=0.6, alpha=0.4)
            elif row['exit_reason'] == 'SL':
                ax.axvline(idx_j, color=RED,   lw=0.6, alpha=0.4)

    # Running max drawdown shading
    running_max = pd.Series(vals).cummax()
    dd = (pd.Series(vals) - running_max) / running_max
    ax2 = ax.twinx()
    ax2.fill_between(x, dd.values * 100, 0, color=RED, alpha=0.18, label='Drawdown %')
    ax2.set_ylim(-100, 0)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax2.yaxis.label.set_color(TEXT_COLOR)
    ax2.set_ylabel("Drawdown %", fontsize=7, color=TEXT_COLOR)
    for spine in ax2.spines.values():
        spine.set_color(GRID_COLOR)

    ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel("Trade #", fontsize=8, color=TEXT_COLOR)
    ax.set_ylabel("Equity (×1)", fontsize=8, color=TEXT_COLOR)
    ax.legend(fontsize=7, facecolor=PANEL_BG2, edgecolor=GRID_COLOR,
              labelcolor=WHITE, framealpha=0.85)


# =====================================================
# RSI + CUBIC POLYNOMIAL FORECAST STRATEGY
# =====================================================

def cubic_rsi_forecast(rsi, window, horizon=1):
    """
    For every day t, fit a cubic polynomial to the last `window` RSI values
    (RSI[t-window+1] ... RSI[t]) and extrapolate it `horizon` days ahead.
    Uses only data up to day t (no look-ahead).

    Returns
    -------
    forecast : np.ndarray  forecast of RSI[t+horizon] made on day t (NaN where unavailable)
    coefs    : np.ndarray  (len, 4) cubic coefficients on the scaled x-axis
    x_mean, x_std : scaling used for the x-axis (for plotting the fitted curve)
    """
    vals = np.asarray(rsi, dtype=float)
    n = len(vals)
    forecast = np.full(n, np.nan)
    coefs = np.full((n, 4), np.nan)
    if n < window:
        return forecast, coefs, 0.0, 1.0

    x = np.arange(window, dtype=float)
    x_mean, x_std = x.mean(), x.std()
    xs = (x - x_mean) / x_std                       # scaled for numerical stability
    X = np.vander(xs, 4)                            # columns: x^3, x^2, x, 1
    P = np.linalg.pinv(X)                           # least-squares solver, same for every window
    xf = (window - 1 + horizon - x_mean) / x_std
    vf = np.array([xf ** 3, xf ** 2, xf, 1.0])

    W = np.lib.stride_tricks.sliding_window_view(vals, window)   # window ending at t = j + window - 1
    C = W @ P.T                                     # (n - window + 1, 4); NaN if the window has NaN
    coefs[window - 1:] = C
    forecast[window - 1:] = C @ vf
    return forecast, coefs, x_mean, x_std


def cubic_rsi_signals(df, rsi_len, oversold, overbought, horizon, use_forecast=True):
    """
    State machine on daily closes.

    Flat:  RSI < oversold          -> buy setup is ARMED
           armed and forecast up   -> BUY  (forecast RSI[t+h] > RSI[t])
           RSI > overbought        -> setup cancelled (never bought)
    Long:  RSI > overbought        -> sell setup is ARMED
           armed and forecast down -> SELL (forecast RSI[t+h] < RSI[t])

    With use_forecast=False the cubic filter is switched off (baseline):
    buy as soon as RSI < oversold, sell as soon as RSI > overbought.
    """
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], rsi_len)
    fc, coefs, x_mean, x_std = cubic_rsi_forecast(df['RSI'].values, rsi_len, horizon)
    df['RSI_fc'] = fc

    rsi_v = df['RSI'].values
    n = len(df)
    pos = np.zeros(n, dtype=int)          # desired position after the close of day t
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    armed_buy_arr = np.zeros(n, dtype=bool)
    armed_sell_arr = np.zeros(n, dtype=bool)

    in_pos, armed_buy, armed_sell = False, False, False
    for t in range(n):
        r, f = rsi_v[t], fc[t]
        if np.isnan(r) or (use_forecast and np.isnan(f)):
            pos[t] = int(in_pos)
            continue
        up = (f > r) if use_forecast else True
        down = (f < r) if use_forecast else True
        if not in_pos:
            if r < oversold:
                armed_buy = True
            elif r > overbought:
                armed_buy = False
            if armed_buy and up:
                buy[t], in_pos, armed_buy, armed_sell = True, True, False, False
        else:
            if r > overbought:
                armed_sell = True
            if armed_sell and down:
                sell[t], in_pos, armed_sell = True, False, False
        pos[t] = int(in_pos)
        armed_buy_arr[t], armed_sell_arr[t] = armed_buy, armed_sell

    df['Position'] = pos
    df['BuySig'] = buy
    df['SellSig'] = sell
    df['ArmedBuy'] = armed_buy_arr
    df['ArmedSell'] = armed_sell_arr
    return df, coefs, x_mean, x_std


def backtest_cubic(df_sig, risk_free_rate, execution="open"):
    """
    Daily backtest of the position series. 100% invested when long; cash earns the
    risk-free rate when flat; TRANSACTION_COST per side.
    execution='open'  : signal at close t, trade at the open of t+1.
    execution='close' : signal at close t, trade at the close of t.
    Returns (daily equity Series, daily strategy returns, trade DataFrame).
    """
    O = df_sig['Open'].values; C = df_sig['Close'].values
    sig = df_sig['Position'].values
    dates = df_sig.index
    n = len(df_sig)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1

    rets = np.zeros(n)
    trades, entry = [], None
    for d in range(1, n):
        if execution == "close":
            p_new = sig[d - 1]
            p_old = sig[d - 2] if d >= 2 else 0
            r = p_new * (C[d] / C[d - 1] - 1) + (1 - p_new) * daily_rf
            if p_new != p_old:                       # traded at the close of d-1
                r = (1 + r) * (1 - TRANSACTION_COST) - 1
                trade_date, trade_px = dates[d - 1], C[d - 1]
        else:
            p_old = sig[d - 2] if d >= 2 else 0       # held overnight into d
            p_new = sig[d - 1]                        # held from the open of d
            overnight = p_old * (O[d] / C[d - 1] - 1)
            intraday = p_new * (C[d] / O[d] - 1)
            r = (1 + overnight) * (1 + intraday) - 1 + (1 - p_new) * daily_rf
            if p_new != p_old:
                r = (1 + r) * (1 - TRANSACTION_COST) - 1
                trade_date, trade_px = dates[d], O[d]
        rets[d] = r

        if p_new != p_old:
            if p_new == 1:
                entry = (trade_date, trade_px, d)
            elif entry is not None:
                gross = trade_px / entry[1] - 1
                trades.append({
                    "entry_date": entry[0], "exit_date": trade_date,
                    "entry_price": entry[1], "exit_price": trade_px,
                    "held_days": d - entry[2],
                    "gross_ret": gross,
                    "net_ret": (1 + gross) * (1 - TRANSACTION_COST) ** 2 - 1,
                    "exit_reason": "Overbought + turn down",
                })
                entry = None

    if entry is not None:                             # open position, marked to market
        gross = C[-1] / entry[1] - 1
        trades.append({
            "entry_date": entry[0], "exit_date": dates[-1],
            "entry_price": entry[1], "exit_price": C[-1],
            "held_days": n - 1 - entry[2],
            "gross_ret": gross,
            "net_ret": (1 + gross) * (1 - TRANSACTION_COST) - 1,
            "exit_reason": "Open (marked to market)",
        })

    equity = pd.Series(np.cumprod(1 + rets), index=dates)
    return equity, pd.Series(rets, index=dates), pd.DataFrame(trades)


def cubic_performance(equity, rets, trade_df, df_sig, risk_free_rate, label):
    n_years = max(len(equity) / 252, 1e-9)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    xs = rets.iloc[1:] - daily_rf
    sharpe = np.sqrt(252) * xs.mean() / xs.std() if xs.std() > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    closed = trade_df[trade_df['exit_reason'] != "Open (marked to market)"] if len(trade_df) else trade_df
    return {
        "Strategy": label,
        "Total Return": f"{total*100:.1f}%",
        "CAGR": f"{cagr*100:.1f}%",
        "Sharpe (excess of rf)": f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A",
        "Max Drawdown": f"{max_dd*100:.1f}%",
        "Calmar (CAGR/MaxDD)": f"{calmar:.2f}" if not np.isnan(calmar) else "N/A",
        "Time in Market": f"{df_sig['Position'].mean()*100:.0f}%" if 'Position' in df_sig else "100%",
        "Trades": len(closed) if len(trade_df) else 0,
        "Win Rate": f"{(closed['net_ret'] > 0).mean()*100:.0f}%" if len(closed) else "N/A",
        "Avg Net Ret/Trade": f"{closed['net_ret'].mean()*100:.2f}%" if len(closed) else "N/A",
        "Avg Hold (days)": f"{closed['held_days'].mean():.1f}" if len(closed) else "N/A",
    }


# =====================================================
# MAIN APP
# =====================================================

st.title("📊 RSI Strategy Suite — Pivot Optimizer + Long/Short + Cubic Forecast")

data = download_stock_data(TICKER, str(START), END)

if data is None:
    st.error("No data found for this ticker.")
    st.stop()

st.success(f"Downloaded **{len(data)}** trading days for **{TICKER}**  |  {str(START)} → {END}")

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔁 RSI Pivot Strategy", "📈 Long / Short Strategies",
                             "🧮 RSI + Cubic Forecast"])

# =====================================================
# TAB 1 — original RSI pivot strategy (unchanged)
# =====================================================
with tab1:
    st.subheader("RSI Pivot Strategy Optimizer")

    split_index = int(len(data) * 0.75)
    train_data = data.iloc[:split_index]
    test_data  = data.iloc[split_index:]

    rsi_range    = [9, 10, 11]
    pivot_range  = [3, 5, 7, 10]
    bottom_range = [25, 30, 33, 37, 40]
    peak_range   = [66, 71, 75, 77, 80]

    best_score  = -999
    best_params = None

    with st.spinner("Optimizing Pivot parameters…"):
        for rsi_len in rsi_range:
            for pivot_lb in pivot_range:
                for bottom_th in bottom_range:
                    for peak_th in peak_range:
                        if peak_th <= bottom_th:
                            continue
                        _, _, sharpe, _ = backtest_strategy(
                            train_data, rsi_len, pivot_lb, bottom_th, peak_th, 0.0
                        )
                        if sharpe > best_score:
                            best_score  = sharpe
                            best_params = (rsi_len, pivot_lb, bottom_th, peak_th)

    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    col_p1.metric("RSI Length",      best_params[0])
    col_p2.metric("Pivot Lookback",  best_params[1])
    col_p3.metric("Bottom Thresh.",  best_params[2])
    col_p4.metric("Peak Thresh.",    best_params[3])
    col_p5.metric("Train Sharpe",    f"{best_score:.2f}")

    train_df, train_ret, train_sharpe, train_dd = backtest_strategy(
        train_data, *best_params, RISK_FREE_RATE_ANNUAL
    )
    test_df, test_ret, test_sharpe, test_dd = backtest_strategy(
        test_data, *best_params, RISK_FREE_RATE_ANNUAL
    )

    buy_hold_train = 10000 * (train_data['Close'] / train_data['Close'].iloc[0])
    buy_hold_test  = 10000 * (test_data['Close']  / test_data['Close'].iloc[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, facecolor=DARK_BG)
    fig.subplots_adjust(hspace=0.35)

    style_ax(ax1)
    ax1.plot(train_df.index, train_df['Portfolio'], label="Strategy (Train)", color=BLUE, lw=1.8)
    ax1.plot(test_df.index,  test_df['Portfolio'],  label="Strategy (Test)",  color=ORANGE, lw=1.8)
    ax1.plot(train_data.index, buy_hold_train, '--', color='#93c5fd', lw=1.2, label="Buy & Hold (Train)")
    ax1.plot(test_data.index,  buy_hold_test,  '--', color='#fdba74', lw=1.2, label="Buy & Hold (Test)")
    ax1.axvline(train_data.index[-1], color=WHITE, lw=1, ls=':', alpha=0.7)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=9)
    ax1.set_title("Portfolio Value Comparison", color=WHITE, fontsize=12, fontweight='bold')
    ax1.set_ylabel("Portfolio Value (₺)", color=TEXT_COLOR, fontsize=9)

    style_ax(ax2)
    ax2.plot(data.index, data['Close'], color='#64748b', lw=1.2, label="Price")
    all_signals = pd.concat([train_df, test_df])
    ax2.scatter(all_signals[all_signals['Bottom']].index,
                all_signals[all_signals['Bottom']]['Close'],
                marker='^', color=GREEN, s=80, zorder=5, label="Buy Signal")
    ax2.scatter(all_signals[all_signals['Peak']].index,
                all_signals[all_signals['Peak']]['Close'],
                marker='v', color=RED, s=80, zorder=5, label="Sell Signal")
    ax2.axvline(train_data.index[-1], color=WHITE, lw=1, ls=':', alpha=0.7)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=9)
    ax2.set_title("Price Chart with Signals", color=WHITE, fontsize=12, fontweight='bold')
    ax2.set_ylabel("Price", color=TEXT_COLOR, fontsize=9)

    fig.patch.set_facecolor(DARK_BG)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Current Signal")
    full_df, _, _, _ = backtest_strategy(data, *best_params, RISK_FREE_RATE_ANNUAL)
    current_rsi = full_df['RSI'].iloc[-1]
    is_bottom   = full_df['Bottom'].iloc[-1]
    is_peak     = full_df['Peak'].iloc[-1]
    last_price  = data['Close'].iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Last Close", f"{float(last_price):.2f}")
    c2.metric("Current RSI", f"{float(current_rsi):.2f}")
    if is_bottom:
        c3.success(f"🟢 BUY SIGNAL")
    elif is_peak:
        c3.error(f"🔴 SELL SIGNAL")
    else:
        c3.info("⚪ HOLD")

# =====================================================
# TAB 2 — Long / Short strategies
# =====================================================
with tab2:
    st.subheader("Long-Only & Short-Only RSI Strategies")

    st.info(
        f"**Parameters from sidebar →**  RSI Period = {rsi_days_ls}  |  "
        f"Long entry RSI < {rsi_entry_long}  |  Short entry RSI > {rsi_entry_short}  |  "
        f"Max Holding Days = {max_holding_days}\n\n"
        "Take-Profit and Stop-Loss are **optimised** by grid search to maximise the **Calmar Ratio** "
        "(Cumulative Return ÷ Max Drawdown)."
    )

    with st.spinner("Optimising Long strategy (TP / SL grid search)…"):
        long_tp, long_sl, long_calmar, long_trades, long_cum, long_equity = optimize_tp_sl(
            data, rsi_days_ls, rsi_entry_long, max_holding_days, direction="long"
        )

    with st.spinner("Optimising Short strategy (TP / SL grid search)…"):
        short_tp, short_sl, short_calmar, short_trades, short_cum, short_equity = optimize_tp_sl(
            data, rsi_days_ls, rsi_entry_short, max_holding_days, direction="short"
        )

    # ---- Optimal parameter display ----
    st.markdown("#### Optimal TP / SL Found")
    oc1, oc2, oc3, oc4, oc5, oc6 = st.columns(6)
    oc1.metric("Long TP",     f"{long_tp*100:.0f}%")
    oc2.metric("Long SL",     f"{long_sl*100:.0f}%")
    oc3.metric("Long Calmar", f"{long_calmar:.2f}" if long_calmar > -998 else "N/A")
    oc4.metric("Short TP",    f"{short_tp*100:.0f}%")
    oc5.metric("Short SL",    f"{short_sl*100:.0f}%")
    oc6.metric("Short Calmar",f"{short_calmar:.2f}" if short_calmar > -998 else "N/A")

    # ---- Equity curve plots ----
    fig2, (axL, axS) = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)
    fig2.subplots_adjust(wspace=0.35)

    plot_equity_curve(
        axL, long_equity, long_trades, GREEN,
        "Long equity",
        f"Long-Only  |  RSI < {rsi_entry_long}  |  TP={long_tp*100:.0f}%  SL={long_sl*100:.0f}%"
    )
    plot_equity_curve(
        axS, short_equity, short_trades, CYAN,
        "Short equity",
        f"Short-Only  |  RSI > {rsi_entry_short}  |  TP={short_tp*100:.0f}%  SL={short_sl*100:.0f}%"
    )

    fig2.patch.set_facecolor(DARK_BG)
    st.pyplot(fig2)
    plt.close(fig2)

    # ---- Trade log (expandable) ----
    col_l, col_s = st.columns(2)
    with col_l:
        with st.expander("📋 Long Trade Log"):
            if long_trades is not None and len(long_trades) > 0:
                disp = long_trades[['entry_date','exit_date','entry_price','exit_price',
                                    'held_days','net_ret','exit_reason']].copy()
                disp['net_ret'] = (disp['net_ret'] * 100).round(2).astype(str) + '%'
                disp.columns = ['Entry','Exit','Entry Px','Exit Px','Days','Net Ret %','Reason']
                st.dataframe(disp, use_container_width=True)
            else:
                st.info("No trades generated.")

    with col_s:
        with st.expander("📋 Short Trade Log"):
            if short_trades is not None and len(short_trades) > 0:
                disp = short_trades[['entry_date','exit_date','entry_price','exit_price',
                                     'held_days','net_ret','exit_reason']].copy()
                disp['net_ret'] = (disp['net_ret'] * 100).round(2).astype(str) + '%'
                disp.columns = ['Entry','Exit','Entry Px','Exit Px','Days','Net Ret %','Reason']
                st.dataframe(disp, use_container_width=True)
            else:
                st.info("No trades generated.")

    # ---- Combined performance summary table ----
    st.markdown("#### 📊 Combined Performance Summary")

    long_summary  = performance_summary(long_trades,  "Long-Only")
    short_summary = performance_summary(short_trades, "Short-Only")

    summary_df = pd.DataFrame([long_summary, short_summary]).set_index("Strategy")
    st.dataframe(summary_df, use_container_width=True)

    # ---- Cumulative return comparison chart ----
    if (long_cum is not None and len(long_cum) > 0) or \
       (short_cum is not None and len(short_cum) > 0):

        fig3, ax3 = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
        style_ax(ax3, PANEL_BG2)

        if long_cum is not None and len(long_cum) > 0:
            ax3.plot(long_cum.values * 100, color=GREEN, lw=2,
                     label=f"Long-Only (TP={long_tp*100:.0f}% / SL={long_sl*100:.0f}%)")
            ax3.fill_between(range(len(long_cum)), long_cum.values * 100, 0,
                             where=(long_cum.values >= 0), color=GREEN, alpha=0.10)
            ax3.fill_between(range(len(long_cum)), long_cum.values * 100, 0,
                             where=(long_cum.values < 0),  color=RED,   alpha=0.10)

        if short_cum is not None and len(short_cum) > 0:
            ax3.plot(short_cum.values * 100, color=CYAN, lw=2,
                     label=f"Short-Only (TP={short_tp*100:.0f}% / SL={short_sl*100:.0f}%)")
            ax3.fill_between(range(len(short_cum)), short_cum.values * 100, 0,
                             where=(short_cum.values >= 0), color=CYAN, alpha=0.10)

        ax3.axhline(0, color=GRID_COLOR, lw=0.8, ls='--')
        ax3.set_title("Cumulative Return: Long vs Short (trade-by-trade)",
                      color=WHITE, fontsize=12, fontweight='bold', pad=8)
        ax3.set_xlabel("Trade #", color=TEXT_COLOR, fontsize=9)
        ax3.set_ylabel("Cumulative Return (%)", color=TEXT_COLOR, fontsize=9)
        ax3.legend(facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=9)

        fig3.patch.set_facecolor(DARK_BG)
        st.pyplot(fig3)
        plt.close(fig3)

# =====================================================
# TAB 3 — RSI + cubic polynomial forecast strategy
# =====================================================
with tab3:
    st.subheader("RSI Oversold / Overbought with Cubic-Polynomial Turn Confirmation")

    exec_mode = "open" if cf_execution.startswith("Next") else "close"
    st.info(
        f"**Rules (from sidebar)** — RSI period = cubic window = **{rsi_days_cf}** days  |  "
        f"Forecast horizon = **{cf_horizon}** day(s)  |  Execution: **{cf_execution}**\n\n"
        f"**BUY:** RSI falls below **{cf_oversold}** (oversold, setup armed) → buy on the first day the "
        f"cubic fit of the last {rsi_days_cf} RSI values forecasts the RSI to **rise** "
        f"(forecast RSI in {cf_horizon} day(s) > today's RSI).\n\n"
        f"**SELL:** while long, RSI rises above **{cf_overbought}** (overbought, exit armed) → sell on the "
        f"first day the cubic fit forecasts the RSI to **fall**. The position is held until then.\n\n"
        f"Long-only, 100% invested when long; cash earns the risk-free rate "
        f"({RISK_FREE_RATE_ANNUAL*100:.0f}% p.a.); cost {TRANSACTION_COST*100:.1f}% per side. "
        "The **baseline** row uses the same thresholds without the cubic filter "
        "(buy at RSI < oversold, sell at RSI > overbought) to show what the forecast adds."
    )

    if cf_overbought <= cf_oversold:
        st.error("Overbought threshold must be above the oversold threshold.")
        st.stop()

    with st.spinner("Running cubic-forecast backtest…"):
        cf_df, cf_coefs, cf_xm, cf_xs = cubic_rsi_signals(
            data, rsi_days_cf, cf_oversold, cf_overbought, cf_horizon, use_forecast=True)
        cf_equity, cf_rets, cf_trades = backtest_cubic(cf_df, RISK_FREE_RATE_ANNUAL, exec_mode)

        base_df, _, _, _ = cubic_rsi_signals(
            data, rsi_days_cf, cf_oversold, cf_overbought, cf_horizon, use_forecast=False)
        base_equity, base_rets, base_trades = backtest_cubic(base_df, RISK_FREE_RATE_ANNUAL, exec_mode)

    bh_rets = data['Close'].pct_change().fillna(0)
    bh_equity = (1 + bh_rets).cumprod()

    # ---- Headline metrics ----
    cf_total = cf_equity.iloc[-1] - 1
    bh_total = bh_equity.iloc[-1] - 1
    cf_mdd = (cf_equity / cf_equity.cummax() - 1).min()
    bh_mdd = (bh_equity / bh_equity.cummax() - 1).min()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Strategy Return", f"{cf_total*100:.1f}%", delta=f"{(cf_total-bh_total)*100:+.1f} pp vs B&H")
    m2.metric("Buy & Hold Return", f"{bh_total*100:.1f}%")
    m3.metric("Strategy Max DD", f"{cf_mdd*100:.1f}%", delta=f"B&H {bh_mdd*100:.1f}%", delta_color="off")
    n_closed = int((cf_trades['exit_reason'] != "Open (marked to market)").sum()) if len(cf_trades) else 0
    m4.metric("Closed Trades", n_closed)
    m5.metric("Time in Market", f"{cf_df['Position'].mean()*100:.0f}%")

    # ---- Main chart: price + signals, RSI + forecast, equity ----
    fig4 = plt.figure(figsize=(16, 13), facecolor=DARK_BG)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.2, 1.3, 1.8], hspace=0.32)
    axP = fig4.add_subplot(gs[0]); axR = fig4.add_subplot(gs[1], sharex=axP)
    axE = fig4.add_subplot(gs[2], sharex=axP)

    style_ax(axP)
    axP.plot(cf_df.index, cf_df['Close'], color='#cbd5e1', lw=1.2, label="Close")
    in_mkt = cf_df['Position'].values.astype(bool)
    axP.fill_between(cf_df.index, cf_df['Close'].min(), cf_df['Close'].max(),
                     where=in_mkt, color=GREEN, alpha=0.07, step='post', label="Long (signal)")
    b_idx = cf_df.index[cf_df['BuySig']]; s_idx = cf_df.index[cf_df['SellSig']]
    axP.scatter(b_idx, cf_df.loc[b_idx, 'Close'], marker='^', color=GREEN, s=90, zorder=5,
                label="Buy signal (oversold + forecast up)")
    axP.scatter(s_idx, cf_df.loc[s_idx, 'Close'], marker='v', color=RED, s=90, zorder=5,
                label="Sell signal (overbought + forecast down)")
    axP.set_title(f"{TICKER} — Price with Cubic-Forecast RSI Signals", color=WHITE,
                  fontsize=12, fontweight='bold')
    axP.set_ylabel("Price", color=TEXT_COLOR, fontsize=9)
    axP.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')

    style_ax(axR)
    axR.plot(cf_df.index, cf_df['RSI'], color=BLUE, lw=1.1, label=f"RSI({rsi_days_cf})")
    axR.axhline(cf_oversold, color=GREEN, ls='--', lw=0.9, label=f"Oversold {cf_oversold}")
    axR.axhline(cf_overbought, color=RED, ls='--', lw=0.9, label=f"Overbought {cf_overbought}")
    axR.fill_between(cf_df.index, 0, 100, where=cf_df['ArmedBuy'].values, color=GREEN, alpha=0.10,
                     step='post', label="Buy setup armed")
    axR.fill_between(cf_df.index, 0, 100, where=cf_df['ArmedSell'].values, color=RED, alpha=0.10,
                     step='post', label="Sell setup armed")
    axR.scatter(b_idx, cf_df.loc[b_idx, 'RSI'], marker='^', color=GREEN, s=50, zorder=5)
    axR.scatter(s_idx, cf_df.loc[s_idx, 'RSI'], marker='v', color=RED, s=50, zorder=5)
    axR.set_ylim(0, 100)
    axR.set_ylabel("RSI", color=TEXT_COLOR, fontsize=9)
    axR.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=7,
               loc='upper left', ncol=5)

    style_ax(axE)
    axE.plot(cf_equity.index, cf_equity.values * 10000, color=GOLD, lw=1.8, label="Cubic-forecast strategy")
    axE.plot(base_equity.index, base_equity.values * 10000, color=CYAN, lw=1.2, alpha=0.8,
             label="Baseline (no cubic filter)")
    axE.plot(bh_equity.index, bh_equity.values * 10000, '--', color='#64748b', lw=1.2, label="Buy & Hold")
    axE.set_ylabel("Portfolio Value (start 10,000)", color=TEXT_COLOR, fontsize=9)
    axE.set_title("Portfolio Value Comparison", color=WHITE, fontsize=12, fontweight='bold')
    axE.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')
    axEd = axE.twinx()
    dd_series = (cf_equity / cf_equity.cummax() - 1) * 100
    axEd.fill_between(dd_series.index, dd_series.values, 0, color=RED, alpha=0.15)
    axEd.set_ylim(-100, 0)
    axEd.tick_params(colors=TEXT_COLOR, labelsize=7)
    axEd.set_ylabel("Strategy drawdown %", fontsize=7, color=TEXT_COLOR)
    for spine in axEd.spines.values():
        spine.set_color(GRID_COLOR)

    fig4.patch.set_facecolor(DARK_BG)
    st.pyplot(fig4)
    plt.close(fig4)

    # ---- Latest cubic fit (illustration of the current forecast) ----
    st.markdown("#### 🔍 Latest Cubic Fit of the RSI")
    last_c = cf_coefs[-1]
    if not np.any(np.isnan(last_c)):
        w = rsi_days_cf
        x_hist = np.arange(w)
        x_fine = np.linspace(0, w - 1 + cf_horizon, 200)
        fitted = np.polyval(last_c, (x_fine - cf_xm) / cf_xs)
        fig5, ax5 = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
        style_ax(ax5, PANEL_BG2)
        ax5.plot(x_hist - (w - 1), cf_df['RSI'].values[-w:], 'o-', color=BLUE, lw=1.2, ms=4,
                 label=f"RSI, last {w} days")
        ax5.plot(x_fine - (w - 1), fitted, color=GOLD, lw=1.8, label="Cubic fit")
        fc_last = cf_df['RSI_fc'].iloc[-1]
        ax5.scatter([cf_horizon], [fc_last], color=GOLD, s=90, marker='*', zorder=5,
                    label=f"Forecast in {cf_horizon} day(s): {fc_last:.1f}")
        ax5.axhline(cf_oversold, color=GREEN, ls='--', lw=0.9)
        ax5.axhline(cf_overbought, color=RED, ls='--', lw=0.9)
        ax5.axvline(0, color=WHITE, ls=':', lw=0.8, alpha=0.6)
        ax5.set_xlabel("Days relative to today (0 = last close)", color=TEXT_COLOR, fontsize=9)
        ax5.set_ylabel("RSI", color=TEXT_COLOR, fontsize=9)
        ax5.legend(facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8)
        fig5.patch.set_facecolor(DARK_BG)
        st.pyplot(fig5)
        plt.close(fig5)

    # ---- Current signal ----
    st.markdown("#### 📍 Current Signal")
    last = cf_df.iloc[-1]
    cur_rsi, cur_fc = float(last['RSI']), float(last['RSI_fc'])
    direction = "rising ↗" if cur_fc > cur_rsi else "falling ↘"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Last Close", f"{float(last['Close']):.2f}")
    k2.metric("Current RSI", f"{cur_rsi:.1f}")
    k3.metric(f"Cubic forecast (+{cf_horizon}d)", f"{cur_fc:.1f}", delta=direction, delta_color="off")
    if last['BuySig']:
        k4.success("🟢 BUY signal today")
    elif last['SellSig']:
        k4.error("🔴 SELL signal today")
    elif last['Position'] == 1:
        k4.info("🟢 HOLD LONG" + (" — exit armed (overbought), waiting for turn down"
                                 if last['ArmedSell'] else ""))
    elif last['ArmedBuy']:
        k4.warning("🟡 FLAT — buy armed (oversold), waiting for forecast to turn up")
    else:
        k4.info("⚪ FLAT — no setup")

    # ---- Trade logs ----
    def show_trades(tdf):
        if tdf is not None and len(tdf) > 0:
            disp = tdf[['entry_date', 'exit_date', 'entry_price', 'exit_price',
                        'held_days', 'net_ret', 'exit_reason']].copy()
            disp['entry_price'] = disp['entry_price'].round(2)
            disp['exit_price'] = disp['exit_price'].round(2)
            disp['net_ret'] = (disp['net_ret'] * 100).round(2).astype(str) + '%'
            disp.columns = ['Entry', 'Exit', 'Entry Px', 'Exit Px', 'Days', 'Net Ret %', 'Reason']
            st.dataframe(disp, use_container_width=True)
        else:
            st.info("No trades generated.")

    t1, t2 = st.columns(2)
    with t1:
        with st.expander("📋 Cubic-Forecast Strategy — Trade Log", expanded=True):
            show_trades(cf_trades)
    with t2:
        with st.expander("📋 Baseline (no cubic filter) — Trade Log"):
            show_trades(base_trades)

    # ---- Performance summary ----
    st.markdown("#### 📊 Performance Summary")
    bh_df_dummy = pd.DataFrame({'Position': np.ones(len(data))}, index=data.index)
    summary_cf = pd.DataFrame([
        cubic_performance(cf_equity, cf_rets, cf_trades, cf_df, RISK_FREE_RATE_ANNUAL,
                          "RSI + Cubic Forecast"),
        cubic_performance(base_equity, base_rets, base_trades, base_df, RISK_FREE_RATE_ANNUAL,
                          "Baseline: RSI thresholds only"),
        cubic_performance(bh_equity, bh_rets, pd.DataFrame(), bh_df_dummy, RISK_FREE_RATE_ANNUAL,
                          "Buy & Hold"),
    ]).set_index("Strategy")
    st.dataframe(summary_cf, use_container_width=True)
    st.caption(
        "Sharpe uses daily returns in excess of the risk-free rate. Calmar = CAGR / |max drawdown|. "
        "A still-open position is marked to market at the last close and is not counted in Trades, "
        "Win Rate or averages. Parameters are chosen by you, not optimised, so the full period is "
        "out of sample with respect to parameter search.")

st.caption("Developed for educational and research purposes — RSI Pivot + Long/Short + Cubic Forecast Strategy Suite.")







# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="RSI Strategy Backtester", layout="wide")

# # ------------------------------
# # Title
# # ------------------------------
# st.title("📈 RSI Threshold Backtester and Signal Estimator")

# # ------------------------------
# # Sidebar Parameters
# # ------------------------------
# st.sidebar.header("🔧 User Parameters")

# ticker = st.sidebar.text_input("Ticker Symbol (e.g., XU030.IS)", "XU030.IS")
# period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y","3y","4y"], index=1)
# tcost = st.sidebar.number_input("Transaction Cost (e.g., 0.002 = 0.2%)", value=0.0020, step=0.0005)
# capital = st.sidebar.number_input("Initial Capital (TRY)", value=1_000_000, step=50_000)
# rsi_period = st.sidebar.slider("RSI Period", 5, 30, 9)
# buy_threshold = st.sidebar.slider("Buy Threshold (RSI < x1)", 5, 45, 40)
# sell_threshold = st.sidebar.slider("Sell Threshold (RSI > x2)", 55, 95, 63)

# # ------------------------------
# # Download Data
# # ------------------------------
# st.subheader(f"📊 Downloading data for: {ticker}")
# df = yf.download(ticker, period=period, auto_adjust=True)

# if df.empty:
#     st.error("⚠️ No data found for the selected ticker.")
#     st.stop()

# df = df.dropna()
# df = df[["Open", "High", "Low", "Close", "Volume"]]

# # ------------------------------
# # RSI Calculation Function
# # ------------------------------
# def compute_rsi(series, period=14):
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
#     loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

# df["RSI1"] = compute_rsi(df["Close"], rsi_period)
# df = df.dropna()

# # ------------------------------
# # Backtest Function
# # ------------------------------
# def backtest_strategy(df, x1, x2):
#     open_positions = []
#     closed_trades = []

#     for i in range(1, len(df)):
#         rsi1 = df["RSI1"].iloc[i]
#         price = df["Close"].iloc[i]
#         date = df.index[i]

#         if rsi1 < x1:
#             open_positions.append({"entry_price": price, "entry_date": date})
#         elif rsi1 > x2 and len(open_positions) > 0:
#             entry = open_positions.pop(0)
#             closed_trades.append({
#                 "buy_date": entry["entry_date"],
#                 "buy_price": entry["entry_price"],
#                 "sell_date": date,
#                 "sell_price": price,
#                 "return": (price - entry["entry_price"]) / entry["entry_price"] - tcost
#             })

#     total_return = np.sum([t["return"] for t in closed_trades])
#     return total_return, closed_trades

# # ------------------------------
# # Run Backtest with User Thresholds
# # ------------------------------
# total_return, trades = backtest_strategy(df, buy_threshold, sell_threshold)

# st.success(f"✅ RSI thresholds used: Buy = {buy_threshold}, Sell = {sell_threshold}")
# st.write(f"**Total Return:** {total_return:.2f} | **Number of Trades:** {len(trades)}| **Return per Trade:** {total_return/len(trades):.2f}")

# # ------------------------------
# # Plot Results
# # ------------------------------
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# ax1.plot(df["Close"], label=f"{ticker} Close", color="black", lw=1.3)
# if trades:
#     for t in trades:
#         ax1.scatter(t["buy_date"], t["buy_price"], color="green", marker="^", s=100)
#         ax1.scatter(t["sell_date"], t["sell_price"], color="red", marker="v", s=100)

# ax1.set_title(f"{ticker} RSI({rsi_period}) Strategy | Buy={buy_threshold}, Sell={sell_threshold}")
# ax1.legend()
# ax1.grid(True)

# ax2.plot(df["RSI1"], color="blue", lw=1.2, label="RSI")
# ax2.axhline(buy_threshold, color="green", linestyle="--", label=f"Buy ({buy_threshold})")
# ax2.axhline(sell_threshold, color="red", linestyle="--", label=f"Sell ({sell_threshold})")
# ax2.legend()
# ax2.grid(True)

# st.pyplot(fig)

# # ------------------------------
# # Latest Signal Estimation
# # ------------------------------
# # latest_close = df["Close"].iloc[-1]
# # latest_rsi = df["RSI1"].iloc[-1]
# latest_close = float(df["Close"].iloc[-1])
# latest_rsi = float(df["RSI1"].iloc[-1])

# if latest_rsi < buy_threshold:
#     signal = "BUY"
# elif latest_rsi > sell_threshold:
#     signal = "SELL"
# else:
#     signal = "HOLD"

# # Prepare recent data for regression
# required_cols = [c for c in ["Close", "RSI1"] if c in df.columns]
# recent = df.tail(20)[required_cols].dropna()
# x = recent["Close"].to_numpy().flatten()
# y = recent["RSI1"].to_numpy().flatten()

# if len(x) >= 2:
#     a, b = np.polyfit(x, y, 1)
#     buy_price_threshold = (buy_threshold - b) / a if a != 0 else np.nan
#     sell_price_threshold = (sell_threshold - b) / a if a != 0 else np.nan
# else:
#     buy_price_threshold = sell_price_threshold = np.nan

# per_trade_capital = capital / max(len(trades), 1)
# order_size = int(per_trade_capital // latest_close)

# # ------------------------------
# # Display Summary
# # ------------------------------
# st.subheader("📋 Latest Signal Summary")

# st.metric("Ticker", ticker)
# st.metric("Latest Close", f"{latest_close:.2f}")
# st.metric("Latest RSI", f"{latest_rsi:.2f}")
# st.metric("Signal", signal)

# if signal == "BUY":
#     st.success(f"**BUY SIGNAL** → Target Price ≈ {buy_price_threshold:.2f} TRY | Suggested Order: {order_size} shares (~{order_size * latest_close:,.0f} TRY)")
# elif signal == "SELL":
#     st.error(f"**SELL SIGNAL** → Target Price ≈ {sell_price_threshold:.2f} TRY | Trigger if price > {sell_price_threshold:.2f}")
# else:
#     st.info(f"**HOLD** → RSI Buy Trigger ≈ {buy_price_threshold:.2f}, Sell Trigger ≈ {sell_price_threshold:.2f}")

# st.caption("Developed for educational and research purposes.")






# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="RSI Strategy Optimizer", layout="wide")

# # ------------------------------
# # Title
# # ------------------------------
# st.title("📈 RSI Threshold Optimization and Signal Estimator")

# # ------------------------------
# # Sidebar Parameters
# # ------------------------------
# st.sidebar.header("🔧 User Parameters")

# ticker = st.sidebar.text_input("Ticker Symbol (e.g., XU030.IS)", "XU030.IS")
# period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=1)
# tcost = st.sidebar.number_input("Transaction Cost (e.g., 0.002 = 0.2%)", value=0.0020, step=0.0005)
# capital = st.sidebar.number_input("Initial Capital (TRY)", value=1_000_000, step=50_000)
# rsi_period = st.sidebar.slider("RSI Period", 5, 30, 9)
# search_low =  st.sidebar.slider("Buy Threshold Range Start", 5, 45, 20)
# search_high =  st.sidebar.slider("Sell Threshold Range Start", 55, 95, 60)

# # st.sidebar.write("Select the ranges where optimal RSI thresholds will be searched.")

# # ------------------------------
# # Download Data
# # ------------------------------
# st.subheader(f"📊 Downloading data for: {ticker}")
# df = yf.download(ticker, period=period, auto_adjust=True)

# if df.empty:
#     st.error("⚠️ No data found for the selected ticker.")
#     st.stop()

# df = df.dropna()
# df = df[["Open", "High", "Low", "Close", "Volume"]]

# # ------------------------------
# # RSI Calculation Function
# # ------------------------------
# def compute_rsi(series, period=14):
#     delta = series.diff()
#     gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
#     loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

# df["RSI1"] = compute_rsi(df["Close"], rsi_period)
# df = df.dropna()

# # ------------------------------
# # Backtest Function
# # ------------------------------
# def backtest_strategy(df, x1, x2):
#     open_positions = []
#     closed_trades = []

#     for i in range(1, len(df)):
#         rsi1 = df["RSI1"].iloc[i]
#         price = df["Close"].iloc[i]
#         date = df.index[i]

#         if rsi1 < x1:
#             open_positions.append({"entry_price": price, "entry_date": date})
#         elif rsi1 > x2 and len(open_positions) > 0:
#             entry = open_positions.pop(0)
#             closed_trades.append({
#                 "buy_date": entry["entry_date"],
#                 "buy_price": entry["entry_price"],
#                 "sell_date": date,
#                 "sell_price": price,
#                 "return": (price - entry["entry_price"]) / entry["entry_price"] - tcost
#             })

#     total_return = np.sum([t["return"] for t in closed_trades])
#     return total_return, closed_trades

# # ------------------------------
# # Optimization
# # ------------------------------
# progress_text = st.empty()
# best_return = -np.inf
# best_x1, best_x2, best_trades = None, None, []

# for x1 in range(search_low, 45):
#     for x2 in range(search_high, 90):
#         total_return, trades = backtest_strategy(df, x1, x2)
#         if total_return > best_return and len(trades) > 0:
#             best_return = total_return
#             best_x1, best_x2 = x1, x2
#             best_trades = trades

# st.success(f"✅ Optimal RSI thresholds found: Buy = {best_x1}, Sell = {best_x2}")
# st.write(f"**Total Return:** {best_return:.2f} | **Number of Trades:** {len(best_trades)}")

# # ------------------------------
# # Plot Results
# # ------------------------------
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# ax1.plot(df["Close"], label=f"{ticker} Close", color="black", lw=1.3)
# if best_trades:
#     for t in best_trades:
#         ax1.scatter(t["buy_date"], t["buy_price"], color="green", marker="^", s=100)
#         ax1.scatter(t["sell_date"], t["sell_price"], color="red", marker="v", s=100)

# ax1.set_title(f"{ticker} RSI({rsi_period}) Strategy | x1={best_x1}, x2={best_x2}")
# ax1.legend()
# ax1.grid(True)

# ax2.plot(df["RSI1"], color="blue", lw=1.2, label="RSI")
# ax2.axhline(best_x1, color="green", linestyle="--", label=f"Buy ({best_x1})")
# ax2.axhline(best_x2, color="red", linestyle="--", label=f"Sell ({best_x2})")
# ax2.legend()
# ax2.grid(True)

# st.pyplot(fig)

# # ------------------------------
# # Latest Signal Estimation
# # ------------------------------
# latest_close = df["Close"].iloc[-1]
# latest_rsi = df["RSI1"].iloc[-1]

# if latest_rsi < best_x1:
#     signal = "BUY"
# elif latest_rsi > best_x2:
#     signal = "SELL"
# else:
#     signal = "HOLD"

# # Ensure required columns exist
# required_cols = [c for c in ["Close", "RSI1"] if c in df.columns]
# recent = df.tail(20)[required_cols].dropna()

# x = recent["Close"].to_numpy().flatten()
# y = recent["RSI1"].to_numpy().flatten()


# if len(x) >= 2:
#     a, b = np.polyfit(x, y, 1)
#     buy_price_threshold = (best_x1 - b) / a if a != 0 else np.nan
#     sell_price_threshold = (best_x2 - b) / a if a != 0 else np.nan
# else:
#     buy_price_threshold = sell_price_threshold = np.nan

# per_trade_capital = capital / max(len(best_trades), 1)
# order_size = int(per_trade_capital // latest_close)

# # ------------------------------
# # Display Summary
# # ------------------------------
# st.subheader("📋 Latest Signal Summary")

# st.metric("Ticker", ticker)
# st.metric("Latest Close", f"{latest_close:.2f}")
# st.metric("Latest RSI", f"{latest_rsi:.2f}")
# st.metric("Signal", signal)

# if signal == "BUY":
#     st.success(f"**BUY SIGNAL** → Target Price ≈ {buy_price_threshold:.2f} TRY | Suggested Order: {order_size} shares (~{order_size * latest_close:,.0f} TRY)")
# elif signal == "SELL":
#     st.error(f"**SELL SIGNAL** → Target Price ≈ {sell_price_threshold:.2f} TRY | Trigger if price > {sell_price_threshold:.2f}")
# else:
#     st.info(f"**HOLD** → RSI Buy Trigger ≈ {buy_price_threshold:.2f}, Sell Trigger ≈ {sell_price_threshold:.2f}")

# st.caption("Developed for educational and research purposes.")
