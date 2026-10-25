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
st.sidebar.subheader("🎯 RSI Delta Strategy (multi-entry)")
rsi_days_d = st.sidebar.slider("RSI Period (Delta strategy)", 5, 30, 14)
d_entry_range = st.sidebar.slider("Entry threshold search range (buy when RSI below)", 10, 45, (20, 35))
d_delta_range = st.sidebar.slider("DELTA search range (sell when RSI rises by)", 5, 60, (15, 50))
d_objective = st.sidebar.selectbox(
    "Optimise for",
    ["Cumulative profit (sum over all trades)",
     "Average profit per trade",
     "Cumulative profit / max drawdown"],
    index=0)
d_min_trades = st.sidebar.slider("Minimum closed trades for a valid setting", 1, 50, 5)
d_position_size = st.sidebar.number_input("Capital per position (₺)", min_value=100, value=1000, step=100)
d_execution = st.sidebar.selectbox(
    "Execute trades at (Delta)",
    ["Same day's close", "Next day's open (no look-ahead)"], index=0,
    help="The signal uses the day's closing RSI. 'Same day's close' assumes you trade in the "
         "closing auction; 'next open' avoids any look-ahead.")

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
# RSI DELTA STRATEGY — multiple entries / multiple exits
# =====================================================

def rsi_delta_exit_table(rsi_vals, deltas, max_entry):
    """
    For every day i with RSI[i] < max_entry, find for each DELTA the first day j > i
    with RSI[j] >= RSI[i] + DELTA.  Returns an int array (n, len(deltas)); -1 = never reached.
    Uses only RSI values after the entry day (the exit is the first time the target is hit).
    """
    n = len(rsi_vals)
    deltas = np.asarray(deltas, dtype=float)
    exit_idx = np.full((n, len(deltas)), -1, dtype=int)
    for i in range(n - 1):
        r = rsi_vals[i]
        if np.isnan(r) or r >= max_entry:
            continue
        future = np.nan_to_num(rsi_vals[i + 1:], nan=-np.inf)
        run_max = np.maximum.accumulate(future)            # non-decreasing
        k = np.searchsorted(run_max, r + deltas, side='left')
        hit = k < len(future)
        exit_idx[i, hit] = i + 1 + k[hit]
    return exit_idx


def rsi_delta_trades(df, rsi_vals, exit_col, entry_th, delta, execution="close"):
    """Build the trade list for one (entry threshold, DELTA) setting."""
    O = df['Open'].values; C = df['Close'].values; dates = df.index
    n = len(df)
    entries = np.where((rsi_vals < entry_th) & ~np.isnan(rsi_vals))[0]
    rows = []
    for i in entries:
        j = exit_col[i]
        if execution == "close":
            e_i, e_px = i, C[i]
            closed = j >= 0
            x_i = j if closed else n - 1
            x_px = C[x_i]
        else:
            if i + 1 >= n:
                continue                                    # no next open to buy at
            e_i, e_px = i + 1, O[i + 1]
            closed = (j >= 0) and (j + 1 < n)
            x_i = j + 1 if closed else n - 1
            x_px = O[x_i] if closed else C[n - 1]
        gross = x_px / e_px - 1
        net = (1 + gross) * (1 - TRANSACTION_COST) ** (2 if closed else 1) - 1
        rows.append({
            "entry_date": dates[e_i], "exit_date": dates[x_i],
            "entry_rsi": rsi_vals[i], "target_rsi": rsi_vals[i] + delta,
            "exit_rsi": rsi_vals[j] if j >= 0 else np.nan,
            "entry_price": e_px, "exit_price": x_px,
            "held_days": x_i - e_i, "gross_ret": gross, "net_ret": net,
            "status": "Closed" if closed else "Open (marked to market)",
            "_e": e_i, "_x": x_i,
        })
    return pd.DataFrame(rows)


def rsi_delta_pnl_curve(df, trades, size):
    """
    Daily mark-to-market P&L (₺) of all positions and the number of open positions.
    Open position (entry cost already paid): size * (C[t] * (1 - cost) / entry_price - 1).
    Closed position: its realised net P&L from the exit day on.
    Vectorised with difference arrays.
    """
    C = df['Close'].values; n = len(df)
    if trades is None or len(trades) == 0:
        z = pd.Series(np.zeros(n), index=df.index)
        return z, z.astype(int)
    e = trades['_e'].values.astype(int); x = trades['_x'].values.astype(int)
    closed = (trades['status'] == "Closed").values
    end = np.where(closed, x, n)                         # open interval [e, end)
    inv = 1.0 / trades['entry_price'].values
    d_inv = np.zeros(n + 1); d_cnt = np.zeros(n + 1); d_real = np.zeros(n + 1)
    np.add.at(d_inv, e, inv);  np.add.at(d_inv, end, -inv)
    np.add.at(d_cnt, e, 1);    np.add.at(d_cnt, end, -1)
    np.add.at(d_real, x[closed], size * trades['net_ret'].values[closed])
    s_inv = np.cumsum(d_inv)[:n]; cnt = np.cumsum(d_cnt)[:n]; real = np.cumsum(d_real)[:n]
    pnl = size * (C * (1 - TRANSACTION_COST) * s_inv - cnt) + real
    return pd.Series(pnl, index=df.index), pd.Series(cnt.astype(int), index=df.index)


def rsi_delta_stats(trades, size):
    if trades is None or len(trades) == 0:
        return dict(total=0.0, n=0, n_closed=0, n_open=0, avg=np.nan, avg_closed=np.nan,
                    win=np.nan, hold=np.nan)
    closed = trades[trades['status'] == "Closed"]
    return dict(
        total=size * trades['net_ret'].sum(),               # realised + open marked to market
        n=len(trades), n_closed=len(closed), n_open=len(trades) - len(closed),
        avg=trades['net_ret'].mean(),
        avg_closed=closed['net_ret'].mean() if len(closed) else np.nan,
        win=(closed['net_ret'] > 0).mean() if len(closed) else np.nan,
        hold=closed['held_days'].mean() if len(closed) else np.nan,
    )


def optimise_rsi_delta(df, rsi_len, entry_grid, delta_grid, execution, size, objective, min_closed):
    d = df.copy()
    d['RSI'] = compute_rsi(d['Close'], rsi_len)
    rsi_vals = d['RSI'].values
    exit_tab = rsi_delta_exit_table(rsi_vals, delta_grid, max(entry_grid))
    rows = []
    for a, E in enumerate(entry_grid):
        for b, D in enumerate(delta_grid):
            tr = rsi_delta_trades(d, rsi_vals, exit_tab[:, b], E, D, execution)
            s = rsi_delta_stats(tr, size)
            if objective.startswith("Cumulative profit /") and s['n'] > 0:
                pnl, _ = rsi_delta_pnl_curve(d, tr, size)
                mdd = (pnl - pnl.cummax()).min()
                s['score'] = s['total'] / abs(mdd) if mdd < 0 else np.inf
                s['mdd'] = mdd
            elif objective.startswith("Average"):
                s['score'] = s['avg']
            else:
                s['score'] = s['total']
            if s['n_closed'] < min_closed:
                s['score'] = np.nan
            rows.append({"Entry RSI <": E, "DELTA": D, **s})
    grid = pd.DataFrame(rows)
    return d, rsi_vals, exit_tab, grid


@st.cache_data(show_spinner=False)
def cached_optimise_rsi_delta(df, rsi_len, entry_grid, delta_grid, execution, size, objective, min_closed):
    return optimise_rsi_delta(df, rsi_len, list(entry_grid), list(delta_grid),
                              execution, size, objective, min_closed)


def best_from_grid(grid):
    g = grid.dropna(subset=['score'])
    if g.empty:
        return None
    return g.sort_values(['score', 'total'], ascending=False).iloc[0]


# =====================================================
# MAIN APP
# =====================================================

st.title("📊 RSI Strategy Suite — Pivot Optimizer + Long/Short + RSI Delta")

data = download_stock_data(TICKER, str(START), END)

if data is None:
    st.error("No data found for this ticker.")
    st.stop()

st.success(f"Downloaded **{len(data)}** trading days for **{TICKER}**  |  {str(START)} → {END}")

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔁 RSI Pivot Strategy", "📈 Long / Short Strategies",
                             "🎯 RSI Delta Strategy"])

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
# TAB 3 — RSI DELTA strategy (multiple entries / exits)
# =====================================================
with tab3:
    st.subheader("RSI Delta Strategy — Multiple Entries, Exit After an RSI Rise of DELTA")

    exec_d = "open" if d_execution.startswith("Next") else "close"
    entry_grid = tuple(range(d_entry_range[0], d_entry_range[1] + 1))
    delta_grid = tuple(range(d_delta_range[0], d_delta_range[1] + 1))

    st.info(
        f"**Rules** — RSI period **{rsi_days_d}**. On **every** day the RSI closes below the entry "
        f"threshold **E**, a new position of **₺{d_position_size:,.0f}** is opened (positions can overlap: "
        "if the RSI is 28 today and 20 tomorrow, both days open a position). Each position is closed on "
        "the first later day its RSI reaches **its own entry RSI + DELTA** (entered at 20 with DELTA 25 → "
        "sold when RSI ≥ 45). No stop-loss and no time limit; positions still open at the end are marked "
        "to market.\n\n"
        f"**Search** — E from {entry_grid[0]} to {entry_grid[-1]} and DELTA from {delta_grid[0]} to "
        f"{delta_grid[-1]} ({len(entry_grid) * len(delta_grid)} settings), objective: **{d_objective}**, "
        f"at least {d_min_trades} closed trades. Cost {TRANSACTION_COST*100:.1f}% per side; "
        f"execution: {d_execution}."
    )

    with st.spinner(f"Searching {len(entry_grid) * len(delta_grid)} (E, DELTA) settings…"):
        d_df, d_rsi, d_exit_tab, d_grid = cached_optimise_rsi_delta(
            data, rsi_days_d, entry_grid, delta_grid, exec_d,
            float(d_position_size), d_objective, d_min_trades)

    best = best_from_grid(d_grid)
    if best is None:
        st.warning("No (E, DELTA) setting produced enough closed trades. "
                   "Lower the minimum number of trades or widen the ranges.")
        st.stop()

    bE, bD = int(best['Entry RSI <']), int(best['DELTA'])
    b_col = delta_grid.index(bD)
    best_trades = rsi_delta_trades(d_df, d_rsi, d_exit_tab[:, b_col], bE, bD, exec_d)
    pnl, n_open = rsi_delta_pnl_curve(d_df, best_trades, d_position_size)
    bst = rsi_delta_stats(best_trades, d_position_size)
    max_conc = int(n_open.max())
    peak_cap = max_conc * d_position_size
    mdd_try = float((pnl - pnl.cummax()).min())

    # Buy & hold with the same peak capital, for comparison
    bh_pnl = peak_cap * (d_df['Close'] / d_df['Close'].iloc[0] - 1)

    # ---- Optimal setting ----
    st.markdown("#### 🏆 Optimal Setting")
    o1, o2, o3, o4, o5, o6 = st.columns(6)
    o1.metric("Entry: RSI below", bE)
    o2.metric("Exit: RSI rise DELTA", bD)
    o3.metric("Cumulative profit", f"₺{bst['total']:,.0f}",
              delta=f"{bst['total'] / d_position_size * 100:.0f}% of one position", delta_color="off")
    o4.metric("Avg net profit / trade", f"{bst['avg']*100:.2f}%",
              delta=f"closed only: {bst['avg_closed']*100:.2f}%" if bst['n_closed'] else None,
              delta_color="off")
    o5.metric("Trades (closed / open)", f"{bst['n_closed']} / {bst['n_open']}")
    o6.metric("Win rate (closed)", f"{bst['win']*100:.0f}%" if bst['n_closed'] else "N/A")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Avg holding (closed)", f"{bst['hold']:.1f} days" if bst['n_closed'] else "N/A")
    p2.metric("Max positions open at once", max_conc, delta=f"peak capital ₺{peak_cap:,.0f}", delta_color="off")
    p3.metric("Return on peak capital", f"{bst['total'] / peak_cap * 100:.1f}%" if peak_cap else "N/A",
              delta=f"B&H same capital: {bh_pnl.iloc[-1] / peak_cap * 100:.1f}%" if peak_cap else None,
              delta_color="off")
    p4.metric("Max drawdown of P&L", f"₺{mdd_try:,.0f}")

    # ---- Heatmaps over the search grid ----
    st.markdown("#### 🗺️ Search Grid")
    hm_total = d_grid.pivot(index='DELTA', columns='Entry RSI <', values='total')
    hm_avg = d_grid.pivot(index='DELTA', columns='Entry RSI <', values='avg') * 100
    hm_valid = d_grid.pivot(index='DELTA', columns='Entry RSI <', values='score').notna()
    figH, (axH1, axH2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=DARK_BG)
    for axh, M, ttl, cmap, fmt in [
            (axH1, hm_total, "Cumulative profit (₺)", 'RdYlGn', "₺{:,.0f}"),
            (axH2, hm_avg, "Average net profit per trade (%)", 'RdYlGn', "{:.1f}%")]:
        style_ax(axh, PANEL_BG2); axh.grid(False)
        Mv = M.where(hm_valid)
        lim = np.nanmax(np.abs(Mv.values)) if np.isfinite(np.nanmax(np.abs(Mv.values))) else 1
        im = axh.imshow(Mv.values, aspect='auto', origin='lower', cmap=cmap, vmin=-lim, vmax=lim,
                        extent=[M.columns[0] - 0.5, M.columns[-1] + 0.5, M.index[0] - 0.5, M.index[-1] + 0.5])
        axh.scatter([bE], [bD], marker='*', s=260, color=WHITE, edgecolor='black', zorder=5,
                    label=f"Optimum: E={bE}, Δ={bD}")
        cb = figH.colorbar(im, ax=axh, fraction=0.046, pad=0.03)
        cb.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        axh.set_xlabel("Entry threshold E (buy when RSI < E)", color=TEXT_COLOR, fontsize=9)
        axh.set_ylabel("DELTA (sell when RSI ≥ entry RSI + DELTA)", color=TEXT_COLOR, fontsize=9)
        axh.set_title(ttl + "  —  grey = too few closed trades", color=WHITE, fontsize=11, fontweight='bold')
        axh.set_facecolor('#475569')
        axh.legend(facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')
    figH.patch.set_facecolor(DARK_BG)
    st.pyplot(figH)
    plt.close(figH)

    # ---- Main chart: P&L, RSI with entries / exits, price ----
    closed_tr = best_trades[best_trades['status'] == "Closed"]
    entry_days = d_df.index[np.where((d_rsi < bE) & ~np.isnan(d_rsi))[0]]
    exit_days = pd.DatetimeIndex(sorted(set(d_df.index[np.array(
        [j for j in d_exit_tab[:, b_col][(d_rsi < bE) & ~np.isnan(d_rsi)] if j >= 0], dtype=int)])))

    fig6 = plt.figure(figsize=(16, 14), facecolor=DARK_BG)
    gs6 = gridspec.GridSpec(3, 1, height_ratios=[1.9, 1.6, 1.9], hspace=0.34)
    ax1d = fig6.add_subplot(gs6[0]); ax2d = fig6.add_subplot(gs6[1], sharex=ax1d)
    ax3d = fig6.add_subplot(gs6[2], sharex=ax1d)

    style_ax(ax1d)
    ax1d.plot(pnl.index, pnl.values, color=GOLD, lw=1.9, label=f"Strategy P&L, marked to market (E={bE}, Δ={bD})")
    ax1d.plot(bh_pnl.index, bh_pnl.values, '--', color='#94a3b8', lw=1.2,
              label=f"Buy & Hold with peak capital ₺{peak_cap:,.0f}")
    ax1d.axhline(0, color=GRID_COLOR, lw=0.8)
    ax1d.set_ylabel("Cumulative P&L (₺)", color=TEXT_COLOR, fontsize=9)
    ax1d.set_title(f"Complete Backtest {d_df.index[0].date()} → {d_df.index[-1].date()}  |  "
                   f"Cumulative profit ₺{bst['total']:,.0f}  ·  {bst['n']} trades",
                   color=WHITE, fontsize=12, fontweight='bold')
    ax1d.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')
    ax1n = ax1d.twinx()
    ax1n.fill_between(n_open.index, n_open.values, 0, step='post', color=CYAN, alpha=0.15)
    ax1n.set_ylim(0, max(max_conc * 3, 1))
    ax1n.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax1n.set_ylabel("Open positions", fontsize=7, color=CYAN)
    for spine in ax1n.spines.values():
        spine.set_color(GRID_COLOR)

    style_ax(ax2d)
    ax2d.plot(d_df.index, d_rsi, color=BLUE, lw=1.1, label=f"RSI({rsi_days_d})")
    ax2d.axhline(bE, color=GREEN, ls='--', lw=1.0, label=f"Entry: RSI < {bE}")
    ax2d.scatter(entry_days, d_df.loc[entry_days, 'RSI'], color=GREEN, s=18, zorder=5,
                 label="Entry day (new position)")
    ax2d.scatter(exit_days, d_df.loc[exit_days, 'RSI'], color=RED, s=22, marker='v', zorder=5,
                 label=f"Exit day (RSI ≥ entry RSI + {bD})")
    last_rsi_d = float(d_rsi[-1])
    ax2d.annotate(f"RSI {last_rsi_d:.1f}", xy=(d_df.index[-1], last_rsi_d), xytext=(8, 8),
                  textcoords='offset points', color=BLUE, fontsize=8, fontweight='bold')
    ax2d.set_ylim(0, 100)
    ax2d.set_ylabel("RSI", color=TEXT_COLOR, fontsize=9)
    ax2d.set_title("RSI with Entries and Exits", color=WHITE, fontsize=12, fontweight='bold')
    ax2d.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=7, loc='upper left', ncol=4)

    style_ax(ax3d)
    ax3d.plot(d_df.index, d_df['Close'], color='#cbd5e1', lw=1.2, label="Close")
    ax3d.scatter(entry_days, d_df.loc[entry_days, 'Close'], marker='^', color=GREEN, s=40, zorder=5, label="Buy")
    ax3d.scatter(exit_days, d_df.loc[exit_days, 'Close'], marker='v', color=RED, s=50, zorder=5, label="Sell")
    ax3d.set_title(f"{TICKER} — Price with Buy / Sell Days", color=WHITE, fontsize=12, fontweight='bold')
    ax3d.set_ylabel("Price", color=TEXT_COLOR, fontsize=9)
    ax3d.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')
    fig6.patch.set_facecolor(DARK_BG)
    st.pyplot(fig6)
    plt.close(fig6)

    # ---- Current state ----
    st.markdown("#### 📍 Current State")
    c1d, c2d, c3d = st.columns(3)
    c1d.metric("Last Close", f"{float(d_df['Close'].iloc[-1]):.2f}")
    c2d.metric("Current RSI", f"{last_rsi_d:.1f}")
    if last_rsi_d < bE:
        c3d.success(f"🟢 BUY today — RSI {last_rsi_d:.1f} < {bE}; target RSI {last_rsi_d + bD:.1f}")
    else:
        c3d.info(f"⚪ No new entry — RSI {last_rsi_d:.1f} ≥ {bE}")
    open_pos = best_trades[best_trades['status'] != "Closed"]
    if len(open_pos):
        op = open_pos[['entry_date', 'entry_price', 'entry_rsi', 'target_rsi', 'held_days', 'net_ret']].copy()
        op['entry_price'] = op['entry_price'].round(2)
        op['entry_rsi'] = op['entry_rsi'].round(1); op['target_rsi'] = op['target_rsi'].round(1)
        op['net_ret'] = (op['net_ret'] * 100).round(2).astype(str) + '%'
        op.columns = ['Entry', 'Entry Px', 'Entry RSI', 'Sell when RSI ≥', 'Days open', 'Unrealised %']
        st.markdown(f"**{len(op)} open position(s)** waiting for their RSI target:")
        st.dataframe(op, use_container_width=True, hide_index=True)
    else:
        st.caption("No open positions.")

    # ---- Trade log ----
    with st.expander(f"📋 Trade Log — E={bE}, DELTA={bD}  ({len(best_trades)} trades)"):
        tl = best_trades[['entry_date', 'exit_date', 'entry_rsi', 'target_rsi', 'exit_rsi',
                          'entry_price', 'exit_price', 'held_days', 'net_ret', 'status']].copy()
        for c in ['entry_rsi', 'target_rsi', 'exit_rsi']:
            tl[c] = tl[c].round(1)
        tl['entry_price'] = tl['entry_price'].round(2); tl['exit_price'] = tl['exit_price'].round(2)
        tl['net_ret'] = (tl['net_ret'] * 100).round(2)
        tl.columns = ['Entry', 'Exit', 'Entry RSI', 'Target RSI', 'Exit RSI',
                      'Entry Px', 'Exit Px', 'Days', 'Net Ret %', 'Status']
        st.dataframe(tl, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download trade log (CSV)", tl.to_csv(index=False).encode("utf-8"),
                           file_name=f"{TICKER}_rsi_delta_trades.csv", mime="text/csv")

    # ---- Top settings ----
    st.markdown("#### 📊 Top 10 Settings")
    top = d_grid.dropna(subset=['score']).sort_values(['score', 'total'], ascending=False).head(10).copy()
    top_disp = pd.DataFrame({
        "Entry RSI <": top['Entry RSI <'], "DELTA": top['DELTA'],
        "Cumulative profit (₺)": top['total'].round(0),
        "Avg net / trade": (top['avg'] * 100).round(2).astype(str) + '%',
        "Closed": top['n_closed'], "Open": top['n_open'],
        "Win rate": (top['win'] * 100).round(0).astype(str) + '%',
        "Avg hold (days)": top['hold'].round(1),
    })
    st.dataframe(top_disp, use_container_width=True, hide_index=True)

    # ---- Profit vs DELTA at the optimal entry threshold ----
    st.markdown(f"#### 📈 Profit vs DELTA at Entry RSI < {bE}")
    sl = d_grid[d_grid['Entry RSI <'] == bE]
    figD, axD = plt.subplots(figsize=(14, 4), facecolor=DARK_BG)
    style_ax(axD, PANEL_BG2)
    axD.bar(sl['DELTA'], sl['total'], color=np.where(sl['total'] >= 0, GREEN, RED), alpha=0.75,
            label="Cumulative profit (₺)")
    axD.set_xlabel("DELTA", color=TEXT_COLOR, fontsize=9)
    axD.set_ylabel("Cumulative profit (₺)", color=TEXT_COLOR, fontsize=9)
    axD2 = axD.twinx()
    axD2.plot(sl['DELTA'], sl['avg'] * 100, color=GOLD, lw=1.8, marker='o', ms=3, label="Avg net / trade (%)")
    axD2.tick_params(colors=TEXT_COLOR, labelsize=7)
    axD2.set_ylabel("Avg net profit per trade (%)", color=GOLD, fontsize=8)
    axD.axvline(bD, color=WHITE, ls=':', lw=1)
    h1, l1 = axD.get_legend_handles_labels(); h2, l2 = axD2.get_legend_handles_labels()
    axD.legend(h1 + h2, l1 + l2, facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8,
               loc='upper left')
    figD.patch.set_facecolor(DARK_BG)
    st.pyplot(figD)
    plt.close(figD)

    # ---- Optional hold-out check ----
    st.markdown("#### 🧪 Hold-out Check (optional)")
    st.caption(
        "The optimum above is chosen on the same data it is evaluated on, so its profit is optimistic: "
        f"it is the best of {len(entry_grid) * len(delta_grid)} settings. This check chooses (E, DELTA) "
        "on the first 75% of the period only and then applies it, unchanged, to the last 25%.")
    if st.checkbox("Run hold-out check", value=False, key="d_holdout"):
        split = int(len(data) * 0.75)
        with st.spinner("Optimising on the first 75%…"):
            tr_df, tr_rsi, tr_exit, tr_grid = cached_optimise_rsi_delta(
                data.iloc[:split], rsi_days_d, entry_grid, delta_grid, exec_d,
                float(d_position_size), d_objective, d_min_trades)
        tb = best_from_grid(tr_grid)
        if tb is None:
            st.warning("Not enough trades in the first 75% to choose a setting.")
        else:
            tE, tD = int(tb['Entry RSI <']), int(tb['DELTA'])
            full_tr = rsi_delta_trades(d_df, d_rsi, d_exit_tab[:, delta_grid.index(tD)], tE, tD, exec_d)
            split_date = data.index[split]
            test_tr = full_tr[full_tr['entry_date'] >= split_date]
            ts = rsi_delta_stats(test_tr, d_position_size)
            h1c, h2c, h3c, h4c = st.columns(4)
            h1c.metric("Chosen on first 75%", f"E={tE}, Δ={tD}")
            h2c.metric("Hold-out cumulative profit", f"₺{ts['total']:,.0f}")
            h3c.metric("Hold-out avg / trade", f"{ts['avg']*100:.2f}%" if ts['n'] else "N/A")
            h4c.metric("Hold-out trades (closed / open)", f"{ts['n_closed']} / {ts['n_open']}")
            st.caption(f"Hold-out period: {split_date.date()} → {data.index[-1].date()}. "
                       "Trades entered in the hold-out period are counted; exits use the full price path.")

st.caption("Developed for educational and research purposes — RSI Pivot + Long/Short + RSI Delta Strategy Suite.")







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
