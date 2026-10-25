import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
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
st.sidebar.subheader("💰 RSI Accumulation (IRR) Strategy")
acc_rsi_days = st.sidebar.slider("RSI Period (Accumulation)", 5, 30, 14)
acc_th_range = st.sidebar.slider("RSI threshold search range (buy 1 share when RSI below)", 5, 95, (10, 70))
acc_min_buys = st.sidebar.slider("Minimum number of purchases for a valid threshold", 1, 100, 10)
acc_include_cost = st.sidebar.checkbox("Include transaction cost on each purchase", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🕯️ Heikin-Ashi Strategy")
ha_execution = st.sidebar.selectbox(
    "Execute trades at (Heikin-Ashi)",
    ["Next day's open (no look-ahead)", "Same day's close"], index=0,
    help="The bar's colour is known only at the close. 'Next day's open' avoids look-ahead; "
         "'Same day's close' assumes you trade in the closing auction.")
ha_zoom_days = st.sidebar.slider("Heikin-Ashi candle chart: last N days", 30, 250, 120)

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
# HEIKIN-ASHI STRATEGY — long while the HA bar is green, cash otherwise
# =====================================================

def heikin_ashi(df):
    """Heikin-Ashi bars computed recursively from past data only."""
    O = df['Open'].values; H = df['High'].values; L = df['Low'].values; C = df['Close'].values
    n = len(df)
    ha_c = (O + H + L + C) / 4.0
    ha_o = np.empty(n)
    ha_o[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum.reduce([H, ha_o, ha_c])
    ha_l = np.minimum.reduce([L, ha_o, ha_c])
    out = pd.DataFrame({'HA_Open': ha_o, 'HA_High': ha_h, 'HA_Low': ha_l, 'HA_Close': ha_c},
                       index=df.index)
    out['Green'] = out['HA_Close'] >= out['HA_Open']
    return out


def backtest_position(df, sig, risk_free_rate, execution="open"):
    """
    Daily backtest of a 0/1 position signal decided at the close of each day.
    100% invested when long; cash earns the risk-free rate when flat; TRANSACTION_COST per side.
    execution='open' : trade at the next day's open.   execution='close': trade at the same close.
    Returns (equity Series, daily return Series, trade DataFrame, held-position Series).
    """
    O = df['Open'].values; C = df['Close'].values
    sig = np.asarray(sig, dtype=int)
    dates = df.index
    n = len(df)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    rets = np.zeros(n); held = np.zeros(n)
    trades, entry = [], None
    for d in range(1, n):
        p_old = sig[d - 2] if d >= 2 else 0
        p_new = sig[d - 1]
        if execution == "close":
            r = p_new * (C[d] / C[d - 1] - 1) + (1 - p_new) * daily_rf
            t_date, t_px = dates[d - 1], C[d - 1]
        else:
            overnight = p_old * (O[d] / C[d - 1] - 1)
            intraday = p_new * (C[d] / O[d] - 1)
            r = (1 + overnight) * (1 + intraday) - 1 + (1 - p_new) * daily_rf
            t_date, t_px = dates[d], O[d]
        if p_new != p_old:
            r = (1 + r) * (1 - TRANSACTION_COST) - 1
            if p_new == 1:
                entry = (t_date, t_px, d)
            elif entry is not None:
                gross = t_px / entry[1] - 1
                trades.append({"entry_date": entry[0], "exit_date": t_date,
                               "entry_price": entry[1], "exit_price": t_px,
                               "held_days": d - entry[2], "gross_ret": gross,
                               "net_ret": (1 + gross) * (1 - TRANSACTION_COST) ** 2 - 1,
                               "status": "Closed"})
                entry = None
        rets[d] = r; held[d] = p_new
    if entry is not None:
        gross = C[-1] / entry[1] - 1
        trades.append({"entry_date": entry[0], "exit_date": dates[-1],
                       "entry_price": entry[1], "exit_price": C[-1],
                       "held_days": n - 1 - entry[2], "gross_ret": gross,
                       "net_ret": (1 + gross) * (1 - TRANSACTION_COST) - 1,
                       "status": "Open (marked to market)"})
    return (pd.Series(np.cumprod(1 + rets), index=dates), pd.Series(rets, index=dates),
            pd.DataFrame(trades), pd.Series(held, index=dates))


def equity_stats(equity, rets, risk_free_rate, held=None, trades=None, label=""):
    n_years = max((len(equity) - 1) / 252, 1e-9)
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    vol = rets.iloc[1:].std() * np.sqrt(252)
    xs = rets.iloc[1:] - daily_rf
    sharpe = np.sqrt(252) * xs.mean() / xs.std() if xs.std() > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    row = {"Strategy": label,
           "Total Return": f"{total*100:.1f}%", "CAGR": f"{cagr*100:.1f}%",
           "Volatility": f"{vol*100:.1f}%",
           "Sharpe (excess of rf)": f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A",
           "Max Drawdown": f"{mdd*100:.1f}%",
           "Calmar": f"{calmar:.2f}" if np.isfinite(calmar) else "N/A",
           "Time in Market": f"{held.iloc[1:].mean()*100:.0f}%" if held is not None else "—"}
    if trades is not None and len(trades):
        closed = trades[trades['status'] == "Closed"]
        row.update({"Trades": len(closed),
                    "Trades / year": f"{len(closed) / n_years:.1f}",
                    "Win Rate": f"{(closed['net_ret'] > 0).mean()*100:.0f}%" if len(closed) else "N/A",
                    "Avg Net / Trade": f"{closed['net_ret'].mean()*100:.2f}%" if len(closed) else "N/A",
                    "Avg Hold (days)": f"{closed['held_days'].mean():.1f}" if len(closed) else "N/A"})
    else:
        row.update({"Trades": "—", "Trades / year": "—", "Win Rate": "—",
                    "Avg Net / Trade": "—", "Avg Hold (days)": "—"})
    return row


def draw_ha_candles(ax, ha, width=0.6):
    """Heikin-Ashi candlesticks (green = HA close >= HA open)."""
    x = mdates.date2num(ha.index.to_pydatetime())
    for xi, (o, h, l, c, g) in zip(x, ha[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'Green']].values):
        col = GREEN if g else RED
        ax.vlines(xi, l, h, color=col, lw=0.8, zorder=3)
        ax.add_patch(plt.Rectangle((xi - width / 2, min(o, c)), width, max(abs(c - o), 1e-9),
                                   facecolor=col, edgecolor=col, alpha=0.85, zorder=4))
    ax.xaxis_date()


# =====================================================
# RSI ACCUMULATION STRATEGY — buy one share on every day RSI < threshold,
# never sell; measure the internal rate of return (IRR)
# =====================================================

def solve_irr(outlays, tau, terminal_value, lo=-5.0, hi=5.0, tol=1e-10, max_iter=200):
    """
    Continuously compounded annual rate r solving
        sum_i outlay_i * exp(r * tau_i) = terminal_value,
    where tau_i = years from purchase i to the valuation date T.
    The left side is increasing in r (all outlays > 0), so the root is unique; bisection.
    """
    outlays = np.asarray(outlays, float); tau = np.asarray(tau, float)
    if len(outlays) == 0 or terminal_value <= 0:
        return np.nan
    f = lambda r: np.sum(outlays * np.exp(r * tau)) - terminal_value
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def accumulation_result(df, rsi_vals, threshold, include_cost=True):
    """Buy one share at the close on every day with RSI < threshold; hold to the last date T."""
    C = df['Close'].values
    mask = (rsi_vals < threshold) & ~np.isnan(rsi_vals)
    idx = np.where(mask)[0]
    T = df.index[-1]
    cost_mult = (1 + TRANSACTION_COST) if include_cost else 1.0
    outlays = C[idx] * cost_mult
    tau = (T - df.index[idx]).days.values / 365.25
    n_sh = len(idx)
    terminal = C[-1] * n_sh
    r = solve_irr(outlays, tau, terminal)
    return {
        "threshold": threshold, "buys": n_sh, "invested": outlays.sum(),
        "final_value": terminal, "profit": terminal - outlays.sum(),
        "irr_cont": r, "irr_eff": np.exp(r) - 1 if np.isfinite(r) else np.nan,
        "avg_price": outlays.sum() / n_sh if n_sh else np.nan,
        "idx": idx, "outlays": outlays, "tau": tau,
    }


def accumulation_path(df, idx, outlays, rf_cont):
    """Daily path: cumulative cash invested, market value of shares held, and the same
    cash flows compounded at the risk-free rate."""
    n = len(df)
    shares = np.zeros(n); cash_in = np.zeros(n); rf_flow = np.zeros(n)
    np.add.at(shares, idx, 1); np.add.at(cash_in, idx, outlays)
    shares = np.cumsum(shares); invested = np.cumsum(cash_in)
    t_years = (df.index - df.index[0]).days.values / 365.25
    # value at t of all outlays made up to t, compounded at rf: sum outlay_i * exp(rf (t - t_i))
    disc = np.cumsum(cash_in * np.exp(-rf_cont * t_years))
    rf_value = disc * np.exp(rf_cont * t_years)
    mkt_value = shares * df['Close'].values
    return pd.DataFrame({"Shares": shares, "Invested": invested, "Market value": mkt_value,
                         "Same cash at risk-free": rf_value}, index=df.index)


# =====================================================
# MAIN APP
# =====================================================

st.title("📊 Strategy Suite — RSI Accumulation + Long/Short + Heikin-Ashi")

data = download_stock_data(TICKER, str(START), END)

if data is None:
    st.error("No data found for this ticker.")
    st.stop()

st.success(f"Downloaded **{len(data)}** trading days for **{TICKER}**  |  {str(START)} → {END}")

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💰 RSI Accumulation (IRR)", "📈 Long / Short Strategies",
                             "🕯️ Heikin-Ashi Strategy"])

# =====================================================
# TAB 1 — RSI accumulation strategy (IRR)
# =====================================================
with tab1:
    st.subheader("RSI Accumulation — Buy One Share Every Day the RSI Is Below the Threshold, Never Sell")

    rf_cont = np.log(1 + RISK_FREE_RATE_ANNUAL)
    st.info(
        f"**Rule** — RSI period **{acc_rsi_days}**. On **every** day the RSI closes below the threshold, buy "
        "**one share** at the close and keep it for life (no sales).\n\n"
        "**Internal rate of return** — with purchases on dates t₁ … tₙ and N shares in total, r solves  "
        "**Σ S(tᵢ)·e^{r(T − tᵢ)} = S(T)·N**, where T is the last date and time is in years. "
        f"Purchases {'include' if acc_include_cost else 'exclude'} the {TRANSACTION_COST*100:.1f}% cost; "
        "the shares are valued at the last close.\n\n"
        f"The threshold is chosen from {acc_th_range[0]} to {acc_th_range[1]} to **maximise r** (at least "
        f"{acc_min_buys} purchases). Benchmarks: **buy & hold** (one share bought on the first day), "
        "**buy one share every day** (the same plan without the RSI filter), and the **risk-free rate**."
    )

    acc_df = data.copy()
    acc_df['RSI'] = compute_rsi(acc_df['Close'], acc_rsi_days)
    acc_rsi = acc_df['RSI'].values
    first_valid = int(np.argmax(~np.isnan(acc_rsi)))
    acc_eval = acc_df.iloc[first_valid:]            # common period: from the first day with an RSI
    rsi_eval = acc_rsi[first_valid:]

    thresholds = list(range(acc_th_range[0], acc_th_range[1] + 1))
    res = [accumulation_result(acc_eval, rsi_eval, th, acc_include_cost) for th in thresholds]
    grid = pd.DataFrame([{k: v for k, v in r_.items() if k not in ("idx", "outlays", "tau")} for r_ in res])
    valid = grid[(grid['buys'] >= acc_min_buys) & grid['irr_cont'].notna()]
    if valid.empty:
        st.warning("No threshold gives enough purchases. Lower the minimum or widen the range.")
        st.stop()
    best_row = valid.loc[valid['irr_cont'].idxmax()]
    best_th = int(best_row['threshold'])
    best = res[thresholds.index(best_th)]

    # Benchmarks over the same period
    C_eval = acc_eval['Close'].values
    years_total = (acc_eval.index[-1] - acc_eval.index[0]).days / 365.25
    cm = (1 + TRANSACTION_COST) if acc_include_cost else 1.0
    bh_cont = np.log(C_eval[-1] / (C_eval[0] * cm)) / years_total
    every_day = accumulation_result(acc_eval, np.zeros(len(acc_eval)), 1.0, acc_include_cost)  # RSI 0 < 1 always
    ed_cont = every_day['irr_cont']

    def pct_eff(r):
        return f"{(np.exp(r) - 1) * 100:.2f}%" if np.isfinite(r) else "N/A"

    # ---- Headline ----
    st.markdown("#### 🏆 Optimal Threshold")
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("Buy when RSI below", best_th)
    a2.metric("IRR (annual, effective)", pct_eff(best['irr_cont']),
              delta=f"continuous r = {best['irr_cont']*100:.2f}%", delta_color="off")
    a3.metric("Buy & Hold (annual)", pct_eff(bh_cont),
              delta=f"{(best['irr_cont'] - bh_cont)*100:+.2f} pp (cont.)")
    a4.metric("Buy every day (annual)", pct_eff(ed_cont),
              delta=f"{(best['irr_cont'] - ed_cont)*100:+.2f} pp (cont.)")
    a5.metric("Risk-free (annual)", f"{RISK_FREE_RATE_ANNUAL*100:.2f}%",
              delta=f"{(best['irr_cont'] - rf_cont)*100:+.2f} pp (cont.)")
    a6.metric("Shares accumulated", f"{best['buys']:,}")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Total invested", f"{best['invested']:,.0f}")
    b2.metric("Value at last close", f"{best['final_value']:,.0f}")
    b3.metric("Profit", f"{best['profit']:,.0f}",
              delta=f"{best['profit'] / best['invested'] * 100:.1f}% of invested", delta_color="off")
    b4.metric("Average cost per share", f"{best['avg_price']:.2f}",
              delta=f"last close {C_eval[-1]:.2f}", delta_color="off")
    st.caption(f"Evaluation period {acc_eval.index[0].date()} → {acc_eval.index[-1].date()} "
               f"({years_total:.2f} years; starts on the first day the RSI is available). "
               "Effective annual rate = e^r − 1. Differences in pp are between continuous rates.")

    # ---- IRR vs threshold ----
    figT, (axT1, axT2) = plt.subplots(1, 2, figsize=(16, 5), facecolor=DARK_BG)
    style_ax(axT1, PANEL_BG2)
    ok = grid['buys'] >= acc_min_buys
    axT1.plot(grid['threshold'], (np.exp(grid['irr_cont']) - 1) * 100, color=GOLD, lw=1.0, alpha=0.4)
    axT1.plot(grid.loc[ok, 'threshold'], (np.exp(grid.loc[ok, 'irr_cont']) - 1) * 100, 'o-', color=GOLD,
              lw=1.8, ms=3, label="IRR of RSI accumulation")
    axT1.axhline((np.exp(bh_cont) - 1) * 100, color='#94a3b8', ls='--', lw=1.2, label="Buy & Hold")
    axT1.axhline((np.exp(ed_cont) - 1) * 100, color=CYAN, ls='-.', lw=1.2, label="Buy every day")
    axT1.axhline(RISK_FREE_RATE_ANNUAL * 100, color=GREEN, ls=':', lw=1.4, label="Risk-free")
    axT1.axvline(best_th, color=WHITE, ls=':', lw=1)
    axT1.scatter([best_th], [(np.exp(best['irr_cont']) - 1) * 100], marker='*', s=220, color=WHITE,
                 edgecolor='black', zorder=5, label=f"Optimum: RSI < {best_th}")
    axT1.set_xlabel("RSI threshold (buy when RSI below)", color=TEXT_COLOR, fontsize=9)
    axT1.set_ylabel("Annual return (%)", color=TEXT_COLOR, fontsize=9)
    axT1.set_title("Internal Rate of Return vs RSI Threshold", color=WHITE, fontsize=11, fontweight='bold')
    axT1.legend(facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8)

    style_ax(axT2, PANEL_BG2)
    axT2.bar(grid['threshold'], grid['buys'], color=BLUE, alpha=0.6, label="Shares bought")
    axT2.set_xlabel("RSI threshold", color=TEXT_COLOR, fontsize=9)
    axT2.set_ylabel("Shares bought", color=TEXT_COLOR, fontsize=9)
    axT2b = axT2.twinx()
    axT2b.plot(grid['threshold'], grid['profit'], color=ORANGE, lw=1.8, label="Profit")
    axT2b.tick_params(colors=TEXT_COLOR, labelsize=7)
    axT2b.set_ylabel("Profit at last close", color=ORANGE, fontsize=8)
    axT2.axvline(best_th, color=WHITE, ls=':', lw=1)
    h1, l1 = axT2.get_legend_handles_labels(); h2, l2 = axT2b.get_legend_handles_labels()
    axT2.legend(h1 + h2, l1 + l2, facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8,
                loc='upper left')
    axT2.set_title("Size of the Plan: Shares Bought and Profit", color=WHITE, fontsize=11, fontweight='bold')
    figT.patch.set_facecolor(DARK_BG)
    st.pyplot(figT)
    plt.close(figT)

    # ---- Accumulation path, RSI and purchases ----
    path = accumulation_path(acc_eval, best['idx'], best['outlays'], rf_cont)
    buy_days = acc_eval.index[best['idx']]
    figA = plt.figure(figsize=(16, 13), facecolor=DARK_BG)
    gsA = gridspec.GridSpec(3, 1, height_ratios=[1.7, 1.3, 1.6], hspace=0.33)
    axW = figA.add_subplot(gsA[0]); axR1 = figA.add_subplot(gsA[1], sharex=axW)
    axS = figA.add_subplot(gsA[2], sharex=axW)

    style_ax(axW)
    axW.plot(path.index, path['Market value'], color=GOLD, lw=1.9, label="Market value of shares held")
    axW.plot(path.index, path['Same cash at risk-free'], color=GREEN, lw=1.3, ls=':',
             label="Same purchases invested at the risk-free rate")
    axW.plot(path.index, path['Invested'], color='#94a3b8', lw=1.2, ls='--', label="Cumulative cash invested")
    axW.set_ylabel("Value", color=TEXT_COLOR, fontsize=9)
    axW.set_title(f"Accumulation Path — RSI < {best_th}: {best['buys']} shares, IRR {pct_eff(best['irr_cont'])} "
                  f"per year", color=WHITE, fontsize=12, fontweight='bold')
    axW.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')

    style_ax(axR1)
    axR1.plot(acc_eval.index, rsi_eval, color=BLUE, lw=1.1, label=f"RSI({acc_rsi_days})")
    axR1.axhline(best_th, color=GREEN, ls='--', lw=1.0, label=f"Buy when RSI < {best_th}")
    axR1.scatter(buy_days, acc_eval.loc[buy_days, 'RSI'], color=GREEN, s=14, zorder=5, label="Purchase day")
    axR1.set_ylim(0, 100)
    axR1.set_ylabel("RSI", color=TEXT_COLOR, fontsize=9)
    axR1.set_title("RSI and Purchase Days", color=WHITE, fontsize=12, fontweight='bold')
    axR1.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=7, loc='upper left', ncol=3)

    style_ax(axS)
    axS.plot(acc_eval.index, acc_eval['Close'], color='#cbd5e1', lw=1.1, label="Close")
    axS.scatter(buy_days, acc_eval.loc[buy_days, 'Close'], marker='^', color=GREEN, s=26, zorder=5,
                label="Buy 1 share")
    axS.axhline(best['avg_price'], color=GOLD, ls='--', lw=1.0, label=f"Average cost {best['avg_price']:.2f}")
    axS.set_ylabel("Price", color=TEXT_COLOR, fontsize=9)
    axS.set_title(f"{TICKER} — Price and Purchases", color=WHITE, fontsize=12, fontweight='bold')
    axS.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')
    figA.patch.set_facecolor(DARK_BG)
    st.pyplot(figA)
    plt.close(figA)

    # ---- Comparison table ----
    st.markdown("#### 📊 Comparison")
    comp = pd.DataFrame([
        {"Plan": f"RSI < {best_th} accumulation (optimal)", "Shares": best['buys'],
         "Invested": round(best['invested']), "Value at last close": round(best['final_value']),
         "Annual return (continuous)": f"{best['irr_cont']*100:.2f}%", "Annual return (effective)": pct_eff(best['irr_cont'])},
        {"Plan": "Buy one share every day", "Shares": every_day['buys'],
         "Invested": round(every_day['invested']), "Value at last close": round(every_day['final_value']),
         "Annual return (continuous)": f"{ed_cont*100:.2f}%", "Annual return (effective)": pct_eff(ed_cont)},
        {"Plan": "Buy & hold (one share on the first day)", "Shares": 1,
         "Invested": round(C_eval[0] * cm, 2), "Value at last close": round(C_eval[-1], 2),
         "Annual return (continuous)": f"{bh_cont*100:.2f}%", "Annual return (effective)": pct_eff(bh_cont)},
        {"Plan": "Risk-free rate", "Shares": "—", "Invested": "—", "Value at last close": "—",
         "Annual return (continuous)": f"{rf_cont*100:.2f}%",
         "Annual return (effective)": f"{RISK_FREE_RATE_ANNUAL*100:.2f}%"},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # ---- All thresholds ----
    with st.expander("📋 IRR for every threshold"):
        gd = grid.copy()
        gd['IRR (effective)'] = ((np.exp(gd['irr_cont']) - 1) * 100).round(2)
        gd = gd[['threshold', 'buys', 'invested', 'final_value', 'profit', 'avg_price', 'IRR (effective)']].round(2)
        gd.columns = ['Buy when RSI <', 'Shares', 'Invested', 'Value', 'Profit', 'Avg cost', 'IRR % p.a.']
        gd['Valid'] = np.where(grid['buys'] >= acc_min_buys, "✓", "too few buys")
        st.dataframe(gd, use_container_width=True, hide_index=True)

    # ---- Current signal ----
    st.markdown("#### 📍 Today")
    cur_rsi = float(rsi_eval[-1])
    t1c, t2c, t3c = st.columns(3)
    t1c.metric("Last Close", f"{C_eval[-1]:.2f}")
    t2c.metric("Current RSI", f"{cur_rsi:.1f}")
    if cur_rsi < best_th:
        t3c.success(f"🟢 BUY one share today — RSI {cur_rsi:.1f} < {best_th}")
    else:
        t3c.info(f"⚪ No purchase today — RSI {cur_rsi:.1f} ≥ {best_th}")

    st.caption(
        "Note: the IRR measures the return per unit invested, not the size of the plan. Very low thresholds "
        "buy only a few shares near deep lows and can show a high IRR on a small amount; the right-hand chart "
        "above shows how many shares each threshold buys. The threshold is chosen on the same data it is "
        "evaluated on, so the optimal IRR is an in-sample figure.")


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
# TAB 3 — Heikin-Ashi strategy
# =====================================================
with tab3:
    st.subheader("Heikin-Ashi Strategy — Long While the HA Bar Is Green, Cash Otherwise")

    exec_ha = "open" if ha_execution.startswith("Next") else "close"
    st.info(
        "**Rule** — Heikin-Ashi bars: HA close = (O + H + L + C) / 4, HA open = average of the previous "
        "HA open and HA close. **Green bar (HA close ≥ HA open) → long; red bar → sell and hold cash.** "
        f"Cash earns the risk-free rate ({RISK_FREE_RATE_ANNUAL*100:.0f}% p.a.). 100% invested when long, "
        f"cost {TRANSACTION_COST*100:.1f}% per side, execution: **{ha_execution}**. "
        "No parameters are estimated, so the whole period is a single out-of-sample backtest."
    )

    ha = heikin_ashi(data)
    ha_sig = ha['Green'].astype(int).values
    ha_equity, ha_rets, ha_trades, ha_held = backtest_position(data, ha_sig, RISK_FREE_RATE_ANNUAL, exec_ha)

    bh_rets = data['Close'].pct_change().fillna(0)
    bh_equity = (1 + bh_rets).cumprod()
    daily_rf = (1 + RISK_FREE_RATE_ANNUAL) ** (1 / 252) - 1
    cash_rets = pd.Series(np.r_[0.0, np.full(len(data) - 1, daily_rf)], index=data.index)
    cash_equity = (1 + cash_rets).cumprod()

    # ---- Headline metrics ----
    ha_total = ha_equity.iloc[-1] - 1; bh_total = bh_equity.iloc[-1] - 1
    ha_mdd = (ha_equity / ha_equity.cummax() - 1).min(); bh_mdd = (bh_equity / bh_equity.cummax() - 1).min()
    n_closed = int((ha_trades['status'] == "Closed").sum()) if len(ha_trades) else 0
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Strategy Return", f"{ha_total*100:.1f}%", delta=f"{(ha_total-bh_total)*100:+.1f} pp vs B&H")
    m2.metric("Buy & Hold Return", f"{bh_total*100:.1f}%")
    m3.metric("Cash Return", f"{(cash_equity.iloc[-1]-1)*100:.1f}%")
    m4.metric("Strategy Max DD", f"{ha_mdd*100:.1f}%", delta=f"B&H {bh_mdd*100:.1f}%", delta_color="off")
    m5.metric("Closed Trades", n_closed)
    m6.metric("Time in Market", f"{ha_held.iloc[1:].mean()*100:.0f}%")

    # ---- Main chart: equity, price with HA colour ----
    fig7 = plt.figure(figsize=(16, 11), facecolor=DARK_BG)
    gs7 = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1.6], hspace=0.3)
    axA = fig7.add_subplot(gs7[0]); axB = fig7.add_subplot(gs7[1], sharex=axA)

    style_ax(axA)
    axA.plot(ha_equity.index, ha_equity.values * 10000, color=GOLD, lw=1.9, label="Heikin-Ashi strategy")
    axA.plot(bh_equity.index, bh_equity.values * 10000, '--', color='#94a3b8', lw=1.2, label="Buy & Hold")
    axA.plot(cash_equity.index, cash_equity.values * 10000, ':', color=CYAN, lw=1.2,
             label=f"Cash ({RISK_FREE_RATE_ANNUAL*100:.0f}% p.a.)")
    axA.axhline(10000, color=GRID_COLOR, lw=0.8)
    axA.set_ylabel("Portfolio Value (start 10,000)", color=TEXT_COLOR, fontsize=9)
    axA.set_title(f"Complete Backtest {data.index[0].date()} → {data.index[-1].date()}  |  "
                  f"HA {ha_total*100:+.1f}%  ·  Buy & Hold {bh_total*100:+.1f}%",
                  color=WHITE, fontsize=12, fontweight='bold')
    axA.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')
    axAd = axA.twinx()
    dd = (ha_equity / ha_equity.cummax() - 1) * 100
    axAd.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.13)
    axAd.set_ylim(-100, 0)
    axAd.tick_params(colors=TEXT_COLOR, labelsize=7)
    axAd.set_ylabel("Strategy drawdown %", fontsize=7, color=TEXT_COLOR)
    for spine in axAd.spines.values():
        spine.set_color(GRID_COLOR)

    style_ax(axB)
    axB.plot(data.index, data['Close'], color='#cbd5e1', lw=1.1, label="Close")
    lo, hi = float(data['Low'].min()), float(data['High'].max())
    axB.fill_between(data.index, lo, hi, where=ha['Green'].values, color=GREEN, alpha=0.10, step='post',
                     label="HA green (long)")
    axB.fill_between(data.index, lo, hi, where=~ha['Green'].values, color=RED, alpha=0.06, step='post',
                     label="HA red (cash)")
    if len(ha_trades):
        axB.scatter(ha_trades['entry_date'], ha_trades['entry_price'], marker='^', color=GREEN, s=28,
                    zorder=5, label="Buy")
        cl = ha_trades[ha_trades['status'] == "Closed"]
        axB.scatter(cl['exit_date'], cl['exit_price'], marker='v', color=RED, s=28, zorder=5, label="Sell")
    axB.set_ylim(lo * 0.98, hi * 1.02)
    axB.set_title(f"{TICKER} — Price with Heikin-Ashi Colour and Trades", color=WHITE, fontsize=12,
                  fontweight='bold')
    axB.set_ylabel("Price", color=TEXT_COLOR, fontsize=9)
    axB.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left', ncol=5)
    fig7.patch.set_facecolor(DARK_BG)
    st.pyplot(fig7)
    plt.close(fig7)

    # ---- Heikin-Ashi candles, recent period ----
    st.markdown(f"#### 🕯️ Heikin-Ashi Candles — Last {ha_zoom_days} Days")
    ha_z = ha.iloc[-ha_zoom_days:]
    fig8, ax8 = plt.subplots(figsize=(16, 5), facecolor=DARK_BG)
    style_ax(ax8, PANEL_BG2)
    draw_ha_candles(ax8, ha_z)
    ax8.plot(ha_z.index, data['Close'].iloc[-ha_zoom_days:], color=WHITE, lw=0.8, alpha=0.6, label="Actual close")
    ax8.set_ylabel("Price", color=TEXT_COLOR, fontsize=9)
    ax8.legend(facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')
    ax8.autoscale_view()
    fig8.patch.set_facecolor(DARK_BG)
    st.pyplot(fig8)
    plt.close(fig8)

    # ---- Current signal ----
    st.markdown("#### 📍 Current Signal")
    last_green = bool(ha['Green'].iloc[-1]); prev_green = bool(ha['Green'].iloc[-2])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Last Close", f"{float(data['Close'].iloc[-1]):.2f}")
    k2.metric("HA Open / Close", f"{ha['HA_Open'].iloc[-1]:.2f} / {ha['HA_Close'].iloc[-1]:.2f}")
    streak = int((ha['Green'] != ha['Green'].shift()).cumsum().value_counts().loc[
        (ha['Green'] != ha['Green'].shift()).cumsum().iloc[-1]])
    k3.metric("Bar colour", "🟢 Green" if last_green else "🔴 Red", delta=f"{streak} bar(s) in a row",
              delta_color="off")
    if last_green and not prev_green:
        k4.success("🟢 BUY — bar turned green today")
    elif not last_green and prev_green:
        k4.error("🔴 SELL — bar turned red today; move to cash")
    elif last_green:
        k4.info("🟢 HOLD LONG")
    else:
        k4.info("⚪ STAY IN CASH")

    # ---- Performance summary ----
    st.markdown("#### 📊 Performance Summary")
    cash_held = pd.Series(0.0, index=data.index)
    st.dataframe(pd.DataFrame([
        equity_stats(ha_equity, ha_rets, RISK_FREE_RATE_ANNUAL, ha_held, ha_trades, "Heikin-Ashi"),
        equity_stats(bh_equity, bh_rets, RISK_FREE_RATE_ANNUAL, pd.Series(1.0, index=data.index), None, "Buy & Hold"),
        equity_stats(cash_equity, cash_rets, RISK_FREE_RATE_ANNUAL, cash_held, None, "Cash (risk-free)"),
    ]).set_index("Strategy"), use_container_width=True)
    st.caption("Sharpe uses daily returns in excess of the risk-free rate (cash therefore has no Sharpe). "
               "Calmar = CAGR / |max drawdown|. A still-open position is valued at the last close and "
               "excluded from the trade statistics.")

    # ---- Calendar-year returns ----
    st.markdown("#### 📅 Calendar-Year Returns")
    yr = pd.DataFrame({
        "Heikin-Ashi": (1 + ha_rets).groupby(ha_rets.index.year).prod() - 1,
        "Buy & Hold": (1 + bh_rets).groupby(bh_rets.index.year).prod() - 1,
        "Cash": (1 + cash_rets).groupby(cash_rets.index.year).prod() - 1,
    })
    yr["HA − B&H"] = yr["Heikin-Ashi"] - yr["Buy & Hold"]
    st.dataframe((yr * 100).round(1).astype(str) + "%", use_container_width=True)

    # ---- Trade log ----
    with st.expander(f"📋 Trade Log ({len(ha_trades)} trades)"):
        if len(ha_trades):
            tl = ha_trades[['entry_date', 'exit_date', 'entry_price', 'exit_price', 'held_days',
                            'net_ret', 'status']].copy()
            tl['entry_price'] = tl['entry_price'].round(2); tl['exit_price'] = tl['exit_price'].round(2)
            tl['net_ret'] = (tl['net_ret'] * 100).round(2)
            tl.columns = ['Entry', 'Exit', 'Entry Px', 'Exit Px', 'Days', 'Net Ret %', 'Status']
            st.dataframe(tl, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download trade log (CSV)", tl.to_csv(index=False).encode("utf-8"),
                               file_name=f"{TICKER}_heikin_ashi_trades.csv", mime="text/csv")
        else:
            st.info("No trades generated.")

st.caption("Developed for educational and research purposes — RSI Accumulation + Long/Short + Heikin-Ashi Strategy Suite.")







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
