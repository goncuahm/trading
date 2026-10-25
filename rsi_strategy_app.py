import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

st.set_page_config(page_title="RSI Accumulation (IRR)", layout="wide")

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.title("💰 RSI Accumulation (IRR)")

st.sidebar.subheader("Asset and period")
TICKER = st.sidebar.text_input("Ticker", value="ISDMR.IS")
START = st.sidebar.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
END = datetime.today().strftime('%Y-%m-%d')

RISK_FREE_RATE_ANNUAL = st.sidebar.number_input(
    "Annual Risk-Free Rate (e.g. 0.30 = 30%)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
TRANSACTION_COST = st.sidebar.number_input(
    "Transaction cost per purchase (e.g. 0.002 = 0.2%)", min_value=0.0, max_value=0.05,
    value=0.002, step=0.0005, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("Buy rule and threshold search")
acc_rsi_days = st.sidebar.slider("RSI Period", 5, 30, 14)
acc_th_range = st.sidebar.slider("RSI threshold search range (buy when RSI below)", 5, 95, (10, 70))
acc_min_per_year = st.sidebar.number_input(
    "Minimum purchases per year (on average)", min_value=0.0, max_value=252.0, value=1.0, step=0.5,
    help="A threshold is valid only if it buys at least this many times per year on average "
         "(total purchases ÷ years in the period). 1 = at least one purchase a year on average.")
acc_min_buys = st.sidebar.slider("Minimum total number of purchases", 1, 100, 1)
acc_include_cost = st.sidebar.checkbox("Include transaction cost on each purchase", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("💼 Savings plan")
monthly_budget = st.sidebar.number_input(
    "Monthly budget for this asset", min_value=0.0, value=10000.0, step=500.0,
    help="Annual budget = 12 × monthly. Amount per purchase day = annual budget ÷ the historical "
         "number of purchase days per year at the chosen threshold.")


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


def budget_plan_backtest(df, buy_idx, monthly_budget, amount_per_purchase, rf_annual, include_cost=True):
    """
    Replay the plan with a real budget.
      * On the first trading day of each month the monthly budget is deposited into a cash account
        that earns the risk-free rate (it can go negative if purchases run ahead of the budget,
        i.e. the plan borrows at the risk-free rate).
      * On each signal day, buy floor(amount_per_purchase / price) whole shares from that account.
    Benchmarks with the same deposits:
      * Monthly buying: each month, buy as many whole shares as the cash account allows.
      * Risk-free saving: deposits only earn the risk-free rate.
    Returns (daily DataFrame, purchase log DataFrame, deposit dates/amounts).
    """
    C = df['Close'].values; dates = df.index; n = len(df)
    cm = (1 + TRANSACTION_COST) if include_cost else 1.0
    daily_rf = (1 + rf_annual) ** (1 / 252) - 1
    month_start = np.r_[True, dates.month[1:] != dates.month[:-1]]
    is_buy = np.zeros(n, dtype=bool); is_buy[buy_idx] = True

    sh = 0; cash = 0.0; dca_sh = 0; dca_cash = 0.0; rf_acc = 0.0; contrib = 0.0; invested = 0.0
    rows, log = [], []
    for t in range(n):
        if t > 0:
            cash *= 1 + daily_rf; dca_cash *= 1 + daily_rf; rf_acc *= 1 + daily_rf
        if month_start[t]:
            cash += monthly_budget; dca_cash += monthly_budget; rf_acc += monthly_budget
            contrib += monthly_budget
            k = int(dca_cash // (C[t] * cm))                 # monthly buying benchmark
            dca_sh += k; dca_cash -= k * C[t] * cm
        if is_buy[t]:
            k = int(amount_per_purchase // (C[t] * cm))
            if k > 0:
                spend = k * C[t] * cm
                sh += k; cash -= spend; invested += spend
                log.append({"Date": dates[t].date(), "Price": C[t], "Shares": k, "Amount": spend})
        rows.append((contrib, invested, sh, sh * C[t], cash, sh * C[t] + cash,
                     dca_sh * C[t] + dca_cash, rf_acc))
    out = pd.DataFrame(rows, index=dates, columns=[
        "Contributed", "Invested", "Shares", "Share value", "Cash account", "Total wealth",
        "Monthly buying wealth", "Risk-free saving"])
    dep_idx = np.where(month_start)[0]
    return out, pd.DataFrame(log), dep_idx


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

st.title("💰 RSI Accumulation Strategy — Internal Rate of Return")

data = download_stock_data(TICKER, str(START), END)

if data is None:
    st.error("No data found for this ticker.")
    st.stop()

st.success(f"Downloaded **{len(data)}** trading days for **{TICKER}**  |  {str(START)} → {END}")


def render_accumulation_page():
    """Main page; returns early when no threshold qualifies."""
    st.subheader("RSI Accumulation — Buy One Share Every Day the RSI Is Below the Threshold, Never Sell")

    rf_cont = np.log(1 + RISK_FREE_RATE_ANNUAL)
    st.info(
        f"**Rule** — RSI period **{acc_rsi_days}**. On **every** day the RSI closes below the threshold, buy "
        "**one share** at the close and keep it for life (no sales).\n\n"
        "**Internal rate of return** — with purchases on dates t₁ … tₙ and N shares in total, r solves  "
        "**Σ S(tᵢ)·e^{r(T − tᵢ)} = S(T)·N**, where T is the last date and time is in years. "
        f"Purchases {'include' if acc_include_cost else 'exclude'} the {TRANSACTION_COST*100:.1f}% cost; "
        "the shares are valued at the last close.\n\n"
        f"The threshold is chosen from {acc_th_range[0]} to {acc_th_range[1]} to **maximise r**, subject to "
        f"at least **{acc_min_per_year:g} purchase(s) per year on average** and at least {acc_min_buys} "
        "purchase(s) in total. Benchmarks: **buy & hold** (one share bought on the first day), "
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
    years_eval = max((acc_eval.index[-1] - acc_eval.index[0]).days / 365.25, 1e-9)
    grid['buys_per_year'] = grid['buys'] / years_eval
    grid['valid'] = ((grid['buys'] >= acc_min_buys) & (grid['buys_per_year'] >= acc_min_per_year)
                     & grid['irr_cont'].notna())
    valid = grid[grid['valid']]
    if valid.empty:
        st.warning(f"No threshold buys at least {acc_min_per_year:g} share(s) per year on average "
                   f"(and {acc_min_buys} in total). Lower the minimums or widen the threshold range.")
        return
    min_valid_th = int(valid['threshold'].min())
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
    a6.metric("Shares accumulated", f"{best['buys']:,}",
              delta=f"{best['buys'] / years_eval:.1f} per year (min {acc_min_per_year:g})", delta_color="off")
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
    ok = grid['valid']
    axT1.plot(grid['threshold'], (np.exp(grid['irr_cont']) - 1) * 100, color=GOLD, lw=1.0, alpha=0.4)
    axT1.plot(grid.loc[ok, 'threshold'], (np.exp(grid.loc[ok, 'irr_cont']) - 1) * 100, 'o-', color=GOLD,
              lw=1.8, ms=3, label="IRR of RSI accumulation")
    axT1.axhline((np.exp(bh_cont) - 1) * 100, color='#94a3b8', ls='--', lw=1.2, label="Buy & Hold")
    axT1.axhline((np.exp(ed_cont) - 1) * 100, color=CYAN, ls='-.', lw=1.2, label="Buy every day")
    axT1.axhline(RISK_FREE_RATE_ANNUAL * 100, color=GREEN, ls=':', lw=1.4, label="Risk-free")
    axT1.axvline(best_th, color=WHITE, ls=':', lw=1)
    axT1.axvspan(grid['threshold'].min() - 0.5, min_valid_th - 0.5, color=RED, alpha=0.08,
                 label=f"Too few purchases (< {acc_min_per_year:g}/yr)")
    axT1.scatter([best_th], [(np.exp(best['irr_cont']) - 1) * 100], marker='*', s=220, color=WHITE,
                 edgecolor='black', zorder=5, label=f"Optimum: RSI < {best_th}")
    axT1.set_xlabel("RSI threshold (buy when RSI below)", color=TEXT_COLOR, fontsize=9)
    axT1.set_ylabel("Annual return (%)", color=TEXT_COLOR, fontsize=9)
    axT1.set_title("Internal Rate of Return vs RSI Threshold", color=WHITE, fontsize=11, fontweight='bold')
    axT1.legend(facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8)

    style_ax(axT2, PANEL_BG2)
    axT2.bar(grid['threshold'], grid['buys_per_year'], color=np.where(grid['valid'], BLUE, '#475569'),
             alpha=0.7, label="Purchases per year")
    axT2.axhline(acc_min_per_year, color=RED, ls='--', lw=1.0, label=f"Minimum {acc_min_per_year:g} per year")
    axT2.set_xlabel("RSI threshold", color=TEXT_COLOR, fontsize=9)
    axT2.set_ylabel("Purchases per year (average)", color=TEXT_COLOR, fontsize=9)
    axT2b = axT2.twinx()
    axT2b.plot(grid['threshold'], grid['profit'], color=ORANGE, lw=1.8, label="Profit")
    axT2b.tick_params(colors=TEXT_COLOR, labelsize=7)
    axT2b.set_ylabel("Profit at last close", color=ORANGE, fontsize=8)
    axT2.axvline(best_th, color=WHITE, ls=':', lw=1)
    h1, l1 = axT2.get_legend_handles_labels(); h2, l2 = axT2b.get_legend_handles_labels()
    axT2.legend(h1 + h2, l1 + l2, facecolor=PANEL_BG2, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8,
                loc='upper left')
    axT2.set_title("Size of the Plan: Purchases per Year and Profit", color=WHITE, fontsize=11, fontweight='bold')
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
    st.dataframe(comp.astype(str), use_container_width=True, hide_index=True)

    # ---- Budget plan: how many shares to buy, and what the plan would be worth now ----
    st.markdown("---")
    st.markdown("### 💼 Your Savings Plan")
    annual_budget = 12 * monthly_budget
    hist_per_year = best['buys'] / years_eval
    amount_per_purchase = annual_budget / hist_per_year if hist_per_year > 0 else 0.0
    st.markdown(
            f"**Monthly budget:** {monthly_budget:,.0f} (set in the sidebar)  \n"
            f"**Annual budget:** {annual_budget:,.0f}  \n"
            f"**Signal:** RSI({acc_rsi_days}) < **{best_th}** — historically **{hist_per_year:.1f}** "
            f"purchase days per year (your minimum: {acc_min_per_year:g})  \n"
            f"**Amount per purchase day** = annual budget ÷ purchases per year = **{amount_per_purchase:,.0f}**")

    last_px = float(C_eval[-1]); cmx = (1 + TRANSACTION_COST) if acc_include_cost else 1.0
    n_today = int(amount_per_purchase // (last_px * cmx)) if last_px > 0 else 0
    s1, s2, s3 = st.columns(3)
    s1.metric("Current RSI", f"{float(rsi_eval[-1]):.1f}", delta=f"threshold {best_th}", delta_color="off")
    s2.metric("Last Close", f"{last_px:.2f}")
    if float(rsi_eval[-1]) < best_th:
        if n_today > 0:
            s3.success(f"🟢 BUY {n_today:,} shares today ≈ {n_today * last_px * cmx:,.0f}")
        else:
            s3.warning(f"🟢 Signal today, but {amount_per_purchase:,.0f} buys less than one share at {last_px:.2f}")
    else:
        s3.info(f"⚪ No purchase today — wait for RSI < {best_th}. "
                f"On the next signal buy ≈ {n_today:,} shares (at today's price).")

    plan, plan_log, dep_idx = budget_plan_backtest(
        acc_eval, best['idx'], monthly_budget, amount_per_purchase, RISK_FREE_RATE_ANNUAL, acc_include_cost)

    if monthly_budget > 0:
        # Money-weighted annual return of each plan on the monthly deposits
        T_last = acc_eval.index[-1]
        dep_tau = (T_last - acc_eval.index[dep_idx]).days.values / 365.25
        dep_amt = np.full(len(dep_idx), monthly_budget)
        fin = plan.iloc[-1]
        irr_plan = solve_irr(dep_amt, dep_tau, fin["Total wealth"])
        irr_dca = solve_irr(dep_amt, dep_tau, fin["Monthly buying wealth"])

        st.markdown(f"#### 📈 If You Had Followed This Plan Since {acc_eval.index[0].date()}")
        w1, w2, w3, w4, w5 = st.columns(5)
        w1.metric("Total deposited", f"{fin['Contributed']:,.0f}")
        w2.metric("Invested in shares", f"{fin['Invested']:,.0f}",
                  delta=f"{int(fin['Shares']):,} shares", delta_color="off")
        w3.metric("Wealth now (shares + cash)", f"{fin['Total wealth']:,.0f}",
                  delta=f"annual return {pct_eff(irr_plan)}", delta_color="off")
        w4.metric("Same deposits, buying monthly", f"{fin['Monthly buying wealth']:,.0f}",
                  delta=f"annual return {pct_eff(irr_dca)}", delta_color="off")
        w5.metric("Same deposits at risk-free", f"{fin['Risk-free saving']:,.0f}",
                  delta=f"{RISK_FREE_RATE_ANNUAL*100:.1f}% p.a.", delta_color="off")
        st.caption(
            "Deposits are made on the first trading day of each month into a cash account earning the "
            "risk-free rate; each signal day buys whole shares from it. A negative cash account means the "
            "plan bought ahead of its budget (borrowing at the risk-free rate). 'Annual return' is the "
            "money-weighted return on the monthly deposits.")

        figB, axB1 = plt.subplots(figsize=(16, 5.5), facecolor=DARK_BG)
        style_ax(axB1)
        axB1.plot(plan.index, plan["Total wealth"], color=GOLD, lw=2.0, label=f"RSI < {best_th} plan: shares + cash")
        axB1.plot(plan.index, plan["Share value"], color=GOLD, lw=1.0, ls='--', alpha=0.7, label="…of which shares")
        axB1.plot(plan.index, plan["Monthly buying wealth"], color=CYAN, lw=1.4, label="Same deposits, buying monthly")
        axB1.plot(plan.index, plan["Risk-free saving"], color=GREEN, lw=1.3, ls=':', label="Same deposits at risk-free")
        axB1.plot(plan.index, plan["Contributed"], color='#94a3b8', lw=1.1, ls='--', label="Total deposited")
        if len(plan_log):
            ld = pd.to_datetime(plan_log['Date'])
            axB1.scatter(ld, plan.loc[ld, "Total wealth"], color=GREEN, s=16, zorder=5, label="Purchase day")
        axB1.set_ylabel("Value", color=TEXT_COLOR, fontsize=9)
        axB1.set_title(f"Savings Plan with {monthly_budget:,.0f} per Month — Wealth Over Time",
                       color=WHITE, fontsize=12, fontweight='bold')
        axB1.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=WHITE, fontsize=8, loc='upper left')
        figB.patch.set_facecolor(DARK_BG)
        st.pyplot(figB)
        plt.close(figB)

        # Year-by-year table
        st.markdown("#### 📅 Year by Year")
        yrs = plan.index.year
        ye = plan.groupby(yrs).last()
        first = plan.groupby(yrs).first()
        prev_contrib = ye["Contributed"].shift(1).fillna(0); prev_inv = ye["Invested"].shift(1).fillna(0)
        if len(plan_log):
            py = pd.to_datetime(plan_log['Date']).dt.year
            buys_y = plan_log.groupby(py)['Shares'].agg(['count', 'sum'])
        else:
            buys_y = pd.DataFrame(columns=['count', 'sum'])
        ytab = pd.DataFrame({
            "Deposited": (ye["Contributed"] - prev_contrib).round(0),
            "Purchase days": buys_y['count'].reindex(ye.index).fillna(0).astype(int),
            "Shares bought": buys_y['sum'].reindex(ye.index).fillna(0).astype(int),
            "Invested": (ye["Invested"] - prev_inv).round(0),
            "Shares held (year end)": ye["Shares"].astype(int),
            "Share value": ye["Share value"].round(0),
            "Cash account": ye["Cash account"].round(0),
            "Total wealth": ye["Total wealth"].round(0),
            "Monthly-buying wealth": ye["Monthly buying wealth"].round(0),
            "Risk-free wealth": ye["Risk-free saving"].round(0),
        })
        ytab.index.name = "Year"
        st.dataframe(ytab, use_container_width=True)
        avg_inv_year = fin["Invested"] / years_eval
        st.caption(f"On average {avg_inv_year:,.0f} invested per year against an annual budget of "
                   f"{annual_budget:,.0f}. Years with many signals invest more than the budget, quiet years "
                   "less; the difference sits in the cash account. The current year is partial.")

        with st.expander(f"📋 Purchase log ({len(plan_log)} purchase days)"):
            if len(plan_log):
                pl = plan_log.copy(); pl['Price'] = pl['Price'].round(2); pl['Amount'] = pl['Amount'].round(0)
                st.dataframe(pl, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download purchase log (CSV)", pl.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{TICKER}_rsi_savings_purchases.csv", mime="text/csv")
            else:
                st.info("The amount per purchase was too small to buy a single share on any signal day.")

    # ---- All thresholds ----
    with st.expander("📋 IRR for every threshold"):
        gd = grid.copy()
        gd['IRR (effective)'] = ((np.exp(gd['irr_cont']) - 1) * 100).round(2)
        gd = gd[['threshold', 'buys', 'buys_per_year', 'invested', 'final_value', 'profit', 'avg_price',
                 'IRR (effective)']].round(2)
        gd.columns = ['Buy when RSI <', 'Shares', 'Per year', 'Invested', 'Value', 'Profit', 'Avg cost',
                      'IRR % p.a.']
        gd['Valid'] = np.where(grid['valid'], "✓", "too few purchases")
        st.dataframe(gd, use_container_width=True, hide_index=True)

    st.caption(
        "Note: the IRR measures the return per unit invested, not the size of the plan. Very low thresholds "
        "buy only a few shares near deep lows and can show a high IRR on a small amount; the minimum "
        "purchases per year rules these out, and the right-hand chart above shows how often each threshold "
        "buys. The per-year minimum is an average over the whole period: purchases can still cluster in a "
        "few years. The threshold is chosen on the same data it is "
        "evaluated on, so the optimal IRR is an in-sample figure.")


render_accumulation_page()
