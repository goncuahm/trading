import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="RSI Strategy Optimizer", layout="wide")

# ------------------------------
# Title
# ------------------------------
st.title("📈 RSI Threshold Optimization and Signal Estimator")

# ------------------------------
# Sidebar Parameters
# ------------------------------
st.sidebar.header("🔧 User Parameters")

ticker = st.sidebar.text_input("Ticker Symbol (e.g., XU030.IS)", "XU030.IS")
period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=1)
tcost = st.sidebar.number_input("Transaction Cost (e.g., 0.002 = 0.4%)", value=0.0020, step=0.0005)
capital = st.sidebar.number_input("Initial Capital (TRY)", value=1_000_000, step=50_000)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 9)
search_low = 20 # st.sidebar.slider("Buy Threshold Range Start", 10, 45, 20)
search_high = 60 # st.sidebar.slider("Sell Threshold Range Start", 55, 90, 60)

# st.sidebar.write("Select the ranges where optimal RSI thresholds will be searched.")

# ------------------------------
# Download Data
# ------------------------------
st.subheader(f"📊 Downloading data for: {ticker}")
df = yf.download(ticker, period=period, auto_adjust=True)

if df.empty:
    st.error("⚠️ No data found for the selected ticker.")
    st.stop()

df = df.dropna()
df = df[["Open", "High", "Low", "Close", "Volume"]]

# ------------------------------
# RSI Calculation Function
# ------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI1"] = compute_rsi(df["Close"], rsi_period)
df = df.dropna()

# ------------------------------
# Backtest Function
# ------------------------------
def backtest_strategy(df, x1, x2):
    open_positions = []
    closed_trades = []

    for i in range(1, len(df)):
        rsi1 = df["RSI1"].iloc[i]
        price = df["Close"].iloc[i]
        date = df.index[i]

        if rsi1 < x1:
            open_positions.append({"entry_price": price, "entry_date": date})
        elif rsi1 > x2 and len(open_positions) > 0:
            entry = open_positions.pop(0)
            closed_trades.append({
                "buy_date": entry["entry_date"],
                "buy_price": entry["entry_price"],
                "sell_date": date,
                "sell_price": price,
                "return": (price - entry["entry_price"]) / entry["entry_price"] - tcost
            })

    total_return = np.sum([t["return"] for t in closed_trades])
    return total_return, closed_trades

# ------------------------------
# Optimization
# ------------------------------
progress_text = st.empty()
best_return = -np.inf
best_x1, best_x2, best_trades = None, None, []

for x1 in range(search_low, 45):
    for x2 in range(search_high, 90):
        total_return, trades = backtest_strategy(df, x1, x2)
        if total_return > best_return and len(trades) > 0:
            best_return = total_return
            best_x1, best_x2 = x1, x2
            best_trades = trades

st.success(f"✅ Optimal RSI thresholds found: Buy = {best_x1}, Sell = {best_x2}")
st.write(f"**Total Return:** {best_return:.2f} | **Number of Trades:** {len(best_trades)}")

# ------------------------------
# Plot Results
# ------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(df["Close"], label=f"{ticker} Close", color="black", lw=1.3)
if best_trades:
    for t in best_trades:
        ax1.scatter(t["buy_date"], t["buy_price"], color="green", marker="^", s=100)
        ax1.scatter(t["sell_date"], t["sell_price"], color="red", marker="v", s=100)

ax1.set_title(f"{ticker} RSI({rsi_period}) Strategy | x1={best_x1}, x2={best_x2}")
ax1.legend()
ax1.grid(True)

ax2.plot(df["RSI1"], color="blue", lw=1.2, label="RSI")
ax2.axhline(best_x1, color="green", linestyle="--", label=f"Buy ({best_x1})")
ax2.axhline(best_x2, color="red", linestyle="--", label=f"Sell ({best_x2})")
ax2.legend()
ax2.grid(True)

st.pyplot(fig)

# ------------------------------
# Latest Signal Estimation
# ------------------------------
latest_close = df["Close"].iloc[-1]
latest_rsi = df["RSI1"].iloc[-1]

if latest_rsi < best_x1:
    signal = "BUY"
elif latest_rsi > best_x2:
    signal = "SELL"
else:
    signal = "HOLD"

recent = df.tail(20).dropna(subset=["Close", "RSI1"])
x = recent["Close"].to_numpy()
y = recent["RSI1"].to_numpy()

if len(x) >= 2:
    a, b = np.polyfit(x, y, 1)
    buy_price_threshold = (best_x1 - b) / a if a != 0 else np.nan
    sell_price_threshold = (best_x2 - b) / a if a != 0 else np.nan
else:
    buy_price_threshold = sell_price_threshold = np.nan

per_trade_capital = capital / max(len(best_trades), 1)
order_size = int(per_trade_capital // latest_close)

# ------------------------------
# Display Summary
# ------------------------------
st.subheader("📋 Latest Signal Summary")

st.metric("Ticker", ticker)
st.metric("Latest Close", f"{latest_close:.2f}")
st.metric("Latest RSI", f"{latest_rsi:.2f}")
st.metric("Signal", signal)

if signal == "BUY":
    st.success(f"**BUY SIGNAL** → Target Price ≈ {buy_price_threshold:.2f} TRY | Suggested Order: {order_size} shares (~{order_size * latest_close:,.0f} TRY)")
elif signal == "SELL":
    st.error(f"**SELL SIGNAL** → Target Price ≈ {sell_price_threshold:.2f} TRY | Trigger if price > {sell_price_threshold:.2f}")
else:
    st.info(f"**HOLD** → RSI Buy Trigger ≈ {buy_price_threshold:.2f}, Sell Trigger ≈ {sell_price_threshold:.2f}")

st.caption("Developed for educational and research purposes.")
