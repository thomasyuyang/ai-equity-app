
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Short-Term Trader", page_icon="💹", layout="centered")

# -----------------------------
# Mobile-friendly style
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 720px;
        padding-top: 0.7rem;
        padding-left: 0.6rem;
        padding-right: 0.6rem;
    }
    h1 {font-size: 1.45rem !important; margin-bottom: 0.2rem !important;}
    h2 {font-size: 1.2rem !important;}
    h3 {font-size: 1.05rem !important;}
    [data-testid="stMetric"] {
        background: rgba(250,250,250,0.98);
        border: 1px solid rgba(49,51,63,0.13);
        border-radius: 12px;
        padding: 0.45rem 0.55rem;
    }
    [data-testid="stMetricLabel"] {font-size: 0.72rem !important;}
    [data-testid="stMetricValue"] {font-size: 1.05rem !important;}
    [data-testid="stMetricDelta"] {font-size: 0.7rem !important;}
    div[data-testid="stExpander"] details summary p {font-size: 0.88rem !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 0.35rem;}
    .stTabs [data-baseweb="tab"] {padding: 0.35rem 0.4rem; font-size: 0.84rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_WATCHLIST = [
    "NVDA", "SMH", "AVGO", "AMD", "TSM", "MU", "MRVL", "ARM",
    "MSFT", "META", "GOOGL", "PLTR", "QQQM", "VGT", "VUG", "MO"
]

# -----------------------------
# Indicators
# -----------------------------

def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length).mean()


def pct_distance(price, level):
    if level == 0 or pd.isna(level):
        return np.nan
    return (price - level) / level * 100


@st.cache_data(ttl=1800)
def load_daily_data(ticker: str, period: str = "1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return None

    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    if len(df) < 80:
        return None

    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["ATR14"] = calculate_atr(df, 14)
    df["ATR14_%"] = df["ATR14"] / df["Close"] * 100

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd - signal

    df["Volume20Avg"] = df["Volume"].rolling(20).mean()
    df["High20"] = df["High"].rolling(20).max()
    df["Low20"] = df["Low"].rolling(20).min()

    ret = df["Close"].pct_change()
    df["Vol20_%"] = ret.rolling(20).std() * np.sqrt(252) * 100
    df["Vol60_%"] = ret.rolling(60).std() * np.sqrt(252) * 100

    return df.dropna()


@st.cache_data(ttl=120)
def load_intraday_snapshot(ticker: str):
    """Best practical short-term price: latest 1-minute close plus intraday VWAP.
    Falls back to None if yfinance is rate-limited or market data is unavailable.
    """
    try:
        intra = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=False)
    except Exception:
        return None

    if intra is None or intra.empty:
        return None
    if isinstance(intra.columns, pd.MultiIndex):
        intra.columns = intra.columns.get_level_values(0)

    intra = intra.dropna()
    if intra.empty or "Close" not in intra.columns:
        return None

    last_price = float(intra["Close"].iloc[-1])
    if "Volume" in intra.columns and intra["Volume"].sum() > 0:
        vwap = float((intra["Close"] * intra["Volume"]).sum() / intra["Volume"].sum())
    else:
        vwap = np.nan

    day_high = float(intra["High"].max()) if "High" in intra.columns else np.nan
    day_low = float(intra["Low"].min()) if "Low" in intra.columns else np.nan
    return {
        "Current_Price": round(last_price, 2),
        "VWAP": round(vwap, 2) if not np.isnan(vwap) else None,
        "Day_High": round(day_high, 2) if not np.isnan(day_high) else None,
        "Day_Low": round(day_low, 2) if not np.isnan(day_low) else None,
    }


@st.cache_data(ttl=1800)
def load_spy(period="1y"):
    return load_daily_data("SPY", period)


def links_for(ticker):
    symbol = quote(ticker.upper())
    return {
        "Yahoo": f"https://finance.yahoo.com/chart/{symbol}",
        "Fidelity": f"https://digital.fidelity.com/prgw/digital/research/quote/dashboard/summary?symbol={symbol}",
    }


def compute_relative_strength(df: pd.DataFrame, spy: pd.DataFrame | None):
    if spy is None or spy.empty:
        return np.nan
    try:
        stock_20 = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1
        spy_20 = spy["Close"].iloc[-1] / spy["Close"].iloc[-21] - 1
        return (stock_20 - spy_20) * 100
    except Exception:
        return np.nan


# -----------------------------
# Short-term trading engine
# -----------------------------

def trade_setup(ticker: str, capital: float, risk_pct: float, max_position_pct: float, target_mode: str):
    df = load_daily_data(ticker)
    if df is None:
        return None

    spy = load_spy()
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    daily_close = float(latest["Close"])
    snap = load_intraday_snapshot(ticker)
    current_price = float(snap["Current_Price"]) if snap else daily_close
    vwap = snap["VWAP"] if snap else None

    ema10 = float(latest["EMA10"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema200 = float(latest["EMA200"])
    rsi = float(latest["RSI14"])
    atr = float(latest["ATR14"])
    atr_pct = atr / current_price * 100 if current_price else float(latest["ATR14_%"])
    vol20 = float(latest["Vol20_%"])
    macd_hist = float(latest["MACD_Hist"])
    prev_macd = float(prev["MACD_Hist"])
    volume = float(latest["Volume"])
    volume_avg = float(latest["Volume20Avg"])
    volume_ratio = volume / volume_avg if volume_avg else 0
    high20_prev = float(df["High"].iloc[-21:-1].max())
    dist20 = pct_distance(current_price, ema20)
    dist50 = pct_distance(current_price, ema50)
    rs20 = compute_relative_strength(df, spy)
    price_vs_vwap = pct_distance(current_price, vwap) if vwap else np.nan

    # Trend score: direction and relative strength
    trend_score = 0
    if current_price > ema20: trend_score += 25
    if ema20 > ema50: trend_score += 25
    if ema50 > ema200: trend_score += 20
    if macd_hist > prev_macd: trend_score += 15
    if not np.isnan(rs20) and rs20 > 0: trend_score += 15

    # Entry score: buy pullbacks in uptrend, avoid chasing
    entry_score = 0
    if -3 <= dist20 <= 2: entry_score += 35
    elif -5 <= dist20 < -3 or 2 < dist20 <= 4: entry_score += 20
    elif dist20 > 8: entry_score -= 20

    if 40 <= rsi <= 60: entry_score += 25
    elif 60 < rsi <= 68: entry_score += 10
    elif rsi > 72: entry_score -= 20
    elif rsi < 35: entry_score -= 10

    if macd_hist > prev_macd: entry_score += 20
    if volume_ratio >= 1.1: entry_score += 10
    if current_price > high20_prev and volume_ratio >= 1.2: entry_score += 10

    # Intraday confirmation
    if vwap:
        if current_price >= vwap and -3 <= dist20 <= 4:
            entry_score += 8
        elif current_price < vwap and trend_score < 60:
            entry_score -= 8

    # Risk score: volatility, extension, weak trend
    risk_score = 0
    if atr_pct > 5: risk_score += 30
    elif atr_pct > 3.5: risk_score += 18
    elif atr_pct > 2.2: risk_score += 10

    if rsi > 70: risk_score += 25
    if dist20 > 8: risk_score += 25
    if current_price < ema50: risk_score += 20
    if not np.isnan(price_vs_vwap) and price_vs_vwap < -1.5: risk_score += 8
    risk_score = min(100, risk_score)

    trade_score = round(max(0, min(100, trend_score * 0.45 + entry_score * 0.42 - risk_score * 0.22 + 20)), 1)

    if trade_score >= 75 and risk_score <= 55 and trend_score >= 60:
        action = "🟢 READY"
        action_note = "Good candidate for staged entry"
    elif trade_score >= 60 and trend_score >= 55:
        action = "🟡 NEAR ENTRY"
        action_note = "Watch for pullback or confirmation"
    elif risk_score >= 70 or dist20 > 8:
        action = "🔴 WAIT"
        action_note = "Too risky or extended"
    else:
        action = "⚪ WATCH"
        action_note = "No clear edge yet"

    # Entry ladder uses real/current price + EMA/ATR
    aggressive_entry = round(min(current_price, max(ema20, current_price - 0.35 * atr)), 2)
    normal_entry = round(max(ema20, current_price - 0.75 * atr), 2)
    strong_entry = round(max(ema50, current_price - 1.25 * atr), 2)

    planned_entry = normal_entry
    stop = round(min(ema50, planned_entry - 1.25 * atr), 2)
    if stop >= planned_entry:
        stop = round(planned_entry - 1.25 * atr, 2)

    if target_mode == "Percent targets":
        t1 = round(planned_entry * 1.05, 2)
        t2 = round(planned_entry * 1.10, 2)
        t3 = round(planned_entry * 1.15, 2)
    else:
        t1 = round(planned_entry + 1.0 * atr, 2)
        t2 = round(planned_entry + 1.5 * atr, 2)
        t3 = round(planned_entry + 2.0 * atr, 2)

    risk_dollars_total = capital * (risk_pct / 100)
    max_position_dollars = capital * (max_position_pct / 100)
    risk_per_share = max(planned_entry - stop, 0.01)
    shares_by_risk = int(risk_dollars_total // risk_per_share)
    shares_by_size = int(max_position_dollars // planned_entry) if planned_entry > 0 else 0
    shares = max(0, min(shares_by_risk, shares_by_size))
    position_value = round(shares * planned_entry, 2)
    dollars_at_risk = round(shares * risk_per_share, 2)
    profit_t1 = round(shares * (t1 - planned_entry), 2)
    profit_t2 = round(shares * (t2 - planned_entry), 2)
    profit_t3 = round(shares * (t3 - planned_entry), 2)
    rr1 = round((t1 - planned_entry) / risk_per_share, 2)
    rr2 = round((t2 - planned_entry) / risk_per_share, 2)
    rr3 = round((t3 - planned_entry) / risk_per_share, 2)

    out = {
        "Ticker": ticker,
        "Action": action,
        "Action_Note": action_note,
        "Current_Price": round(current_price, 2),
        "Daily_Close": round(daily_close, 2),
        "VWAP": vwap,
        "Price_vs_VWAP_%": round(price_vs_vwap, 2) if not np.isnan(price_vs_vwap) else None,
        "Trade_Score": trade_score,
        "Trend_Score": round(trend_score, 1),
        "Entry_Score": round(entry_score, 1),
        "Risk_Score": round(risk_score, 1),
        "RSI": round(rsi, 1),
        "ATR": round(atr, 2),
        "ATR_%": round(atr_pct, 2),
        "Vol20_%": round(vol20, 1),
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "EMA200": round(ema200, 2),
        "Dist_EMA20_%": round(dist20, 2),
        "Dist_EMA50_%": round(dist50, 2),
        "Volume_Ratio": round(volume_ratio, 2),
        "RelStrength20_%": round(rs20, 2) if not np.isnan(rs20) else None,
        "Aggressive_Entry": aggressive_entry,
        "Normal_Entry": normal_entry,
        "Strong_Entry": strong_entry,
        "Planned_Entry": planned_entry,
        "Stop": stop,
        "Target_1": t1,
        "Target_2": t2,
        "Target_3": t3,
        "Shares": shares,
        "Position_Value": position_value,
        "Dollars_At_Risk": dollars_at_risk,
        "Profit_T1": profit_t1,
        "Profit_T2": profit_t2,
        "Profit_T3": profit_t3,
        "RR1": rr1,
        "RR2": rr2,
        "RR3": rr3,
    }
    out.update(links_for(ticker))
    return out


def label_score(kind, score):
    score = float(score)
    if kind == "trade":
        if score >= 75: return "🟢 High probability"
        if score >= 60: return "🟡 Near entry"
        if score >= 45: return "⚪ Watch"
        return "🔴 Weak"
    if kind == "risk":
        if score <= 35: return "🟢 Controlled"
        if score <= 55: return "🟡 Manageable"
        if score <= 70: return "🟠 High"
        return "🔴 Too high"
    return ""


def show_candidate(row, key_prefix="main"):
    with st.container(border=True):
        left, right = st.columns([1.35, 1])
        with left:
            st.markdown(f"### {row['Ticker']}  {row['Action']}")
            st.caption(row["Action_Note"])
        with right:
            st.metric("Trade Score", f"{row['Trade_Score']}/100", label_score("trade", row["Trade_Score"]))

        c1, c2 = st.columns(2)
        c1.metric("Current", f"${row['Current_Price']}")
        c2.metric("VWAP", "N/A" if row["VWAP"] is None else f"${row['VWAP']}", "" if row["Price_vs_VWAP_%"] is None else f"{row['Price_vs_VWAP_%']}%")

        c3, c4 = st.columns(2)
        c3.metric("Entry", f"${row['Planned_Entry']}")
        c4.metric("Stop", f"${row['Stop']}", f"Risk ${row['Dollars_At_Risk']}")

        t1, t2, t3 = st.columns(3)
        t1.metric("T1", f"${row['Target_1']}", f"${row['Profit_T1']}")
        t2.metric("T2", f"${row['Target_2']}", f"${row['Profit_T2']}")
        t3.metric("T3", f"${row['Target_3']}", f"${row['Profit_T3']}")

        s1, s2 = st.columns(2)
        s1.metric("Shares", int(row["Shares"]))
        s2.metric("Position", f"${row['Position_Value']}")

        st.markdown(
            f"**Quick read:** RSI **{row['RSI']}** · ATR **{row['ATR_%']}%** · EMA20 dist **{row['Dist_EMA20_%']}%** · RS vs SPY **{row['RelStrength20_%']}%**"
        )

        with st.expander("Entry ladder + details", expanded=False):
            st.markdown(f"""
**Staged entry**
- Aggressive: **${row['Aggressive_Entry']}**
- Normal: **${row['Normal_Entry']}**
- Strong: **${row['Strong_Entry']}**

**Risk / reward**
- T1 R/R: **{row['RR1']}**
- T2 R/R: **{row['RR2']}**
- T3 R/R: **{row['RR3']}**

**Setup data**
- Trend score: **{row['Trend_Score']}/100**
- Entry score: **{row['Entry_Score']}/100**
- Risk score: **{row['Risk_Score']}/100**
- Volume ratio: **{row['Volume_Ratio']}x**
- Vol20: **{row['Vol20_%']}%**
- Daily close source price: **${row['Daily_Close']}**
""")
            l1, l2 = st.columns(2)
            l1.link_button("Yahoo Chart", row["Yahoo"], use_container_width=True)
            l2.link_button("Fidelity", row["Fidelity"], use_container_width=True)

        with st.expander("Chart", expanded=False):
            hist = load_daily_data(row["Ticker"], "6mo")
            if hist is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA20"], mode="lines", name="EMA20"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA50"], mode="lines", name="EMA50"))
                fig.add_hline(y=row["Planned_Entry"], line_dash="dash", annotation_text="Entry")
                fig.add_hline(y=row["Stop"], line_dash="dash", annotation_text="Stop")
                fig.add_hline(y=row["Target_2"], line_dash="dash", annotation_text="Target 2")
                fig.update_layout(height=300, margin=dict(l=4, r=4, t=25, b=4), legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_prefix}_{row['Ticker']}")


# -----------------------------
# App UI
# -----------------------------

st.title("💹 AI Short-Term Trader")
st.caption("Short-term AI stock/ETF scanner · real-time price · VWAP · entry/target plan")
st.warning("Educational tool only. Not financial advice. Use limit orders and confirm prices with Fidelity/Yahoo.")

with st.expander("Trading settings", expanded=True):
    capital = st.number_input("Trading bucket", min_value=1000.0, max_value=1000000.0, value=10000.0, step=500.0, format="%.0f")
    risk_pct = st.slider("Max risk per trade", 0.25, 3.0, 1.0, 0.25)
    max_position_pct = st.slider("Max position size per ticker", 5, 60, 30, 5)
    target_mode = st.radio("Target method", ["ATR targets", "Percent targets"], horizontal=True)
    watchlist_text = st.text_area("Watchlist", value=", ".join(DEFAULT_WATCHLIST), height=95)

watchlist = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
scan = st.button("🔍 Scan for Trades", use_container_width=True)

if scan or "trade_results" not in st.session_state:
    rows = []
    progress = st.progress(0)
    for i, ticker in enumerate(watchlist):
        row = trade_setup(ticker, capital, risk_pct, max_position_pct, target_mode)
        if row:
            rows.append(row)
        progress.progress((i + 1) / max(len(watchlist), 1))
    st.session_state["trade_results"] = pd.DataFrame(rows)
    st.session_state["last_scan"] = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %I:%M:%S %p %Z")

results = st.session_state.get("trade_results", pd.DataFrame())

if results.empty:
    st.info("No data loaded yet. Tap Scan for Trades.")
else:
    st.caption(f"Last scan: {st.session_state.get('last_scan', 'N/A')}")
    results = results.sort_values("Trade_Score", ascending=False)

    m1, m2 = st.columns(2)
    m1.metric("Ready", len(results[results["Action"].str.contains("READY")]))
    m2.metric("Near Entry", len(results[results["Action"].str.contains("NEAR")]))

    tabs = st.tabs(["Best", "Ready", "Near", "All", "Rules"])

    with tabs[0]:
        st.subheader("Top trade candidates")
        for i, row in results.head(5).iterrows():
            show_candidate(row, key_prefix=f"best_{i}")

    with tabs[1]:
        ready = results[results["Action"].str.contains("READY")]
        if ready.empty:
            st.info("No ready trades now. Preserve cash and wait.")
        for i, row in ready.iterrows():
            show_candidate(row, key_prefix=f"ready_{i}")

    with tabs[2]:
        near = results[results["Action"].str.contains("NEAR")]
        if near.empty:
            st.info("No near-entry trades now.")
        for i, row in near.iterrows():
            show_candidate(row, key_prefix=f"near_{i}")

    with tabs[3]:
        st.dataframe(results[[
            "Ticker", "Action", "Trade_Score", "Current_Price", "VWAP",
            "Planned_Entry", "Stop", "Target_1", "Target_2", "Target_3",
            "Shares", "Position_Value", "Dollars_At_Risk", "RSI", "ATR_%",
            "Dist_EMA20_%", "RelStrength20_%"
        ]], use_container_width=True, hide_index=True)
        st.download_button(
            "Download trade plan CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="ai_short_term_trade_plan.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tabs[4]:
        st.markdown("""
### Rules for the $10,000 trading bucket

**Goal:** capture repeatable 5%–15% swings, not perfect tops or bottoms.

**Best setup:**
- Trade Score **75+**
- Trend Score **60+**
- Risk Score **below 55**
- Price near EMA20 or pulling back in an uptrend
- RSI roughly **40–60**
- Current price near or above VWAP for intraday confirmation

**Risk control:**
- Risk per trade around **1%** of your trading bucket.
- Use the stop before entering.
- Do not average down blindly.

**Selling rule:**
- Sell part at Target 1.
- Sell more at Target 2.
- Let only a smaller piece try for Target 3.
""")
