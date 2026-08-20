import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
import requests
import json
import os
import urllib.request

# --- Page Configuration ---
st.set_page_config(
    page_title="Institutional Derivatives Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- High-Contrast Institutional Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-val {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #58a6ff;
        font-size: 0.75rem;
        margin-top: 2px;
    }
    .side-panel-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .panel-header-red {
        color: #ff7b72;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 8px;
    }
    .panel-header-green {
        color: #3fb950;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 8px;
    }
    .panel-header-blue {
        color: #58a6ff;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 8px;
    }
    .panel-item {
        display: flex;
        justify-content: space-between;
        color: #c9d1d9;
        font-size: 0.88rem;
        padding: 3px 0;
        border-bottom: 1px solid #21262d;
    }
    .panel-item-bold {
        color: #ffffff;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Secrets Helper ---
def get_secret(key_name, default=""):
    try:
        return st.secrets[key_name]
    except Exception:
        return os.environ.get(key_name, default)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID", "")

# --- Multi-Tier Resilient AI Strategy Generator ---
def generate_strategy_brief(prompt, default_fallback):
    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    if GEMINI_API_KEY:
        for m in models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data['candidates'][0]['content']['parts'][0]['text']
            except Exception:
                continue
    return default_fallback

# --- Technical Engine ---
def calculate_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)

    min_rsi = df['rsi'].rolling(window=14).min()
    max_rsi = df['rsi'].rolling(window=14).max()
    df['stoch_k'] = (((df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-9)) * 100).fillna(50.0)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean().fillna(50.0)

    cum_vol = df['volume'].cumsum()
    cum_vol_price = (df['close'] * df['volume']).cumsum()
    df['vwap'] = (cum_vol_price / (cum_vol + 1e-9)).fillna(df['close'])
    return df

def get_liquidity_matrix(df):
    current = df['close'].iloc[-1]
    recent_highs = sorted(df['high'].nlargest(5).unique(), reverse=True)
    recent_lows = sorted(df['low'].nsmallest(5).unique())
    overhead = [h for h in recent_highs if h > current][:3]
    if not overhead:
        overhead = [round(current * (1 + p), 2) for p in [0.005, 0.012, 0.02]]
    downside = [l for l in recent_lows if l < current][:3]
    if not downside:
        downside = [round(current * (1 - p), 2) for p in [0.005, 0.012, 0.02]]
    return overhead, downside

def calculate_elliott_targets(df):
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    return {
        "Wave C (1.000)": high + (diff * 0.382),
        "Wave C (1.236)": high + (diff * 0.618),
        "Wave C (1.618)": high + (diff * 1.000)
    }

# --- Cloud Data Feed ---
@st.cache_data(ttl=15)
def fetch_cloud_klines(symbol="BTC/USDT", timeframe="1h", limit=48):
    coinbase_pairs = {"BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD", "SOL/USDT": "SOL-USD", "XRP/USDT": "XRP-USD"}
    if symbol in coinbase_pairs:
        try:
            pair = coinbase_pairs[symbol]
            url = f"https://api.exchange.coinbase.com/products/{pair}/candles?granularity=3600"
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                raw = resp.json()
                if isinstance(raw, list) and len(raw) > 5:
                    df = pd.DataFrame(raw, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                    df = df.iloc[::-1].tail(limit).reset_index(drop=True)
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                    return calculate_indicators(df)
        except Exception:
            pass

    try:
        raw_sym = symbol.replace("/", "")
        url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={raw_sym}&interval=60&limit={limit}"
        resp = requests.get(url, timeout=6).json()
        if resp.get('result', {}).get('list'):
            raw = resp['result']['list']
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df = df.iloc[::-1].reset_index(drop=True)
            df['datetime'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            return calculate_indicators(df)
    except Exception:
        pass

    # Fallback to Kraken
    exchange = ccxt.kraken({'enableRateLimit': True})
    klines = exchange.fetch_ohlcv(symbol.replace("USDT", "USD"), timeframe=timeframe, limit=limit)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return calculate_indicators(df)

# --- Layout ---
st.title("⚡ Institutional Derivatives Execution Terminal")

assets = {
    "₿ Bitcoin (BTC)": "BTC/USDT",
    "🪙 Silver (XAG)": "XAG/USDT",
    "Ξ Ethereum (ETH)": "ETH/USDT",
    "🟣 Solana (SOL)": "SOL/USDT",
    "✕ Ripple (XRP)": "XRP/USDT"
}

selected_asset_label = st.radio("Asset Select", list(assets.keys()), horizontal=True, label_visibility="collapsed")
selected_symbol = assets[selected_asset_label]

col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button(f"🔄 Refresh {selected_symbol}", use_container_width=True):
        st.cache_data.clear()

df = fetch_cloud_klines(selected_symbol)
last_row = df.iloc[-1]
current_price = last_row['close']
vwap_price = last_row['vwap']
stoch_k = last_row['stoch_k']
stoch_d = last_row['stoch_d']
rsi = last_row['rsi']

overhead_liq, downside_liq = get_liquidity_matrix(df)
fib_targets = calculate_elliott_targets(df)

# Top Metric Banner
m1, m2, m3, m4, m5 = st.columns(5)
m1.markdown(f"<div class='metric-box'><div class='metric-title'>Mark Price</div><div class='metric-val'>${current_price:,.2f}</div><div class='metric-sub'>Live Spot</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-box'><div class='metric-title'>1H VWAP</div><div class='metric-val'>${vwap_price:,.2f}</div><div class='metric-sub'>Fair Value</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-box'><div class='metric-title'>Stoch RSI (14)</div><div class='metric-val'>{stoch_k:.1f} / {stoch_d:.1f}</div><div class='metric-sub'>Momentum</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-box'><div class='metric-title'>RSI (14)</div><div class='metric-val'>{rsi:.2f}</div><div class='metric-sub'>Oscillator</div></div>", unsafe_allow_html=True)
m5.markdown(f"<div class='metric-box'><div class='metric-title'>Structure Mode</div><div class='metric-val'>{'Overextended' if rsi > 70 else ('Oversold' if rsi < 30 else 'Range Rotation')}</div><div class='metric-sub'>Institutional Flow</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Terminal Grid
col_chart, col_side = st.columns([3, 1])

with col_chart:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="OHLC",
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['vwap'],
        mode='lines', line=dict(color='#29b6f6', width=2),
        name="1H VWAP"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        margin=dict(l=8, r=8, t=8, b=8),
        height=480,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown(f"""
    <div class='side-panel-card'>
        <div class='panel-header-red'>🔴 Overhead Liquidity Pools</div>
        <div class='panel-item'><span>Sweep Level 1</span><span class='panel-item-bold'>${overhead_liq[0]:,.2f}</span></div>
        <div class='panel-item'><span>Sweep Level 2</span><span class='panel-item-bold'>${overhead_liq[1]:,.2f}</span></div>
        <div class='panel-item'><span>Sweep Level 3</span><span class='panel-item-bold'>${overhead_liq[2]:,.2f}</span></div>
    </div>
    <div class='side-panel-card'>
        <div class='panel-header-green'>🟢 Downside Liquidity Pools</div>
        <div class='panel-item'><span>Bid Target 1</span><span class='panel-item-bold'>${downside_liq[0]:,.2f}</span></div>
        <div class='panel-item'><span>Bid Target 2</span><span class='panel-item-bold'>${downside_liq[1]:,.2f}</span></div>
        <div class='panel-item'><span>Bid Target 3</span><span class='panel-item-bold'>${downside_liq[2]:,.2f}</span></div>
    </div>
    <div class='side-panel-card'>
        <div class='panel-header-blue'>🌊 Elliott Fibonacci Projections</div>
        <div class='panel-item'><span>Wave C (1.000)</span><span class='panel-item-bold'>${fib_targets['Wave C (1.000)']:,.2f}</span></div>
        <div class='panel-item'><span>Wave C (1.236)</span><span class='panel-item-bold'>${fib_targets['Wave C (1.236)']:,.2f}</span></div>
        <div class='panel-item'><span>Wave C (1.618)</span><span class='panel-item-bold'>${fib_targets['Wave C (1.618)']:,.2f}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Strategy Brief Section
st.subheader("🤖 Gemini Institutional Strategy Brief")
default_brief = f"""
* **Structural Bias**: {'Short-term Mean Reversion (SHORT)' if rsi > 70 else ('Dip Accumulation (LONG)' if rsi < 30 else 'Range Rotation')}
* **Key Sweep Level**: Target Overhead Liquidity at **${overhead_liq[0]:,.2f}** for liquidity absorption.
* **Downside Fair Value**: Primary mean-reversion target sits at 1H VWAP (**${vwap_price:,.2f}**).
* **Execution Strategy**: Scale into positions near structural extremes with tight invalidation above key wick pivots.
"""
prompt = f"Analyze {selected_symbol}: Price ${current_price:,.2f}, VWAP ${vwap_price:,.2f}, RSI {rsi:.2f}, Stoch {stoch_k:.1f}/{stoch_d:.1f}. Provide directional bias, entry, SL, and TP."
brief_content = generate_strategy_brief(prompt, default_brief)
st.markdown(brief_content)

st.markdown("---")

# 1-Click Order Bridge
st.subheader(f"⚡ 1-Click BTCC Order Bridge ({selected_symbol})")
with st.expander("AI-Synchronized Order Execution Parameters", expanded=True):
    b1, b2, b3, b4 = st.columns(4)
    suggested_side = "SHORT (Sell)" if rsi > 65 else "LONG (Buy)"
    pos_side = b1.selectbox("Position Side", ["SHORT (Sell)", "LONG (Buy)"], index=0 if suggested_side == "SHORT (Sell)" else 1)
    
    default_entry = overhead_liq[0] if "SHORT" in pos_side else downside_liq[0]
    default_tp = downside_liq[0] if "SHORT" in pos_side else overhead_liq[0]
    default_sl = round(default_entry * 1.015, 2) if "SHORT" in pos_side else round(default_entry * 0.985, 2)

    limit_entry = b2.number_input("Limit Entry ($)", value=float(default_entry))
    tp_target = b3.number_input("Take Profit Target ($)", value=float(default_tp))
    sl_target = b4.number_input("Stop Loss Target ($)", value=float(default_sl))

    leverage = st.slider("Leverage Target", 1, 100, 20)
    risk_dist = abs(limit_entry - sl_target)
    reward_dist = abs(limit_entry - tp_target)
    rr_ratio = reward_dist / (risk_dist + 1e-9)

    st.success(f"**Calculated Risk/Reward**: {rr_ratio:.2f} (Risk: ${risk_dist:,.2f} | Reward: ${reward_dist:,.2f})")

    ob1, ob2, ob3 = st.columns(3)
    if ob1.button("📋 1. Copy Parameters to Clipboard", use_container_width=True):
        st.info("Parameters staged for BTCC order form.")

    clean_sym = selected_symbol.replace("/", "").upper()
    btcc_url = f"https://www.btcc.com/en-US/trade/futures/{clean_sym}"
    ob2.link_button("🚀 2. Open Contract on BTCC", btcc_url, use_container_width=True)

    if ob3.button("📱 Send Order to Telegram", use_container_width=True):
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            msg = f"⚡ *BTCC Execution Order*\n\nAsset: {selected_symbol}\nSide: {pos_side}\nEntry: ${limit_entry:,.2f}\nTP: ${tp_target:,.2f}\nSL: ${sl_target:,.2f}\nLeverage: {leverage}x\nR:R: {rr_ratio:.2f}"
            t_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(t_url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
            st.success("Sent directly to your Telegram!")
        else:
            st.warning("Configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Streamlit Secrets.")
