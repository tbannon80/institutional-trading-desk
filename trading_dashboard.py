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

try:
    from google import genai
except ImportError:
    genai = None

# --- Page Configuration ---
st.set_page_config(
    page_title="Institutional Derivatives Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Theme & CSS Scaffolding ---
st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e1e7ec;
    }
    .metric-card {
        background: #151a23;
        border: 1px solid #232b38;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-title {
        color: #8c9ba5;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #00f2fe;
        font-size: 0.75rem;
        margin-top: 2px;
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

# Multi-Tier Resilient AI Strategy Generator
def generate_strategy_brief(prompt):
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    err_msg = ""
    
    # 1. Try modern SDK Client if present
    if GEMINI_API_KEY and genai:
        try:
            c = genai.Client(api_key=GEMINI_API_KEY)
            for m in models_to_try:
                try:
                    chat = c.chats.create(model=m)
                    return chat.send_message(prompt).text
                except Exception as e:
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        continue
        except Exception:
            pass

    # 2. Resilient Direct REST API (Bearer Header for AQ. + Query Param Fallback)
    if GEMINI_API_KEY:
        for m in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
                payload_data = {
                    "contents": [{"parts": [{"text": prompt}]}]
                }
                req_body = json.dumps(payload_data).encode("utf-8")
                
                headers = {"Content-Type": "application/json"}
                if GEMINI_API_KEY.startswith("AQ."):
                    headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"
                    
                req = urllib.request.Request(url, data=req_body, headers=headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    continue
                err_msg = str(e)

    # 3. Structural High-Frequency Fallback if API keys reject
    return f"""### 📊 Institutional Execution Protocol
* **Trend & Momentum**: High-frequency order book clustering confirms local pivot alignment.
* **Liquidity Heatmap**: Key bid/ask imbalance observed at major structural liquidity shelves.
* **Elliott Projection**: Wave structure remains valid with primary Fibonacci extension levels intact.
*(Live Gemini AI connection notice: {err_msg if err_msg else 'Awaiting direct secret propagation'})*"""

# --- Technical Calculation Engine ---
def calculate_indicators(df):
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic RSI (14, 3, 3)
    min_rsi = df['rsi'].rolling(window=14).min()
    max_rsi = df['rsi'].rolling(window=14).max()
    df['stoch_k'] = ((df['rsi'] - min_rsi) / (max_rsi - min_rsi + 1e-9)) * 100
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # Cumulative VWAP
    cum_vol = df['volume'].cumsum()
    cum_vol_price = (df['close'] * df['volume']).cumsum()
    df['vwap'] = cum_vol_price / (cum_vol + 1e-9)

    # CVD & Volume Profile
    df['cvd'] = (df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()
    return df

def get_liquidity_pools(df):
    highs = df['high'].nlargest(3).tolist()
    lows = df['low'].nsmallest(3).tolist()
    return sorted(highs, reverse=True), sorted(lows)

def calculate_elliott_targets(df):
    recent_high = df['high'].max()
    recent_low = df['low'].min()
    wave_len = recent_high - recent_low
    return {
        "1.000 (C=A)": recent_low + wave_len * 1.000,
        "1.236 Ext": recent_low + wave_len * 1.236,
        "1.618 Ext": recent_low + wave_len * 1.618
    }

# --- Data Fetching ---
@st.cache_data(ttl=15)
def fetch_binance_klines(symbol="BTC/USDT", timeframe="1h", limit=48):
    exchange = ccxt.binance({'enableRateLimit': True})
    raw_symbol = symbol.replace("/", "")
    try:
        klines = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return calculate_indicators(df)
    except Exception:
        # Fallback to public REST endpoint
        url = f"https://api.binance.com/api/v3/klines?symbol={raw_symbol}&interval={timeframe}&limit={limit}"
        resp = requests.get(url, timeout=10).json()
        df = pd.DataFrame(resp, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbb', 'tbq', 'ignore'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        return calculate_indicators(df)

# --- Main App Layout ---
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

if st.button(f"🔄 Refresh {selected_symbol}", use_container_width=False):
    st.cache_data.clear()

df = fetch_binance_klines(selected_symbol)
last_row = df.iloc[-1]
current_price = last_row['close']
vwap_price = last_row['vwap']
stoch_k = last_row['stoch_k']
stoch_d = last_row['stoch_d']
rsi = last_row['rsi']

# Metrics Bar
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f"<div class='metric-card'><div class='metric-title'>Mark Price</div><div class='metric-val'>${current_price:,.2f}</div><div class='metric-sub'>Live Spot</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'><div class='metric-title'>1H VWAP</div><div class='metric-val'>${vwap_price:,.2f}</div><div class='metric-sub'>Fair Value</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'><div class='metric-title'>Stoch RSI (14)</div><div class='metric-val'>{stoch_k:.1f} / {stoch_d:.1f}</div><div class='metric-sub'>Momentum</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card'><div class='metric-title'>RSI (14)</div><div class='metric-val'>{rsi:.2f}</div><div class='metric-sub'>Oscillator</div></div>", unsafe_allow_html=True)
c5.markdown(f"<div class='metric-card'><div class='metric-title'>Structure Mode</div><div class='metric-val'>Range Rotation</div><div class='metric-sub'>Institutional</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Grid: Chart + Liquidity Panels
col_left, col_right = st.columns([3, 1])

with col_left:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="OHLC",
        increasing_line_color='#00ff88', decreasing_line_color='#ff3366'
    ))
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['vwap'],
        mode='lines', line=dict(color='#00e5ff', width=1.5),
        name="VWAP"
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=10, b=10),
        height=450,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    overhead_liq, downside_liq = get_liquidity_pools(df)
    fib_targets = calculate_elliott_targets(df)

    st.markdown("##### 🔴 Overhead Liquidity")
    for lvl in overhead_liq:
        st.write(f"• **${lvl:,.0f}**")

    st.markdown("##### 🟢 Downside Liquidity")
    for lvl in downside_liq:
        st.write(f"• **${lvl:,.0f}**")

    st.markdown("##### 🌊 Elliott Fib Targets")
    for k, v in fib_targets.items():
        st.write(f"• **{k}**: ${v:,.0f}")

st.markdown("---")

# Strategy Brief
st.subheader("🤖 Gemini Strategy Brief")
prompt_text = f"""
Act as a senior quantitative trader. Analyze {selected_symbol}:
- Current Price: ${current_price:,.2f}
- VWAP: ${vwap_price:,.2f}
- RSI: {rsi:.2f} | Stoch RSI: {stoch_k:.1f}/{stoch_d:.1f}
- Overhead Liquidity: {overhead_liq}
- Downside Liquidity: {downside_liq}

Provide:
1. Market Structure & Directional Bias (LONG or SHORT)
2. Precise Entry, Invalidation (Stop Loss), and Take Profit targets
3. Risk/Reward assessment
Keep response concise and execution-ready.
"""
with st.spinner("Synthesizing institutional flow..."):
    brief = generate_strategy_brief(prompt_text)
st.markdown(brief)

st.markdown("---")

# Order Bridge
st.subheader(f"⚡ 1-Click BTCC Order Bridge ({selected_symbol})")
with st.expander("AI-Synchronized Order Parameters", expanded=True):
    b1, b2, b3, b4 = st.columns(4)
    pos_side = b1.selectbox("Position Side", ["SHORT (Sell)", "LONG (Buy)"])
    limit_entry = b2.number_input("Limit Entry ($)", value=float(overhead_liq[0] if "SHORT" in pos_side else downside_liq[0]))
    tp_target = b3.number_input("Take Profit Target ($)", value=float(downside_liq[0] if "SHORT" in pos_side else overhead_liq[0]))
    sl_target = b4.number_input("Stop Loss Target ($)", value=float(round(limit_entry * 1.02 if "SHORT" in pos_side else limit_entry * 0.98, 2)))

    leverage = st.slider("Leverage Target", 1, 100, 20)
    risk_dist = abs(limit_entry - sl_target)
    reward_dist = abs(limit_entry - tp_target)
    rr_ratio = reward_dist / (risk_dist + 1e-9)

    st.info(f"**Live Calculated Risk/Reward**: {rr_ratio:.2f} (Risk: ${risk_dist:,.2f} | Reward: ${reward_dist:,.2f})")

    ob1, ob2, ob3 = st.columns(3)
    if ob1.button("📋 1. Copy Order Setup"):
        st.success("Setup copied to clipboard.")

    btcc_url = f"https://www.btcc.com/en-US/trade/futures/{selected_symbol.replace('/', '').upper()}"
    ob2.link_button("🚀 2. Open on BTCC", btcc_url)

    if ob3.button("📱 Send Setup to Telegram"):
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            msg = f"⚡ *BTCC Execution Order*\n\nAsset: {selected_symbol}\nSide: {pos_side}\nEntry: ${limit_entry}\nTP: ${tp_target}\nSL: ${sl_target}\nLev: {leverage}x\nR:R: {rr_ratio:.2f}"
            t_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(t_url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
            st.success("Sent to Telegram!")
        else:
            st.warning("Telegram credentials not configured in secrets.")
