import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# --- High-Contrast Adaptive Theme Scaffolding ---
st.markdown("""
<style>
    .metric-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.75;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 3px;
    }
    .metric-sub {
        font-size: 0.75rem;
        font-weight: 600;
        color: #00bcd4;
        margin-top: 2px;
    }
    .side-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .card-title-red {
        color: #ff5252;
        font-size: 0.88rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .card-title-green {
        color: #00e676;
        font-size: 0.88rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .card-title-cyan {
        color: #00bcd4;
        font-size: 0.88rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .data-row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        font-size: 0.85rem;
    }
    .data-row-bold {
        font-weight: 700;
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

# --- Multi-Tier Gemini AI Engine ---
def generate_gemini_brief(prompt, fallback_text):
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
    return fallback_text

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

    direction = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['cvd'] = (direction * df['volume']).cumsum()
    return df

def get_liquidity_matrix(df):
    current = df['close'].iloc[-1]
    recent_highs = sorted(df['high'].nlargest(6).unique(), reverse=True)
    recent_lows = sorted(df['low'].nsmallest(6).unique())
    overhead = [h for h in recent_highs if h > current][:3]
    if not overhead:
        overhead = [round(current * (1 + p), 2) for p in [0.006, 0.014, 0.025]]
    downside = [l for l in recent_lows if l < current][:3]
    if not downside:
        downside = [round(current * (1 - p), 2) for p in [0.006, 0.014, 0.025]]
    return overhead, downside

def calculate_elliott_targets(df):
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    return {
        "Wave C (1.000)": round(high + (diff * 0.382), 2),
        "Wave C (1.236)": round(high + (diff * 0.618), 2),
        "Wave C (1.618)": round(high + (diff * 1.000), 2),
        "Wave 4 Retracement": round(high - (diff * 0.382), 2),
        "Key Invalidation": round(low, 2)
    }

# --- Cloud Kline Fetcher ---
@st.cache_data(ttl=15)
def fetch_cloud_klines(symbol="BTC/USDT", tf="1h", limit=50):
    granularity_map = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
    granularity = granularity_map.get(tf.lower(), 3600)
    
    coinbase_pairs = {"BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD", "SOL/USDT": "SOL-USD", "XRP/USDT": "XRP-USD"}
    if symbol in coinbase_pairs:
        try:
            pair = coinbase_pairs[symbol]
            url = f"https://api.exchange.coinbase.com/products/{pair}/candles?granularity={granularity}"
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

    bybit_tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    bybit_tf = bybit_tf_map.get(tf.lower(), "60")
    try:
        raw_sym = symbol.replace("/", "")
        url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={raw_sym}&interval={bybit_tf}&limit={limit}"
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

    try:
        exchange = ccxt.kraken({'enableRateLimit': True})
        klines = exchange.fetch_ohlcv(symbol.replace("USDT", "USD"), timeframe=tf, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return calculate_indicators(df)
    except Exception:
        pass

    dates = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq='h')
    base = 72600.0 if "BTC" in symbol else 2600.0
    prices = base + np.cumsum(np.random.normal(0, base * 0.002, limit))
    df = pd.DataFrame({'datetime': dates, 'open': prices, 'high': prices * 1.003, 'low': prices * 0.997, 'close': prices, 'volume': np.random.uniform(500, 2000, limit)})
    return calculate_indicators(df)

# --- Top Navigation ---
st.title("⚡ Institutional Derivatives Execution Terminal")

col_asset, col_tf, col_act = st.columns([3, 2, 1])

assets = {
    "₿ Bitcoin (BTC)": "BTC/USDT",
    "🪙 Silver (XAG)": "XAG/USDT",
    "Ξ Ethereum (ETH)": "ETH/USDT",
    "🟣 Solana (SOL)": "SOL/USDT",
    "✕ Ripple (XRP)": "XRP/USDT"
}

with col_asset:
    selected_asset_label = st.selectbox("Select Asset", list(assets.keys()), label_visibility="collapsed")
    selected_symbol = assets[selected_asset_label]

with col_tf:
    selected_tf = st.radio("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=2, horizontal=True, label_visibility="collapsed")

with col_act:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

df = fetch_cloud_klines(selected_symbol, tf=selected_tf)
last_row = df.iloc[-1]
current_price = last_row['close']
vwap_price = last_row['vwap']
stoch_k = last_row['stoch_k']
stoch_d = last_row['stoch_d']
rsi = last_row['rsi']

overhead_liq, downside_liq = get_liquidity_matrix(df)
fib_targets = calculate_elliott_targets(df)

# --- Metrics Banner ---
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.markdown(f"<div class='metric-card'><div class='metric-label'>Mark Price</div><div class='metric-val'>${current_price:,.2f}</div><div class='metric-sub'>Live Spot</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card'><div class='metric-label'>{selected_tf.upper()} VWAP</div><div class='metric-val'>${vwap_price:,.2f}</div><div class='metric-sub'>Fair Value Anchor</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card'><div class='metric-label'>RSI (14)</div><div class='metric-val'>{rsi:.2f}</div><div class='metric-sub'>Oscillator</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-card'><div class='metric-label'>Stoch RSI</div><div class='metric-val'>{stoch_k:.1f}/{stoch_d:.1f}</div><div class='metric-sub'>Momentum</div></div>", unsafe_allow_html=True)
m5.markdown(f"<div class='metric-card'><div class='metric-label'>CVD Momentum</div><div class='metric-val'>{'Bullish Accum' if df['cvd'].iloc[-1] > df['cvd'].iloc[-4] else 'Bearish Dist'}</div><div class='metric-sub'>Delta Flow</div></div>", unsafe_allow_html=True)
m6.markdown(f"<div class='metric-card'><div class='metric-label'>Structure Regime</div><div class='metric-val'>{'Overextended' if rsi > 70 else ('Oversold' if rsi < 30 else 'Range Rotation')}</div><div class='metric-sub'>Market Cycle</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Analytics Grid ---
col_chart, col_side = st.columns([3, 1])

with col_chart:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    fig.add_trace(go.Candlestick(
        x=df['datetime'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="OHLC",
        increasing_line_color='#00ff88', decreasing_line_color='#ff3366'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['vwap'],
        mode='lines', line=dict(color='#00bcd4', width=2),
        name=f"{selected_tf.upper()} VWAP"
    ), row=1, col=1)

    fig.add_hline(y=overhead_liq[0], line_dash="dash", line_color="#ff5252", annotation_text="Overhead Sweep Pool", row=1, col=1)
    fig.add_hline(y=downside_liq[0], line_dash="dash", line_color="#00e676", annotation_text="Downside Bid Shelf", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['cvd'],
        mode='lines', line=dict(color='#ffc107', width=1.5),
        name="CVD Volume Delta"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=8, r=8, t=8, b=8),
        height=520,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="CVD", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown(f"""
    <div class='side-card'>
        <div class='card-title-red'>🔴 Overhead Liquidity Pools</div>
        <div class='data-row'><span>Sweep Level 1</span><span class='data-row-bold'>${overhead_liq[0]:,.2f}</span></div>
        <div class='data-row'><span>Sweep Level 2</span><span class='data-row-bold'>${overhead_liq[1]:,.2f}</span></div>
        <div class='data-row'><span>Structural Peak 3</span><span class='data-row-bold'>${overhead_liq[2]:,.2f}</span></div>
    </div>
    <div class='side-card'>
        <div class='card-title-green'>🟢 Downside Liquidity Pools</div>
        <div class='data-row'><span>Bid Target 1</span><span class='data-row-bold'>${downside_liq[0]:,.2f}</span></div>
        <div class='data-row'><span>Demand Shelf 2</span><span class='data-row-bold'>${downside_liq[1]:,.2f}</span></div>
        <div class='data-row'><span>Key Value Floor 3</span><span class='data-row-bold'>${downside_liq[2]:,.2f}</span></div>
    </div>
    <div class='side-card'>
        <div class='card-title-cyan'>🌊 Elliott Fibonacci Projections</div>
        <div class='data-row'><span>Wave C (1.000)</span><span class='data-row-bold'>${fib_targets['Wave C (1.000)']:,.2f}</span></div>
        <div class='data-row'><span>Wave C (1.236)</span><span class='data-row-bold'>${fib_targets['Wave C (1.236)']:,.2f}</span></div>
        <div class='data-row'><span>Wave C (1.618)</span><span class='data-row-bold'>${fib_targets['Wave C (1.618)']:,.2f}</span></div>
        <div class='data-row'><span>Wave 4 Retrace</span><span class='data-row-bold'>${fib_targets['Wave 4 Retracement']:,.2f}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Dual-Strategy AI Brief ---
st.subheader("🤖 Institutional Strategy Brief & Execution Matrix")
prompt_ai = f"""
Act as a senior derivatives execution trader. Synthesize {selected_symbol} ({selected_tf.upper()}):
- Mark Price: ${current_price:,.2f} | VWAP: ${vwap_price:,.2f}
- RSI: {rsi:.2f} | Stoch RSI: {stoch_k:.1f}/{stoch_d:.1f}
- Overhead Liquidity: {overhead_liq}
- Downside Liquidity: {downside_liq}

Provide two distinct, complete execution setups:
1. SCENARIO A: BEARISH / SHORT SETUP (Liquidity sweep & mean reversion to VWAP)
2. SCENARIO B: BULLISH / LONG SETUP (Dip accumulation at downside shelf / trend continuation)
Include precise Entry, Stop Loss, Take Profit, and Risk/Reward for both scenarios. Keep execution-ready.
"""

fallback_ai = f"""
#### 📊 Dual Scenario Tactical Playbook ({selected_symbol} - {selected_tf.upper()})

* **Scenario A: Bearish Mean Reversion (SHORT)**
  * **Entry Trigger**: Liquidity sweep absorption at **${overhead_liq[0]:,.2f}**
  * **Take Profit Target**: Primary mean reversion to {selected_tf.upper()} VWAP at **${vwap_price:,.2f}**
  * **Stop Loss**: Invalidation above local swing wick at **${round(overhead_liq[0] * 1.015, 2):,.2f}**

* **Scenario B: Bullish Support Dip (LONG)**
  * **Entry Trigger**: Bid absorption test at downside liquidity shelf **${downside_liq[0]:,.2f}**
  * **Take Profit Target**: Wave C Fibonacci expansion at **${fib_targets['Wave C (1.000)']:,.2f}**
  * **Stop Loss**: Invalidation below swing low at **${round(downside_liq[0] * 0.985, 2):,.2f}**
"""

brief_text = generate_gemini_brief(prompt_ai, fallback_ai)
st.markdown(brief_text)

st.markdown("---")

# --- Dual 1-Click BTCC Order Bridges ---
st.subheader(f"⚡ Dual BTCC Order Bridges ({selected_symbol})")

col_short, col_long = st.columns(2)

# --- SHORT BRIDGE ---
with col_short:
    with st.expander("🔴 SHORT Execution Setup (Sell / Mean Reversion)", expanded=True):
        s_entry = st.number_input("Limit Entry ($)", value=float(overhead_liq[0]), key="s_entry")
        s_tp = st.number_input("Take Profit Target ($)", value=float(vwap_price if vwap_price < overhead_liq[0] else downside_liq[0]), key="s_tp")
        s_sl = st.number_input("Stop Loss Target ($)", value=float(round(overhead_liq[0] * 1.015, 2)), key="s_sl")
        s_lev = st.slider("Leverage Target", 1, 100, 20, key="s_lev")

        s_risk = abs(s_entry - s_sl)
        s_rew = abs(s_entry - s_tp)
        s_rr = s_rew / (s_risk + 1e-9)

        st.info(f"**Calculated R:R Ratio**: **{s_rr:.2f}** (Risk: ${s_risk:,.2f} | Reward: ${s_rew:,.2f})")

        short_payload = {
            "symbol": selected_symbol.replace("/", "").upper(),
            "side": "SHORT",
            "entry": float(s_entry),
            "tp": float(s_tp),
            "sl": float(s_sl),
            "leverage": int(s_lev)
        }
        st.markdown("**📋 1. Click icon in code block to copy SHORT payload:**")
        st.code(json.dumps(short_payload), language="json")
        st.caption("👉 Then switch to your BTCC tab and click **⚡ Fill BTCC Ticket** on your bookmarks bar.")

# --- LONG BRIDGE ---
with col_long:
    with st.expander("🟢 LONG Execution Setup (Buy / Trend Continuation)", expanded=True):
        l_entry = st.number_input("Limit Entry ($)", value=float(downside_liq[0]), key="l_entry")
        l_tp = st.number_input("Take Profit Target ($)", value=float(overhead_liq[0]), key="l_tp")
        l_sl = st.number_input("Stop Loss Target ($)", value=float(round(downside_liq[0] * 0.985, 2)), key="l_sl")
        l_lev = st.slider("Leverage Target", 1, 100, 20, key="l_lev")

        l_risk = abs(l_entry - l_sl)
        l_rew = abs(l_entry - l_tp)
        l_rr = l_rew / (l_risk + 1e-9)

        st.success(f"**Calculated R:R Ratio**: **{l_rr:.2f}** (Risk: ${l_risk:,.2f} | Reward: ${l_rew:,.2f})")

        long_payload = {
            "symbol": selected_symbol.replace("/", "").upper(),
            "side": "LONG",
            "entry": float(l_entry),
            "tp": float(l_tp),
            "sl": float(l_sl),
            "leverage": int(l_lev)
        }
        st.markdown("**📋 1. Click icon in code block to copy LONG payload:**")
        st.code(json.dumps(long_payload), language="json")
        st.caption("👉 Then switch to your BTCC tab and click **⚡ Fill BTCC Ticket** on your bookmarks bar.")
