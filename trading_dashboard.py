import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import urllib.request
from datetime import datetime, timezone

# Import the sanitized math core
from smc_engine import calculate_clean_indicators, get_structural_levels

st.set_page_config(
    page_title="Institutional Tactical Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-Contrast Styling (Webull Dark Mode Inspired)
st.markdown("""
<style>
.metric-box {
    background: #131722;
    border: 1px solid #2a2e39;
    border-radius: 6px;
    padding: 10px 14px;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #787b86;
    letter-spacing: 0.05em;
}
.metric-val {
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 2px;
    color: #d1d4dc;
}
.metric-sub {
    font-size: 0.72rem;
    font-weight: 600;
    color: #2962ff;
    margin-top: 2px;
}
.playbook-card {
    background: #1e222d;
    border: 1px solid #2a2e39;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.bull-title {
    color: #089981;
    font-size: 1.1rem;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.bear-title {
    color: #f23645;
    font-size: 1.1rem;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.section-title {
    color: #d1d4dc;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 10px;
    margin-bottom: 4px;
}
.roadmap-tag {
    background: #2a2e39;
    padding: 4px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.85rem;
    color: #f0f3fa;
}
</style>
""", unsafe_allow_html=True)

def get_secret(key_name, default=""):
    try:
        return st.secrets[key_name]
    except Exception:
        return os.environ.get(key_name, default)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "")

def fetch_live_spot_price(symbol="BTC/USDT"):
    headers = {"User-Agent": "Mozilla/5.0"}
    fsym = "BTC" if "BTC" in symbol else ("ETH" if "ETH" in symbol else ("SOL" if "SOL" in symbol else ("XRP" if "XRP" in symbol else "XAG")))
    if fsym == "XAG":
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/ticker/price?symbol=XAGUSDT", headers=headers, timeout=3).json()
            if r.get('price'): return float(r['price'])
        except Exception:
            pass
        return 69.340
    else:
        try:
            r = requests.get(f"https://api.coinbase.com/v2/prices/{fsym}-USD/spot", headers=headers, timeout=3).json()
            if r.get('data', {}).get('amount'): return float(r['data']['amount'])
        except Exception:
            pass
        return 77246.0

@st.cache_data(ttl=5)
def fetch_cloud_klines(symbol="BTC/USDT", tf="5m", limit=60):
    headers = {"User-Agent": "Mozilla/5.0"}
    fsym = "BTC" if "BTC" in symbol else ("ETH" if "ETH" in symbol else ("SOL" if "SOL" in symbol else ("XRP" if "XRP" in symbol else "XAG")))
    df = None
    try:
        agg = 5 if tf == "5m" else (15 if tf == "15m" else 60)
        endpoint = "histominute" if "m" in tf else ("histohour" if "h" in tf else "histoday")
        tsym = "USD" if fsym == "XAG" else "USDT"
        url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}?fsym={fsym}&tsym={tsym}&limit={limit}&aggregate={agg}"
        r = requests.get(url, headers=headers, timeout=4).json()
        if r.get('Response') == 'Success' and r.get('Data', {}).get('Data'):
            raw = r['Data']['Data']
            df = pd.DataFrame(raw)
            df = df[['time', 'open', 'high', 'low', 'close', 'volumeto']]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    except Exception:
        pass
    if df is None:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq='5min')
        base = 77246.0
        prices = base + np.cumsum(np.random.normal(0, 30.0, limit))
        df = pd.DataFrame({
            'timestamp': dates.astype(int) // 10**9,
            'open': prices - 10,
            'high': prices + 25,
            'low': prices - 25,
            'close': prices,
            'volume': np.random.uniform(500, 2000, limit)
        })
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s') - pd.Timedelta(hours=5)
    return calculate_clean_indicators(df)

# Header Asset Controls
col_asset, col_tf, col_act = st.columns([4, 2, 1])
assets = {
    "₿ Bitcoin (BTCUSDT)": "BTC/USDT",
    "🪙 Silver (SILVERUSDT)": "XAG/USDT",
    "Ξ Ethereum (ETHUSDT)": "ETH/USDT",
    "🟣 Solana (SOLUSDT)": "SOL/USDT",
    "✕ Ripple (XRPUSDT)": "XRP/USDT"
}

with col_asset:
    selected_label = st.selectbox("Select Asset", list(assets.keys()), index=0, label_visibility="collapsed")
    selected_symbol = assets[selected_label]

with col_tf:
    selected_tf = st.radio("Timeframe", ["5m", "15m", "1h", "4h"], index=0, horizontal=True, label_visibility="collapsed")

with col_act:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()

# Compute Data & Levels via Sanitized smc_engine
df = fetch_cloud_klines(selected_symbol, tf=selected_tf)
live_spot = fetch_live_spot_price(selected_symbol)
levels = get_structural_levels(df, selected_symbol, live_spot)

dec = levels['decimals']
def fmt(val):
    if dec == 4: return f"{val:,.4f}"
    elif dec == 3: return f"{val:,.3f}"
    elif dec == 2: return f"{val:,.2f}"
    return f"{val:,.1f}"

# Top Level Operational Matrix
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.markdown(f"<div class='metric-box'><div class='metric-label'>Live Spot</div><div class='metric-val'>${fmt(levels['spot'])}</div><div class='metric-sub'>USDT Perpetual</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-box'><div class='metric-label'>Session VWAP</div><div class='metric-val'>${fmt(levels['vwap'])}</div><div class='metric-sub'>Fair Value Pivot</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-box'><div class='metric-label'>Decision Range</div><div class='metric-val'>${fmt(levels['long_plan']['entry'])} - ${fmt(levels['short_plan']['entry'])}</div><div class='metric-sub'>Execution Bounds</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-box'><div class='metric-label'>ATR Buffer</div><div class='metric-val'>${fmt(levels['atr'])}</div><div class='metric-sub'>14-Period Volatility</div></div>", unsafe_allow_html=True)
m5.markdown(f"<div class='metric-box'><div class='metric-label'>RSI ({selected_tf})</div><div class='metric-val'>{levels['rsi']}</div><div class='metric-sub'>{'Overbought' if levels['rsi'] > 70 else ('Oversold' if levels['rsi'] < 30 else 'Neutral Flow')}</div></div>", unsafe_allow_html=True)
m6.markdown(f"<div class='metric-box'><div class='metric-label'>Asset Type</div><div class='metric-val'>{'COMEX / SMT' if levels['is_silver'] else 'Crypto L2'}</div><div class='metric-sub'>{'Session Levels' if levels['is_silver'] else 'OrderBook Depth'}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 1-Hour Tactical Execution Playbook (Webull Style)
col_bull, col_bear = st.columns(2)

s_plan = levels['short_plan']
l_plan = levels['long_plan']

with col_bull:
    st.markdown(f"""
    <div class='playbook-card' style='border-top: 3px solid #089981;'>
        <div class='bull-title'>🟢 Bullish Execution Plan</div>
        <div class='section-title'>Thesis</div>
        <p style='font-size: 0.88rem; margin-bottom: 8px;'>Defense of the demand boundary at <b>${fmt(l_plan['entry'])}</b> and holding above VWAP (<b>${fmt(levels['vwap'])}</b>) opens the path toward upper supply rotation.</p>
        <div class='section-title'>Trigger Setup</div>
        <ul style='font-size: 0.85rem; padding-left: 20px; margin-bottom: 8px;'>
            <li>Hold above structural support floor at <b>${fmt(l_plan['entry'])}</b>.</li>
            <li>Reclaim Session VWAP at <b>${fmt(levels['vwap'])}</b> on positive delta.</li>
            <li>Confirmation close above immediate reaction wick.</li>
        </ul>
        <div class='section-title'>Upside Roadmap</div>
        <div style='margin-bottom: 12px;'>
            <span class='roadmap-tag'>Entry: ${fmt(l_plan['entry'])}</span> ➔ 
            <span class='roadmap-tag'>TP1: ${fmt(levels['vwap'])}</span> ➔ 
            <span class='roadmap-tag'>TP2: ${fmt(l_plan['tp'])}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    bull_ladder_df = pd.DataFrame({
        "Level": [f"TP1 (VWAP): ${fmt(levels['vwap'])}", f"TP2 (Supply): ${fmt(l_plan['tp'])}", f"TP3 (Expansion): ${fmt(levels['overhead'][2])}"],
        "Action": ["Take 40% Off", "Take 40% Off", "Runner Target"],
        "Execution": ["Move SL to Breakeven", "Lock Guaranteed Profit", "Trailing Pivot Stop"]
    })
    st.table(bull_ladder_df)
    st.error(f"🚨 **Bullish Invalidation Line:** Loss of **${fmt(l_plan['sl'])}** voids setup.")

with col_bear:
    st.markdown(f"""
    <div class='playbook-card' style='border-top: 3px solid #f23645;'>
        <div class='bear-title'>🔴 Bearish Execution Plan</div>
        <div class='section-title'>Thesis</div>
        <p style='font-size: 0.88rem; margin-bottom: 8px;'>Exhaustion wick into the overhead supply zone at <b>${fmt(s_plan['entry'])}</b> with failure to reclaim VWAP indicates a liquidity sweep ready to rotate down.</p>
        <div class='section-title'>Trigger Setup</div>
        <ul style='font-size: 0.85rem; padding-left: 20px; margin-bottom: 8px;'>
            <li>Rejection wick or SFP printed at <b>${fmt(s_plan['entry'])}</b>.</li>
            <li>Break below Session VWAP at <b>${fmt(levels['vwap'])}</b>.</li>
            <li>CVD delta rolling over on the 5-minute candle.</li>
        </ul>
        <div class='section-title'>Downside Roadmap</div>
        <div style='margin-bottom: 12px;'>
            <span class='roadmap-tag'>Entry: ${fmt(s_plan['entry'])}</span> ➔ 
            <span class='roadmap-tag'>TP1: ${fmt(levels['vwap'])}</span> ➔ 
            <span class='roadmap-tag'>TP2: ${fmt(s_plan['tp'])}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    bear_ladder_df = pd.DataFrame({
        "Level": [f"TP1 (VWAP): ${fmt(levels['vwap'])}", f"TP2 (Demand): ${fmt(s_plan['tp'])}", f"TP3 (Expansion): ${fmt(levels['downside'][2])}"],
        "Action": ["Take 40% Off", "Take 40% Off", "Runner Target"],
        "Execution": ["Move SL to Breakeven", "Lock Guaranteed Profit", "Trailing Pivot Stop"]
    })
    st.table(bear_ladder_df)
    st.error(f"🚨 **Bearish Invalidation Line:** Clean close above **${fmt(s_plan['sl'])}** voids setup.")

# Fast BTCC Bridges
st.divider()
st.subheader(f"⚡ Fast BTCC Order Bridges ({selected_symbol})")

col_bridge_short, col_bridge_long = st.columns(2)
clean_ticker = "SILVERUSDT" if "XAG" in selected_symbol else selected_symbol.replace("/", "").upper()
key_id = f"{clean_ticker}_{selected_tf}_{s_plan['entry']}_{l_plan['entry']}"

with col_bridge_short:
    with st.expander("🔴 SHORT Execution Payload (BTCC)", expanded=True):
        s_e = st.number_input("Limit Entry (USDT)", value=float(s_plan['entry']), step=float(10**(-dec)), format=f"%.{dec}f", key=f"se_{key_id}")
        s_tp = st.number_input("Take Profit Target (USDT)", value=float(s_plan['tp']), step=float(10**(-dec)), format=f"%.{dec}f", key=f"stp_{key_id}")
        s_sl = st.number_input("Stop Loss Target (USDT)", value=float(s_plan['sl']), step=float(10**(-dec)), format=f"%.{dec}f", key=f"ssl_{key_id}")
        s_lev = st.slider("Leverage Target", 1, 100, 20, key=f"slev_{key_id}")
        
        s_rr = abs(s_e - s_tp) / (abs(s_e - s_sl) + 1e-9)
        st.info(f"**Calculated R:R Ratio**: **1:{s_rr:.2f}**")
        
        short_payload = {
            "symbol": clean_ticker,
            "side": "SHORT",
            "entry": float(round(s_e, dec)),
            "tp": float(round(s_tp, dec)),
            "sl": float(round(s_sl, dec)),
            "leverage": int(s_lev)
        }
        st.code(json.dumps(short_payload), language="json")

with col_bridge_long:
    with st.expander("🟢 LONG Execution Payload (BTCC)", expanded=True):
        l_e = st.number_input("Limit Entry (USDT)", value=float(l_plan['entry']), step=float(10**(-dec)), format=f"%.{dec}f", key=f"le_{key_id}")
        l_tp = st.number_input("Take Profit Target (USDT)", value=float(l_plan['tp']), step=float(10**(-dec)), format=f"%.{dec}f", key=f"ltp_{key_id}")
        l_sl = st.number_input("Stop Loss Target (USDT)", value=float(l_plan['sl']), step=float(10**(-dec)), format=f"%.{dec}f", key=f"lsl_{key_id}")
        l_lev = st.slider("Leverage Target", 1, 100, 20, key=f"llev_{key_id}")
        
        l_rr = abs(l_e - l_tp) / (abs(l_e - l_sl) + 1e-9)
        st.success(f"**Calculated R:R Ratio**: **1:{l_rr:.2f}**")
        
        long_payload = {
            "symbol": clean_ticker,
            "side": "LONG",
            "entry": float(round(l_e, dec)),
            "tp": float(round(l_tp, dec)),
            "sl": float(round(l_sl, dec)),
            "leverage": int(l_lev)
        }
        st.code(json.dumps(long_payload), language="json")
