import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timezone

from smc_engine import calculate_clean_indicators, get_structural_levels

st.set_page_config(
    page_title="Institutional Tactical Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    margin-bottom: 8px;
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

def fetch_live_spot_price(symbol="BTC/USDT"):
    headers = {"User-Agent": "Mozilla/5.0"}
    fsym = "BTC" if "BTC" in symbol else ("ETH" if "ETH" in symbol else ("SOL" if "SOL" in symbol else ("XRP" if "XRP" in symbol else "XAG")))
    
    if fsym == "XAG":
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/SI=F?interval=1m&range=1d"
            r = requests.get(url, headers=headers, timeout=3).json()
            price = r['chart']['result'][0]['meta']['regularMarketPrice']
            if price:
                return float(price)
        except Exception:
            pass
        return 28.50  # Updated current spot silver fallback
    else:
        try:
            mexc_symbol = f"{fsym}USDT"
            url = f"https://api.mexc.com/api/v3/ticker/price?symbol={mexc_symbol}"
            r = requests.get(url, headers=headers, timeout=3).json()
            if r.get('price'):
                return float(r['price'])
        except Exception:
            pass
        return 96500.0 if fsym == "BTC" else (2600.0 if fsym == "ETH" else (140.0 if fsym == "SOL" else 1.35))

@st.cache_data(ttl=5)
def fetch_cloud_klines(symbol="BTC/USDT", tf="5m", limit=60, current_spot=78900.0):
    headers = {"User-Agent": "Mozilla/5.0"}
    fsym = "BTC" if "BTC" in symbol else ("ETH" if "ETH" in symbol else ("SOL" if "SOL" in symbol else ("XRP" if "XRP" in symbol else "XAG")))
    df = None
    
    if fsym == "XAG":
        # Fetch from Yahoo Finance
        try:
            interval = tf # "5m", "15m", "1h", "4h"
            if tf == "5m": range_val = "2d"
            elif tf == "15m": range_val = "5d"
            elif tf == "1h": range_val = "15d"
            else: range_val = "60d" # 4h
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/SI=F?interval={interval}&range={range_val}"
            r = requests.get(url, headers=headers, timeout=5).json()
            res = r['chart']['result'][0]
            timestamps = res['timestamp']
            quote = res['indicators']['quote'][0]
            
            df_temp = pd.DataFrame({
                'timestamp': timestamps,
                'open': quote['open'],
                'high': quote['high'],
                'low': quote['low'],
                'close': quote['close'],
                'volume': quote['volume']
            })
            df = df_temp.dropna().reset_index(drop=True).tail(limit)
        except Exception as e:
            print(f"Error fetching XAG from Yahoo Finance: {e}")
            pass
    else:
        # Fetch from MEXC API
        try:
            mexc_symbol = f"{fsym}USDT"
            mexc_interval = "5m" if tf == "5m" else ("15m" if tf == "15m" else ("60m" if tf == "1h" else "4h"))
            url = f"https://api.mexc.com/api/v3/klines?symbol={mexc_symbol}&interval={mexc_interval}&limit={limit}"
            r = requests.get(url, headers=headers, timeout=5).json()
            if isinstance(r, list) and len(r) > 0:
                rows = []
                for row in r:
                    rows.append({
                        'timestamp': int(row[0]) // 1000,
                        'open': float(row[1]),
                        'high': float(row[2]),
                        'low': float(row[3]),
                        'close': float(row[4]),
                        'volume': float(row[5])
                    })
                df = pd.DataFrame(rows)
        except Exception as e:
            print(f"Error fetching {fsym} from MEXC: {e}")
            pass
            
    # Fallback to random data if fetching failed
    if df is None or df.empty or df['close'].iloc[-1] == 0:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq='5min')
        base = float(current_spot)
        spread = base * 0.002
        prices = base + np.cumsum(np.random.normal(0, spread * 0.4, limit))
        df = pd.DataFrame({
            'timestamp': dates.astype(int) // 10**9,
            'open': prices - (spread * 0.1),
            'high': prices + spread,
            'low': prices - spread,
            'close': prices,
            'volume': np.random.uniform(500, 2000, limit)
        })
        
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s') - pd.Timedelta(hours=5)
    return calculate_clean_indicators(df)

# Asset Bar
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
    if st.button("🔄 Refresh", key="top_refresh_btn", use_container_width=True):
        st.cache_data.clear()

live_spot = fetch_live_spot_price(selected_symbol)
df = fetch_cloud_klines(selected_symbol, tf=selected_tf, current_spot=live_spot)
levels = get_structural_levels(df, selected_symbol, live_spot)

dec = levels['decimals']
def fmt(val):
    if dec == 4: return f"{val:,.4f}"
    elif dec == 3: return f"{val:,.3f}"
    elif dec == 2: return f"{val:,.2f}"
    return f"{val:,.1f}"

# Fetch Brain Conviction Scores and Sentiment
try:
    from brain_db import get_setting
    from brain_metrics import fetch_binance_funding_and_oi, fetch_fred_dxy
    from brain_scorer import calculate_conviction_score
    
    fred_key = get_setting("fred_api_key")
    fred_data = fetch_fred_dxy(fred_key) if fred_key else {"dxy_value": 100.0, "dxy_trend": "FLAT", "dxy_sma_5d": 100.0}
    btc_metrics = fetch_binance_funding_and_oi("BTCUSDT")
    
    bullish_fvg_present = any(df['bullish_fvg'].iloc[-5:]) if 'bullish_fvg' in df.columns else False
    bearish_fvg_present = any(df['bearish_fvg'].iloc[-5:]) if 'bearish_fvg' in df.columns else False
    
    # Calculate Bullish (LONG) conviction
    f_rate = btc_metrics["funding_rate"] if "BTC" in selected_symbol.upper() else None
    oi_t = btc_metrics["oi_trend"] if "BTC" in selected_symbol.upper() else None
    dxy_t = fred_data["dxy_trend"] if "XAG" in selected_symbol.upper() or "SILVER" in selected_symbol.upper() else None
    
    long_score, long_reasons = calculate_conviction_score(
        selected_symbol, "LONG", levels['long_plan']['entry'], levels['vwap'], levels['rsi'], bullish_fvg_present,
        funding_rate=f_rate, oi_trend=oi_t, dxy_trend=dxy_t
    )
    
    # Calculate Bearish (SHORT) conviction
    short_score, short_reasons = calculate_conviction_score(
        selected_symbol, "SHORT", levels['short_plan']['entry'], levels['vwap'], levels['rsi'], bearish_fvg_present,
        funding_rate=f_rate, oi_trend=oi_t, dxy_trend=dxy_t
    )
except Exception as e:
    long_score = 50.0
    short_score = 50.0
    long_reasons = [f"Calculation fallback: {e}"]
    short_reasons = [f"Calculation fallback: {e}"]
    fred_data = {"dxy_value": 100.0, "dxy_trend": "FLAT"}

long_reasons_html = "".join([f"<li>{r}</li>" for r in long_reasons])
if not long_reasons_html:
    long_reasons_html = "<li>No significant directional biases found by the Brain.</li>"

short_reasons_html = "".join([f"<li>{r}</li>" for r in short_reasons])
if not short_reasons_html:
    short_reasons_html = "<li>No significant directional biases found by the Brain.</li>"

# Operational Header Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.markdown(f"<div class='metric-box'><div class='metric-label'>Live Spot</div><div class='metric-val'>${fmt(levels['spot'])}</div><div class='metric-sub'>USDT Perpetual</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-box'><div class='metric-label'>Session VWAP</div><div class='metric-val'>${fmt(levels['vwap'])}</div><div class='metric-sub'>Fair Value Pivot</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-box'><div class='metric-label'>Dealing Range</div><div class='metric-val'>${fmt(levels['long_plan']['entry'])} - ${fmt(levels['short_plan']['entry'])}</div><div class='metric-sub'>Discount / Premium</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-box'><div class='metric-label'>ATR Buffer</div><div class='metric-val'>${fmt(levels['atr'])}</div><div class='metric-sub'>14-Period Volatility</div></div>", unsafe_allow_html=True)
m5.markdown(f"<div class='metric-box'><div class='metric-label'>RSI ({selected_tf})</div><div class='metric-val'>{levels['rsi']}</div><div class='metric-sub'>{'Overbought' if levels['rsi'] > 70 else ('Oversold' if levels['rsi'] < 30 else 'Neutral Flow')}</div></div>", unsafe_allow_html=True)
m6.markdown(f"<div class='metric-box'><div class='metric-label'>Macro Context</div><div class='metric-val'>{'DXY ' + str(round(fred_data['dxy_value'], 2)) if levels['is_silver'] else 'L2 Delta Flow'}</div><div class='metric-sub'>{'COMEX / ' + fred_data['dxy_trend'] if levels['is_silver'] else 'Crypto Derivatives'}</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# High-Conviction Bullish and Bearish Plans (The Big Anchors)
col_bull, col_bear = st.columns(2)
s_plan = levels['short_plan']
l_plan = levels['long_plan']

with col_bull:
    st.markdown(f"""
    <div class='playbook-card' style='border-top: 3px solid #089981;'>
        <div class='bull-title'>🟢 Bullish High-Conviction Anchor (Brain Score: 🔥 {long_score}%)</div>
        <div class='section-title'>Macro Thesis</div>
        <p style='font-size: 0.88rem; margin-bottom: 8px;'>Institutional accumulation in deep Discount demand at <b>${fmt(l_plan['entry'])}</b> with absorption of sell-side liquidity opens rotation back to Equilibrium and Premium supply.</p>
        <div class='section-title'>Brain Sentiment &amp; Biases</div>
        <ul style='font-size: 0.85rem; padding-left: 20px; margin-bottom: 8px;'>
            {long_reasons_html}
        </ul>
        <div class='section-title'>Execution Rules</div>
        <ul style='font-size: 0.85rem; padding-left: 20px; margin-bottom: 8px;'>
            <li>Sweep of sell-side liquidity into the Discount Floor at <b>${fmt(l_plan['entry'])}</b>.</li>
            <li>5m absorption wicks with positive CVD delta shift.</li>
            <li>Targeting mean-reversion rotation toward Session VWAP.</li>
        </ul>
        <div class='section-title'>Upside Roadmap</div>
        <div style='margin-bottom: 12px;'>
            <span class='roadmap-tag'>Demand Floor: ${fmt(l_plan['entry'])}</span> ➔ 
            <span class='roadmap-tag'>VWAP / EQ: ${fmt(levels['vwap'])}</span> ➔ 
            <span class='roadmap-tag'>Supply Target: ${fmt(l_plan['tp'])}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.error(f"🚨 **Bullish Invalidation:** Clean 5m close below **${fmt(l_plan['sl'])}** voids thesis.")

with col_bear:
    st.markdown(f"""
    <div class='playbook-card' style='border-top: 3px solid #f23645;'>
        <div class='bear-title'>🔴 Bearish High-Conviction Anchor (Brain Score: 🔥 {short_score}%)</div>
        <div class='section-title'>Macro Thesis</div>
        <p style='font-size: 0.88rem; margin-bottom: 8px;'>Exhaustion wick into the overhead Premium Supply block at <b>${fmt(s_plan['entry'])}</b> with failure to accept above POC confirms distribution ready to rotate lower.</p>
        <div class='section-title'>Brain Sentiment &amp; Biases</div>
        <ul style='font-size: 0.85rem; padding-left: 20px; margin-bottom: 8px;'>
            {short_reasons_html}
        </ul>
        <div class='section-title'>Execution Rules</div>
        <ul style='font-size: 0.85rem; padding-left: 20px; margin-bottom: 8px;'>
            <li>Sweep of buy-side liquidity into Premium Supply at <b>${fmt(s_plan['entry'])}</b>.</li>
            <li>5m rejection candle / CHoCH breakdown beneath local shelf.</li>
            <li>Failure to hold structural Equilibrium.</li>
        </ul>
        <div class='section-title'>Downside Roadmap</div>
        <div style='margin-bottom: 12px;'>
            <span class='roadmap-tag'>Supply Ceiling: ${fmt(s_plan['entry'])}</span> ➔ 
            <span class='roadmap-tag'>VWAP / EQ: ${fmt(levels['vwap'])}</span> ➔ 
            <span class='roadmap-tag'>Discount Floor: ${fmt(s_plan['tp'])}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.error(f"🚨 **Bearish Invalidation:** Clean 5m close above **${fmt(s_plan['sl'])}** voids thesis.")

# Tactical Execution Matrix (Replacing the unstable recommendation banner)
st.divider()
st.subheader(f"📊 Tactical Execution Matrix ({selected_symbol})")

matrix_df = pd.DataFrame(levels['tactical_matrix'])
st.table(matrix_df)

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
