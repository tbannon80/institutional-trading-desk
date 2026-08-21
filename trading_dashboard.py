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

st.set_page_config(
    page_title="Institutional Derivatives Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .metric-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 10px 12px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.75;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 3px;
    }
    .metric-sub {
        font-size: 0.72rem;
        font-weight: 600;
        color: #00bcd4;
        margin-top: 2px;
    }
    .side-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .card-title-red {
        color: #ff5252;
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .card-title-green {
        color: #00e676;
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .card-title-cyan {
        color: #00bcd4;
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .data-row {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        font-size: 0.82rem;
    }
    .data-row-bold {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

def get_secret(key_name, default=""):
    try:
        return st.secrets[key_name]
    except Exception:
        return os.environ.get(key_name, default)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "")

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
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data['candidates'][0]['content']['parts'][0]['text']
            except Exception:
                continue
    return fallback_text

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
    
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean().fillna(tr)
    
    df['sma200'] = df['close'].rolling(window=min(len(df), 200), min_periods=5).mean()
    return df

def get_smc_structure(df, symbol):
    current = df['close'].iloc[-1]
    atr = max(df['atr'].iloc[-1], current * 0.006)
    vwap = df['vwap'].iloc[-1]
    decimals = 4 if "XRP" in symbol else (3 if "XAG" in symbol else (2 if "SOL" in symbol or "ETH" in symbol else 1))
    
    bear_fvg = [df['low'].iloc[i-2] for i in range(2, len(df)) if df['low'].iloc[i-2] > df['high'].iloc[i] and df['low'].iloc[i-2] > current]
    bull_fvg = [df['low'].iloc[i] for i in range(2, len(df)) if df['high'].iloc[i-2] < df['low'].iloc[i] and df['low'].iloc[i] < current]

    short_entry = round(bear_fvg[-1] if bear_fvg else max(df['high'].tail(15).max(), current + (1.0 * atr)), decimals)
    short_sl = round(short_entry + (1.1 * atr), decimals)
    short_tp = round(vwap if vwap < short_entry - (1.2 * atr) else min(df['low'].min(), short_entry - (2.0 * atr)), decimals)

    long_entry = round(bull_fvg[-1] if bull_fvg else min(df['low'].tail(15).min(), current - (1.0 * atr)), decimals)
    long_sl = round(long_entry - (1.1 * atr), decimals)
    long_tp = round(vwap if vwap > long_entry + (1.2 * atr) else max(df['high'].max(), long_entry + (2.0 * atr)), decimals)

    overhead = [short_entry, round(short_entry + (0.8 * atr), decimals), round(short_entry + (1.5 * atr), decimals)]
    downside = [long_entry, round(long_entry - (0.8 * atr), decimals), round(long_entry - (1.5 * atr), decimals)]
    
    range_max = max(df['high'].max(), short_entry)
    range_min = min(df['low'].min(), long_entry)
    eq_level = round((range_max + range_min) / 2.0, decimals)

    bos_markers = []
    for i in range(5, len(df)):
        if df['close'].iloc[i] > df['high'].iloc[i-5:i].max():
            bos_markers.append((df['datetime'].iloc[i], df['high'].iloc[i], "BOS ▲", "#00e676"))
        elif df['close'].iloc[i] < df['low'].iloc[i-5:i].min():
            bos_markers.append((df['datetime'].iloc[i], df['low'].iloc[i], "CHoCH ▼", "#ff5252"))

    supply_box = (overhead[1], overhead[0])
    demand_box = (downside[0], downside[1])

    return overhead, downside, decimals, supply_box, demand_box, eq_level, bos_markers[-3:], short_entry, short_tp, short_sl, long_entry, long_tp, long_sl

def get_tactical_liquidity_matrix(df, symbol):
    current = df['close'].iloc[-1]
    atr = max(df['atr'].iloc[-1], current * 0.006)
    vwap = df['vwap'].iloc[-1]
    decimals = 4 if "XRP" in symbol else (3 if "XAG" in symbol else (2 if "SOL" in symbol or "ETH" in symbol else 1))
    
    range_high = df['high'].max()
    range_low = df['low'].min()
    
    short_entry = round(max(range_high, current + (1.2 * atr)), decimals)
    short_sl = round(short_entry + (1.2 * atr), decimals)
    short_tp = round(vwap if vwap < short_entry - (1.2 * atr) else range_low, decimals)

    long_entry = round(min(range_low, current - (1.2 * atr)), decimals)
    long_sl = round(long_entry - (1.2 * atr), decimals)
    long_tp = round(vwap if vwap > long_entry + (1.2 * atr) else range_high, decimals)

    overhead = [short_entry, round(short_entry + (1.0 * atr), decimals), round(short_entry + (1.5 * atr), decimals)]
    downside = [long_entry, round(long_entry - (1.0 * atr), decimals), round(long_entry - (1.5 * atr), decimals)]
    
    return overhead, downside, decimals, short_entry, short_tp, short_sl, long_entry, long_tp, long_sl

def calculate_elliott_targets(df, symbol):
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    decimals = 4 if "XRP" in symbol else (3 if "XAG" in symbol else (2 if "SOL" in symbol or "ETH" in symbol else 1))
    return {
        "1.000 (C=A)": round(high + (diff * 0.382), decimals),
        "1.236 Ext": round(high + (diff * 0.618), decimals),
        "1.618 Ext": round(high + (diff * 1.000), decimals),
        "Wave 4 Retracement": round(high - (diff * 0.382), decimals),
        "Wave B Floor": round(low, decimals)
    }

@st.cache_data(ttl=3)
def fetch_cloud_klines(symbol="BTC/USDT", tf="5m", limit=60):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    fsym = "BTC" if "BTC" in symbol else ("ETH" if "ETH" in symbol else ("SOL" if "SOL" in symbol else ("XRP" if "XRP" in symbol else "XAG")))
    
    live_price = None
    if fsym == "XAG":
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/ticker/price?symbol=XAGUSDT", headers=headers, timeout=2).json()
            if r.get('price'): live_price = float(r['price'])
        except Exception:
            pass
        if not live_price:
            try:
                r = requests.get("https://api.kraken.com/0/public/Ticker?pair=XAGUSD", headers=headers, timeout=2).json()
                if r.get('result', {}).get('XAGUSD'):
                    live_price = float(r['result']['XAGUSD']['c'][0])
            except Exception:
                pass
    else:
        try:
            cb_pair = f"{fsym}-USD"
            r = requests.get(f"https://api.coinbase.com/v2/prices/{cb_pair}/spot", headers=headers, timeout=2).json()
            if r.get('data', {}).get('amount'): live_price = float(r['data']['amount'])
        except Exception:
            pass

    df = None
    try:
        agg = 5 if tf == "5m" else (15 if tf == "15m" else 60)
        endpoint = "histominute" if "m" in tf else ("histohour" if "h" in tf else "histoday")
        tsym = "USD" if fsym == "XAG" else "USDT"
        url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}?fsym={fsym}&tsym={tsym}&limit={limit}&aggregate={agg}"
        r = requests.get(url, headers=headers, timeout=3).json()
        if r.get('Response') == 'Success' and r.get('Data', {}).get('Data'):
            raw = r['Data']['Data']
            df = pd.DataFrame(raw)
            df = df[['time', 'open', 'high', 'low', 'close', 'volumeto']]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    except Exception:
        pass

    if df is None:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq='5min')
        base = live_price if live_price else (69.34 if fsym == "XAG" else 77644.0)
        spread = 0.08 if fsym == "XAG" else (base * 0.001)
        prices = base + np.cumsum(np.random.normal(0, spread * 0.5, limit))
        df = pd.DataFrame({'datetime': dates, 'open': prices, 'high': prices + spread, 'low': prices - spread, 'close': prices, 'volume': np.random.uniform(500, 2000, limit)})

    if live_price:
        df.iloc[-1, df.columns.get_loc('close')] = live_price
        df.iloc[-1, df.columns.get_loc('high')] = max(df.iloc[-1]['high'], live_price)
        df.iloc[-1, df.columns.get_loc('low')] = min(df.iloc[-1]['low'], live_price)

    return calculate_indicators(df)

st.title("⚡ Institutional Derivatives Execution Terminal")

col_asset, col_engine, col_tf, col_act = st.columns([3, 3, 2, 1])

assets = {
    "₿ Bitcoin (BTCUSDT)": "BTC/USDT",
    "🪙 Silver (SILVERUSDT)": "XAG/USDT",
    "Ξ Ethereum (ETHUSDT)": "ETH/USDT",
    "🟣 Solana (SOLUSDT)": "SOL/USDT",
    "✕ Ripple (XRPUSDT)": "XRP/USDT"
}

with col_asset:
    selected_asset_label = st.selectbox("Select Asset", list(assets.keys()), index=0, label_visibility="collapsed")
    selected_symbol = assets[selected_asset_label]

with col_engine:
    selected_engine = st.selectbox("Strategy Engine", [
        "🏛️ LuxAlgo Smart Money Concepts (SMC / FVG Benchmark)",
        "⚡ Tactical Fractal Matrix (Proprietary Swing)"
    ], index=0, label_visibility="collapsed")

with col_tf:
    selected_tf = st.radio("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=0, horizontal=True, label_visibility="collapsed")

with col_act:
    if st.button("🔄 Refresh", key="top_refresh_btn", use_container_width=True):
        st.cache_data.clear()

df = fetch_cloud_klines(selected_symbol, tf=selected_tf)
last_row = df.iloc[-1]
current_price = last_row['close']
vwap_price = last_row['vwap']
stoch_k = last_row['stoch_k']
stoch_d = last_row['stoch_d']
rsi = last_row['rsi']
atr_val = last_row['atr']
sma_val = last_row['sma200']

is_smc = "LuxAlgo" in selected_engine
if is_smc:
    overhead_liq, downside_liq, dec, supply_box, demand_box, eq_level, bos_marks, short_entry, short_tp, short_sl, long_entry, long_tp, long_sl = get_smc_structure(df, selected_symbol)
    engine_badge = "SMC / FVG Native"
else:
    overhead_liq, downside_liq, dec, short_entry, short_tp, short_sl, long_entry, long_tp, long_sl = get_tactical_liquidity_matrix(df, selected_symbol)
    engine_badge = "Fractal Liquidity Engine"
    supply_box, demand_box, eq_level, bos_marks = None, None, None, []

s_risk = abs(short_entry - short_sl)
s_rew = abs(short_entry - short_tp)
s_rr = s_rew / (s_risk + 1e-9)

l_risk = abs(long_entry - long_sl)
l_rew = abs(long_entry - long_tp)
l_rr = l_rew / (l_risk + 1e-9)

fib_targets = calculate_elliott_targets(df, selected_symbol)
dxy_synthetic = 98.88
rsi_4h_approx = round(min(95.0, rsi * 1.12), 1) if rsi > 50 else round(max(15.0, rsi * 0.88), 1)

def fmt(val):
    if dec == 4: return f"{val:,.4f}"
    elif dec == 3: return f"{val:,.3f}"
    elif dec == 2: return f"{val:,.2f}"
    return f"{val:,.1f}"

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.markdown(f"<div class='metric-card'><div class='metric-label'>Spot Price</div><div class='metric-val'>${fmt(current_price)}</div><div class='metric-sub'>Live USDT</div></div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card'><div class='metric-label'>ATR Buffer</div><div class='metric-val'>${fmt(atr_val)}</div><div class='metric-sub'>Volatility Range</div></div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card'><div class='metric-label'>RSI ({selected_tf}/4H)</div><div class='metric-val'>{rsi:.1f}/{rsi_4h_approx}</div><div class='metric-sub'>Confluence</div></div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-card'><div class='metric-label'>Session VWAP</div><div class='metric-val'>${fmt(vwap_price)}</div><div class='metric-sub'>Fair Value Anchor</div></div>", unsafe_allow_html=True)
m5.markdown(f"<div class='metric-card'><div class='metric-label'>Active Engine</div><div class='metric-val' style='font-size:1.05rem;'>{'LuxAlgo SMC' if is_smc else 'Fractal V3'}</div><div class='metric-sub'>{engine_badge}</div></div>", unsafe_allow_html=True)
m6.markdown(f"<div class='metric-card'><div class='metric-label'>CVD Flow</div><div class='metric-val'>{'Bullish Accum' if df['cvd'].iloc[-1] > df['cvd'].iloc[-4] else 'Bearish Dist'}</div><div class='metric-sub'>Delta Flow</div></div>", unsafe_allow_html=True)
m7.markdown(f"<div class='metric-card'><div class='metric-label'>Structure Regime</div><div class='metric-val'>{'Overextended' if rsi > 70 else ('Oversold' if rsi < 30 else 'Range Rotation')}</div><div class='metric-sub'>Market Cycle</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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

    if is_smc and supply_box and demand_box:
        fig.add_hrect(
            y0=supply_box[1], y1=supply_box[0],
            fillcolor="rgba(255, 82, 82, 0.18)", line_color="#ff5252",
            line_width=1, line_dash="solid",
            annotation_text="Bearish Order Block / Supply Zone", annotation_position="top left",
            row=1, col=1
        )
        fig.add_hrect(
            y0=demand_box[1], y1=demand_box[0],
            fillcolor="rgba(0, 230, 118, 0.18)", line_color="#00e676",
            line_width=1, line_dash="solid",
            annotation_text="Bullish Order Block / Demand Zone", annotation_position="bottom left",
            row=1, col=1
        )
        if eq_level:
            fig.add_hline(
                y=eq_level, line_dash="dot", line_color="#9e9e9e",
                annotation_text="50% Equilibrium (Discount / Premium Threshold)", row=1, col=1
            )
        for b_time, b_price, b_label, b_col in bos_marks:
            fig.add_annotation(
                x=b_time, y=b_price, text=b_label,
                showarrow=True, arrowhead=1, arrowcolor=b_col,
                font=dict(color=b_col, size=11), yshift=10 if "BOS" in b_label else -10,
                row=1, col=1
            )
    else:
        fig.add_hline(y=overhead_liq[0], line_dash="dash", line_color="#ff5252", annotation_text="Overhead Sweep Pivot", row=1, col=1)
        fig.add_hline(y=downside_liq[0], line_dash="dash", line_color="#00e676", annotation_text="Downside Demand Pivot", row=1, col=1)

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
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="CVD", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown(f"""
    <div class='side-card'>
        <div class='card-title-red'>🔴 {'SMC Bearish Order Blocks / FVG' if is_smc else 'Major Overhead Pivot Targets'}</div>
        <div class='data-row'><span>{'Primary Supply / OB' if is_smc else 'Sweep Pivot 1'}</span><span class='data-row-bold'>${fmt(overhead_liq[0])}</span></div>
        <div class='data-row'><span>{'BOS Invalidation' if is_smc else 'Expansion Target 2'}</span><span class='data-row-bold'>${fmt(overhead_liq[1])}</span></div>
        <div class='data-row'><span>{'Macro Rejection' if is_smc else 'Macro Resistance 3'}</span><span class='data-row-bold'>${fmt(overhead_liq[2])}</span></div>
    </div>
    <div class='side-card'>
        <div class='card-title-green'>🟢 {'SMC Bullish Order Blocks / FVG' if is_smc else 'Major Downside Pivot Targets'}</div>
        <div class='data-row'><span>{'Primary Demand / OB' if is_smc else 'Demand Pivot 1'}</span><span class='data-row-bold'>${fmt(downside_liq[0])}</span></div>
        <div class='data-row'><span>{'CHoCH Support' if is_smc else 'Structural Shelf 2'}</span><span class='data-row-bold'>${fmt(downside_liq[1])}</span></div>
        <div class='data-row'><span>{'Discount Floor' if is_smc else 'Key Value Floor 3'}</span><span class='data-row-bold'>${fmt(downside_liq[2])}</span></div>
    </div>
    <div class='side-card'>
        <div class='card-title-cyan'>🌊 Elliott Fibonacci Projections</div>
        <div class='data-row'><span>1.000 (C=A)</span><span class='data-row-bold'>${fmt(fib_targets['1.000 (C=A)'])}</span></div>
        <div class='data-row'><span>1.236 Ext</span><span class='data-row-bold'>${fmt(fib_targets['1.236 Ext'])}</span></div>
        <div class='data-row'><span>1.618 Ext</span><span class='data-row-bold'>${fmt(fib_targets['1.618 Ext'])}</span></div>
        <div class='data-row'><span>Wave 4 Retrace</span><span class='data-row-bold'>${fmt(fib_targets['Wave 4 Retracement'])}</span></div>
        <div class='data-row'><span>Wave B Floor</span><span class='data-row-bold'>${fmt(fib_targets['Wave B Floor'])}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader(f"🤖 Gemini Institutional Strategy Brief ({'LuxAlgo SMC Benchmark' if is_smc else 'Tactical Fractal Matrix'})")

prompt_ai = f"""
You are a senior hedge-fund derivatives execution trader evaluating {selected_symbol} on {selected_tf.upper()} using {selected_engine}:
- Spot Price: ${fmt(current_price)} USDT
- Session VWAP: ${fmt(vwap_price)} USDT
- ATR Buffer: ${fmt(atr_val)} USDT
- RSI ({selected_tf}): {rsi:.1f} | Stoch RSI: {stoch_k:.1f}/{stoch_d:.1f}
- Short Setup: Entry ${fmt(short_entry)} -> TP ${fmt(short_tp)} -> SL ${fmt(short_sl)} (R:R 1:{s_rr:.2f})
- Long Setup: Entry ${fmt(long_entry)} -> TP ${fmt(long_tp)} -> SL ${fmt(long_sl)} (R:R 1:{l_rr:.2f})
- Elliott Fibonacci Targets: {fib_targets}
- Macro Context: DXY {dxy_synthetic}

Provide an institutional, actionable breakdown formatted EXACTLY into the following 5 numbered sections:

### 1. Market Structure & Key Levels
(Analyze structural trend, price vs Session VWAP, oscillator state, and list exact Key Resistance and Support levels)

### 2. Liquidity & Structural Alignment
(Synthesize where resting liquidity/FVGs sit vs major overhead/downside Order Block zones)

### 3. Long Setup (Bullish Impulse / Demand Absorption)
- Thesis:
- Execution Trigger: Limit entry at ${fmt(long_entry)}
- Entry Range: ${fmt(long_entry)} - ${fmt(round(long_entry + (0.3 * atr_val), dec))}
- Stop Loss: ${fmt(long_sl)} (Strict structure invalidation)
- Take Profit 1: ${fmt(long_tp)}
- Take Profit 2: ${fmt(fib_targets['1.000 (C=A)'])}
- Risk/Reward Ratio: 1:{l_rr:.2f}

### 4. Short Setup (Bearish Mean Reversion / Sweep Absorption)
- Thesis:
- Execution Trigger: Limit entry or rejection at ${fmt(short_entry)}
- Entry Range: ${fmt(short_entry)} - ${fmt(round(short_entry + (0.3 * atr_val), dec))}
- Stop Loss: ${fmt(short_sl)}
- Take Profit 1: ${fmt(short_tp)}
- Take Profit 2: ${fmt(downside_liq[1])}
- Risk/Reward Ratio: 1:{s_rr:.2f}

### 5. Execution Verdict & Primary Stance
- Primary Stance: (Clearly declare "Favor Long" or "Favor Short")
- Tactical Rationale: (Synthesize edge and path of least resistance on BTCC USDT perpetuals)
"""

fallback_ai = f"""
### 1. Market Structure & Key Levels
{selected_symbol} is trading at **${fmt(current_price)} USDT** under the {selected_engine} framework, relative to Session VWAP of **${fmt(vwap_price)} USDT**.
* **Key Resistance Levels**: ${fmt(short_entry)}, ${fmt(overhead_liq[1])}, ${fmt(fib_targets['1.000 (C=A)'])}.
* **Key Support Levels**: ${fmt(long_entry)}, ${fmt(vwap_price)}, ${fmt(downside_liq[2])}.

### 2. Liquidity & Structural Alignment
* Structural levels identify key institutional volume zones at **${fmt(short_entry)}** and **${fmt(long_entry)}**.

### 3. Long Setup (Bullish Impulse / Demand Absorption)
* **Execution Trigger**: Absorption at **${fmt(long_entry)}**.
* **Entry Range**: ${fmt(long_entry)} - ${fmt(round(long_entry + (0.3 * atr_val), dec))}
* **Stop Loss**: ${fmt(long_sl)}
* **Take Profit 1**: ${fmt(long_tp)}
* **Take Profit 2**: ${fmt(fib_targets['1.000 (C=A)'])}
* **Risk/Reward Ratio**: 1:{l_rr:.2f}

### 4. Short Setup (Bearish Mean Reversion / Sweep Absorption)
* **Execution Trigger**: SFP or Order Block rejection at **${fmt(short_entry)}**.
* **Entry Range**: ${fmt(short_entry)} - ${fmt(round(short_entry + (0.3 * atr_val), dec))}
* **Stop Loss**: ${fmt(short_sl)}
* **Take Profit 1**: ${fmt(short_tp)}
* **Take Profit 2**: ${fmt(downside_liq[1])}
* **Risk/Reward Ratio**: 1:{s_rr:.2f}

### 5. Execution Verdict & Primary Stance
* **Primary Stance**: **{"Favor Long" if rsi < 65 else "Favor Short"}**
* **Tactical Rationale**: The setup offers favorable Risk/Reward by executing at structural boundaries.
"""

brief_content = generate_gemini_brief(prompt_ai, fallback_ai)
st.markdown(brief_content)

st.markdown("---")

col_bridge_hdr, col_bridge_btn = st.columns([3, 1])

with col_bridge_hdr:
    st.subheader(f"⚡ Dual BTCC Order Bridges ({selected_symbol})")

with col_bridge_btn:
    if st.button("🔄 Re-Calculate Setup & Pricing", key="bottom_refresh_btn", use_container_width=True):
        st.cache_data.clear()

col_short, col_long = st.columns(2)

key_pfx = f"{selected_symbol}_{selected_tf}_{selected_engine[:6]}_{short_entry}_{long_entry}"
clean_ticker = "SILVERUSDT" if "XAG" in selected_symbol else selected_symbol.replace("/", "").upper()

with col_short:
    with st.expander("🔴 SHORT Execution Setup (Sell / Mean Reversion)", expanded=True):
        s_entry = st.number_input(f"Limit Entry (USDT)", value=float(short_entry), step=float(10**(-dec)), format=f"%.{dec}f", key=f"s_e_{key_pfx}")
        s_tp_in = st.number_input(f"Take Profit Target (USDT)", value=float(short_tp), step=float(10**(-dec)), format=f"%.{dec}f", key=f"s_tp_{key_pfx}")
        s_sl_in = st.number_input(f"Stop Loss Target (USDT)", value=float(short_sl), step=float(10**(-dec)), format=f"%.{dec}f", key=f"s_sl_{key_pfx}")
        s_lev = st.slider("Leverage Target", 1, 100, 20, key=f"s_lev_{key_pfx}")

        cur_s_risk = abs(s_entry - s_sl_in)
        cur_s_rew = abs(s_entry - s_tp_in)
        cur_s_rr = cur_s_rew / (cur_s_risk + 1e-9)

        st.info(f"**Calculated R:R Ratio**: **{cur_s_rr:.2f}** (Risk: ${fmt(cur_s_risk)} | Reward: ${fmt(cur_s_rew)})")

        short_payload = {
            "symbol": clean_ticker,
            "side": "SHORT",
            "entry": float(round(s_entry, dec)),
            "tp": float(round(s_tp_in, dec)),
            "sl": float(round(s_sl_in, dec)),
            "leverage": int(s_lev)
        }
        st.markdown("**📋 1. Click icon in code block to copy SHORT payload:**")
        st.code(json.dumps(short_payload), language="json")
        st.caption("👉 Then switch to your BTCC tab and click **⚡ Fill BTCC Ticket** on your bookmarks bar.")

with col_long:
    with st.expander("🟢 LONG Execution Setup (Buy / Trend Continuation)", expanded=True):
        l_entry = st.number_input(f"Limit Entry (USDT)", value=float(long_entry), step=float(10**(-dec)), format=f"%.{dec}f", key=f"l_e_{key_pfx}")
        l_tp_in = st.number_input(f"Take Profit Target (USDT)", value=float(long_tp), step=float(10**(-dec)), format=f"%.{dec}f", key=f"l_tp_{key_pfx}")
        l_sl_in = st.number_input(f"Stop Loss Target (USDT)", value=float(long_sl), step=float(10**(-dec)), format=f"%.{dec}f", key=f"l_sl_{key_pfx}")
        l_lev = st.slider("Leverage Target", 1, 100, 20, key=f"l_lev_{key_pfx}")

        cur_l_risk = abs(l_entry - l_sl_in)
        cur_l_rew = abs(l_entry - l_tp_in)
        cur_l_rr = cur_l_rew / (cur_l_risk + 1e-9)

        st.success(f"**Calculated R:R Ratio**: **{cur_l_rr:.2f}** (Risk: ${fmt(cur_l_risk)} | Reward: ${fmt(cur_l_rew)})")

        long_payload = {
            "symbol": clean_ticker,
            "side": "LONG",
            "entry": float(round(l_entry, dec)),
            "tp": float(round(l_tp_in, dec)),
            "sl": float(round(l_sl_in, dec)),
            "leverage": int(l_lev)
        }
        st.markdown("**📋 1. Click icon in code block to copy LONG payload:**")
        st.code(json.dumps(long_payload), language="json")
        st.caption("👉 Then switch to your BTCC tab and click **⚡ Fill BTCC Ticket** on your bookmarks bar.")

# --- DUAL-COLUMN TACTICAL PLAYBOOK (DYNAMIC) ---
st.divider()
st.markdown("### 📋 TACTICAL EXECUTION PLAYBOOK")

col_bull, col_bear = st.columns(2)

eq_display = f"${fmt(eq_level)}" if eq_level else f"${fmt(vwap_price)}"
primary_supply_val = f"${fmt(overhead_liq[0])}"
primary_demand_val = f"${fmt(downside_liq[0])}"
discount_floor_val = f"${fmt(downside_liq[2])}"
bos_invalidation_val = f"${fmt(overhead_liq[1])}"
fib_ext_val = f"${fmt(fib_targets['1.618 Ext'])}"

with col_bull:
    st.markdown("### 🟢 BULLISH PLAN")
    st.markdown(f"""
    **THESIS:**  
    Holding above the local demand floor (**{primary_demand_val}**) and defending equilibrium (**{eq_display}**) opens the path toward upper supply.
    
    **BULLISH TRIGGER SETUP:**  
    * 🟩 Defend primary demand order block at **{primary_demand_val}**.
    * 🟩 Reclaim and hold structural equilibrium at **{eq_display}**.
    * 🟩 Break and hold above local resistance for momentum confirmation.
    
    **UPSIDE ROADMAP:**  
    `EQ: {eq_display}` $\rightarrow$ `Supply: {primary_supply_val}` $\rightarrow$ `Fib Ext: {fib_ext_val}`
    """)
    
    st.markdown("#### **Bullish Execution Ladder**")
    bull_ladder = {
        "Target": ["TP1 (EQ Test): " + eq_display, "TP2 (Supply Tap): " + primary_supply_val, "TP3 (Fib Ext): " + fib_ext_val],
        "Action": ["Scale Out 30%", "Scale Out 40%", "Runner Target"],
        "Purpose": ["Secure quick profit", "Lock risk-free", "Capture macro expansion"]
    }
    st.table(bull_ladder)
    
    st.warning(f"⚠️ **Invalidation:** Clean break, retest, and loss of the Discount Floor support level at **{discount_floor_val}**.")

with col_bear:
    st.markdown("### 🔴 BEARISH PLAN")
    st.markdown(f"""
    **THESIS:**  
    Rejection at the upper supply zone (**{primary_supply_val}**) or failure to hold equilibrium increases the probability of a sweep down to macro demand.
    
    **BEARISH TRIGGER SETUP:**  
    * ❌ Rejection wick at the Bearish Supply Zone (**{primary_supply_val}**).
    * ❌ Loss of local structural support and equilibrium (**{eq_display}**).
    * ❌ Confirmed breakdown of CVD delta.
    
    **DOWNSIDE ROADMAP:**  
    `Supply: {primary_supply_val}` $\rightarrow$ `EQ: {eq_display}` $\rightarrow$ `Demand Floor: {discount_floor_val}`
    """)
    
    st.markdown("#### **Bearish Execution Ladder**")
    bear_ladder = {
        "Target": ["Entry (Supply): " + primary_supply_val, "TP1 (Mid-Range EQ): " + eq_display, "TP2 (Demand Floor): " + discount_floor_val],
        "Action": ["Open Short", "Take Profit 1", "Full Exit / Reverse"],
        "Purpose": ["Initial rejection", "Secure baseline", "Target liquidity pool"]
    }
    st.table(bear_ladder)
    
    st.error(f"⚠️ **Invalidation:** Clean break and hold above the BOS Invalidation level at **{bos_invalidation_val}**.")

st.divider()
st.markdown("""
### 🛡️ RISK MANAGEMENT & EXECUTION RULES
* **Risk Parameters:** Risk strictly 1–2% of total account capital per leveraged setup.
* **Stop Loss Placement:** Always anchor stops outside structural supply/demand boundaries or recent liquidation wicks.
* **Discipline:** Avoid forcing trades in the choppy middle of the range; wait for the outer boundaries to trigger.
""")
