import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
import plotly.graph_objects as go
import pyperclip
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from google import genai

# --- Page Layout Configuration ---
st.set_page_config(
    page_title="Institutional Derivatives Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Institutional CSS styling
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #13171f;
        border: 1px solid #23272e;
        border-radius: 8px;
        padding: 10px 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.82rem !important;
        color: #8b949e !important;
        margin-bottom: 2px !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.80rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Secure API Configuration (Cloud Secrets + Local Fallback)
def get_secret(key, default_val):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return default_val

GEMINI_API_KEY = get_secret("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
TELEGRAM_BOT_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID", ""))

# Multi-Tier Resilient AI Strategy Generator (Native REST Fallback)
def generate_strategy_brief(prompt):
    models_to_try = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite"
    ]
    
    # 1. Try modern google-genai client
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

    # 2. Resilient Direct REST API fallback for AQ. / AIza keys
    for m in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
            payload_data = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            req_body = json.dumps(payload_data).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_body,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                continue
            return f"API Notice: {e}"

    return "Market brief temporarily delayed due to API traffic. Click Refresh in 15 seconds."

# Telegram Utility
def send_telegram_alert(text):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({'chat_id': TELEGRAM_CHAT_ID, 'text': text}).encode()
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, "Alert sent successfully!"
    except Exception as e:
        return False, str(e)

# --- Pattern Detection Utility ---
def detect_high_confluence_patterns(df, vwap, key_zones=[], is_crypto=True):
    if df is None or len(df) < 20:
        return "Awaiting Data"
    
    close_col = 'close' if is_crypto else 'Close'
    open_col = 'open' if is_crypto else 'Open'
    high_col = 'high' if is_crypto else 'High'
    low_col = 'low' if is_crypto else 'Low'
    vol_col = 'volume' if is_crypto else 'Volume'

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        avg_vol = df[vol_col].tail(20).mean() if vol_col in df.columns else 0
        
        body = max(abs(last[close_col] - last[open_col]), 0.0001)
        upper_wick = last[high_col] - max(last[close_col], last[open_col])
        lower_wick = min(last[close_col], last[open_col]) - last[low_col]
        high_volume = (last[vol_col] > (avg_vol * 1.15)) if avg_vol > 0 else False
        
        near_key_level = abs(last[close_col] - vwap) / vwap < 0.0035 if vwap > 0 else False
        for zone in key_zones:
            if abs(last[close_col] - zone) / zone < 0.0035:
                near_key_level = True

        pattern = "Range Rotation"
        if lower_wick >= (2 * body) and upper_wick <= (0.5 * body):
            if near_key_level or high_volume:
                pattern = "Bullish Pin Bar"
        elif upper_wick >= (2 * body) and lower_wick <= (0.5 * body):
            if near_key_level or high_volume:
                pattern = "Bearish Pin Bar"
        elif last[close_col] > last[open_col] and prev[close_col] < prev[open_col] and last[close_col] > prev[open_col] and last[open_col] < prev[close_col]:
            if near_key_level or high_volume:
                pattern = "Bullish Engulfing"
        elif last[close_col] < last[open_col] and prev[close_col] > prev[open_col] and last[close_col] < prev[low_col] and last[open_col] > prev[high_col]:
            if near_key_level or high_volume:
                pattern = "Bearish Engulfing"

        return pattern
    except Exception:
        return "Range Rotation"

# --- Elliott Wave Calculation Engine ---
def calculate_elliott_abc(df, is_crypto=True, round_decimals=2):
    high_col = 'high' if is_crypto else 'High'
    low_col = 'low' if is_crypto else 'Low'

    if df is None or len(df) < 30:
        return {"summary": "Awaiting candle data", "pivots": [], "levels": {}, "table": pd.DataFrame()}

    try:
        window = 3
        highs = df[high_col].values
        lows = df[low_col].values
        times = df.index.values
        
        pivots = []
        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i - window : i + window + 1]):
                pivots.append(('High', highs[i], times[i]))
            elif lows[i] == min(lows[i - window : i + window + 1]):
                pivots.append(('Low', lows[i], times[i]))

        filtered_pivots = []
        for p in pivots:
            if not filtered_pivots or filtered_pivots[-1][0] != p[0]:
                filtered_pivots.append(p)
            else:
                if p[0] == 'High' and p[1] > filtered_pivots[-1][1]:
                    filtered_pivots[-1] = p
                elif p[0] == 'Low' and p[1] < filtered_pivots[-1][1]:
                    filtered_pivots[-1] = p

        if len(filtered_pivots) < 3:
            return {"summary": "Awaiting clear structural pivot formation", "pivots": [], "levels": {}, "table": pd.DataFrame()}

        p0, p1, p2 = filtered_pivots[-3], filtered_pivots[-2], filtered_pivots[-1]
        
        if p0[0] == 'High' and p1[0] == 'Low' and p2[0] == 'High':
            origin, wave_a, wave_b = p0[1], p1[1], p2[1]
            wave_a_dist = origin - wave_a
            c_1000 = wave_b - (wave_a_dist * 1.000)
            c_1236 = wave_b - (wave_a_dist * 1.236)
            c_1618 = wave_b - (wave_a_dist * 1.618)
            structure = "Bearish A-B-C (Downside Wave C Targets)"
        else:
            origin, wave_a, wave_b = p0[1], p1[1], p2[1]
            wave_a_dist = wave_a - origin
            c_1000 = wave_b + (wave_a_dist * 1.000)
            c_1236 = wave_b + (wave_a_dist * 1.236)
            c_1618 = wave_b + (wave_a_dist * 1.618)
            structure = "Bullish A-B-C (Upside Wave C Targets)"

        round_val = max(round_decimals, 0)
        fib_table = pd.DataFrame([
            {"Ratio": "1.000 (C=A)", "Target": f"${round(c_1000, round_val):,.{round_val}f}"},
            {"Ratio": "1.236 Ext", "Target": f"${round(c_1236, round_val):,.{round_val}f}"},
            {"Ratio": "1.618 Ext", "Target": f"${round(c_1618, round_val):,.{round_val}f}"}
        ])

        summary = (
            f"{structure} | Origin: ${origin:,.{round_val}f} -> "
            f"Wave A: ${wave_a:,.{round_val}f} -> Wave B: ${wave_b:,.{round_val}f} | "
            f"1.000 Target: ${c_1000:,.{round_val}f} | 1.618 Target: ${c_1618:,.{round_val}f}"
        )

        return {
            "summary": summary,
            "pivots": [p0, p1, p2],
            "levels": {"1.000": c_1000, "1.236": c_1236, "1.618": c_1618},
            "table": fib_table
        }
    except Exception:
        return {"summary": "Pivot calculation pending", "pivots": [], "levels": {}, "table": pd.DataFrame()}

# --- Clean Ray Chart Engine ---
def build_pro_chart(df, vwap, long_liqs_raw, short_liqs_raw, elliott_dict, is_crypto=True, symbol_label="BTC/USDT"):
    open_c = 'open' if is_crypto else 'Open'
    high_c = 'high' if is_crypto else 'High'
    low_c = 'low' if is_crypto else 'Low'
    close_c = 'close' if is_crypto else 'Close'

    plot_df = df.tail(40).copy()
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df[open_c],
        high=plot_df[high_c],
        low=plot_df[low_c],
        close=plot_df[close_c],
        name="1H Candle",
        increasing=dict(line=dict(color='#089981', width=1.5), fillcolor='#089981'),
        decreasing=dict(line=dict(color='#f23645', width=1.5), fillcolor='#f23645'),
        hoverinfo='x+y'
    ))

    ray_start = plot_df.index[-15]
    ray_end = plot_df.index[-1]
    if vwap > 0:
        fig.add_trace(go.Scatter(
            x=[ray_start, ray_end],
            y=[vwap, vwap],
            mode="lines+text",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            text=["", f" VWAP ${vwap:,.2f}"],
            textposition="middle right",
            name="Session VWAP",
            hoverinfo="name"
        ))

    for i, val in enumerate(short_liqs_raw[:2]):
        fig.add_trace(go.Scatter(
            x=[ray_start, ray_end],
            y=[val, val],
            mode="lines+text",
            line=dict(color="rgba(239, 68, 68, 0.8)", width=1.5, dash="dot"),
            text=["", f" Short Liq ${val:,.2f}"],
            textposition="middle right",
            name=f"Overhead Liq {i+1}",
            hoverinfo="name"
        ))

    for i, val in enumerate(long_liqs_raw[:2]):
        fig.add_trace(go.Scatter(
            x=[ray_start, ray_end],
            y=[val, val],
            mode="lines+text",
            line=dict(color="rgba(16, 185, 129, 0.8)", width=1.5, dash="dot"),
            text=["", f" Long Liq ${val:,.2f}"],
            textposition="middle right",
            name=f"Downside Liq {i+1}",
            hoverinfo="name"
        ))

    fibs = elliott_dict.get('levels', {})
    if '1.000' in fibs:
        fig.add_trace(go.Scatter(
            x=[ray_start, ray_end],
            y=[fibs['1.000'], fibs['1.000']],
            mode="lines+text",
            line=dict(color="#38bdf8", width=1.8, dash="dashdot"),
            text=["", f" Wave C ${fibs['1.000']:,.2f}"],
            textposition="middle right",
            name="Elliott 1.000 Target",
            hoverinfo="name"
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111418",
        plot_bgcolor="#111418",
        height=460,
        margin=dict(l=10, r=120, t=10, b=10),
        xaxis=dict(rangeslider=dict(visible=False), showgrid=True, gridcolor='#1e242c', type="date"),
        yaxis=dict(showgrid=True, gridcolor='#1e242c', side='right', tickformat="$,.2f" if is_crypto else "$,.3f"),
        showlegend=False,
        hovermode="x unified"
    )
    return fig

# --- Data Engines ---
def get_crypto_data(symbol="BTC/USDT", round_decimals=2):
    exchange = ccxt.kraken({'enableRateLimit': True})
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']

    df_15m = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe='15m', limit=35), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_1h = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_4h = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe='4h', limit=210), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_1d = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe='1d', limit=210), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    for d in [df_15m, df_1h, df_4h, df_1d]:
        d['timestamp'] = pd.to_datetime(d['timestamp'], unit='ms')
        d.set_index('timestamp', inplace=True)

    df_4h['tr'] = df_4h[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1)
    atr_4h = df_4h['tr'].tail(14).mean()

    def calc_rsi(s):
        d = s.diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        return 100 - (100 / (1 + (g / l)))

    rsi_15m = calc_rsi(df_15m['close']).iloc[-1]
    rsi_4h = calc_rsi(df_4h['close']).iloc[-1]
    ma_200_daily = df_1d['close'].tail(200).mean() if len(df_1d) >= 200 else df_1d['close'].mean()
    ma_200_4h = df_4h['close'].tail(200).mean() if len(df_4h) >= 200 else df_4h['close'].mean()

    df_15m['tp'] = (df_15m['high'] + df_15m['low'] + df_15m['close']) / 3
    vwap = (df_15m['tp'] * df_15m['volume']).sum() / df_15m['volume'].sum()

    macro = yf.Tickers("DX-Y.NYB ES=F")
    df_dxy = macro.tickers["DX-Y.NYB"].history(period="2d", interval="1d")
    df_es = macro.tickers["ES=F"].history(period="2d", interval="1d")
    dxy_val = df_dxy['Close'].iloc[-1] if not df_dxy.empty else 99.0
    dxy_chg = ((dxy_val - df_dxy['Open'].iloc[-1]) / df_dxy['Open'].iloc[-1]) * 100 if not df_dxy.empty else 0.0
    es_val = df_es['Close'].iloc[-1] if not df_es.empty else 0

    long_liqs_raw, short_liqs_raw = [], []
    for _, row in df_1h.iterrows():
        p, vol = row['close'], row['volume']
        for lev in [25, 50, 100]:
            l_zone = p * (1 - (1/lev) + 0.004)
            s_zone = p * (1 + (1/lev) - 0.004)
            if l_zone < current_price: long_liqs_raw.append(l_zone)
            if s_zone > current_price: short_liqs_raw.append(s_zone)

    top_long_raw = sorted(long_liqs_raw, reverse=True)[:3] if long_liqs_raw else []
    top_short_raw = sorted(short_liqs_raw)[:3] if short_liqs_raw else []

    pattern = detect_high_confluence_patterns(df_15m, vwap, top_long_raw + top_short_raw, is_crypto=True)
    elliott_data = calculate_elliott_abc(df_1h, is_crypto=True, round_decimals=round_decimals)

    round_val = max(round_decimals, 0)
    df_l_liq = pd.DataFrame([{"Target": f"${round(x, round_val):,.{round_val}f}"} for x in top_long_raw])
    df_s_liq = pd.DataFrame([{"Target": f"${round(x, round_val):,.{round_val}f}"} for x in top_short_raw])

    payload = f"""
--- REAL-TIME {symbol} MARKET SNAPSHOT ---
- Spot Price: ${current_price:,.{max(round_decimals, 2)}f} | 24h Range: ${ticker['low']:,.{max(round_decimals, 2)}f} - ${ticker['high']:,.{max(round_decimals, 2)}f}
- Session VWAP: ${vwap:,.{max(round_decimals, 2)}f} | 4H ATR: ${atr_4h:,.{max(round_decimals, 2)}f}
- RSI: 15m={rsi_15m:.1f} | 4H={rsi_4h:.1f}
- 200 SMA: Daily=${ma_200_daily:,.{max(round_decimals, 2)}f} | 4H=${ma_200_4h:,.{max(round_decimals, 2)}f}
- 15m Pattern: {pattern}
- Elliott Wave Structure: {elliott_data['summary']}
- Macro: DXY={dxy_val:.2f} ({dxy_chg:+.2f}%) | ES Futures={es_val:,.2f}
- Downside Long Liquidation Pools: {", ".join([f"${x:,.2f}" for x in top_long_raw]) if top_long_raw else 'None'}
- Overhead Short Liquidation Pools: {", ".join([f"${x:,.2f}" for x in top_short_raw]) if top_short_raw else 'None'}
"""
    return {
        'price': current_price, 'vwap': vwap, 'atr': atr_4h,
        'rsi_15m': rsi_15m, 'rsi_4h': rsi_4h, 'dxy': dxy_val,
        'dxy_chg': dxy_chg, 'es': es_val, 'ma_200': ma_200_daily,
        'pattern': pattern, 'long_raw': top_long_raw, 'short_raw': top_short_raw,
        'long_df': df_l_liq, 'short_df': df_s_liq,
        'elliott': elliott_data, 'df_1h': df_1h, 'payload': payload
    }

def get_silver_data():
    tickers = yf.Tickers("SI=F GC=F DX-Y.NYB")
    
    df_15m = tickers.tickers["SI=F"].history(period="5d", interval="15m")
    df_1h = tickers.tickers["SI=F"].history(period="14d", interval="1h")
    df_1d = tickers.tickers["SI=F"].history(period="250d", interval="1d")
    
    if df_1d.empty:
        df_1d = yf.Ticker("SLV").history(period="250d", interval="1d")
    
    silver_price = df_15m['Close'].iloc[-1] if not df_15m.empty else (df_1d['Close'].iloc[-1] if not df_1d.empty else 67.25)
    
    df_gold = tickers.tickers["GC=F"].history(period="5d", interval="1d")
    gold_price = df_gold['Close'].iloc[-1] if not df_gold.empty else 4580.00
    gsr = gold_price / silver_price if silver_price > 0 else 68.0
    
    df_dxy = tickers.tickers["DX-Y.NYB"].history(period="5d", interval="1d")
    dxy_val = df_dxy['Close'].iloc[-1] if not df_dxy.empty else 98.75
    dxy_chg = ((dxy_val - df_dxy['Open'].iloc[-1]) / df_dxy['Open'].iloc[-1]) * 100 if not df_dxy.empty else 0.0

    if not df_1d.empty:
        df_1d['tr'] = df_1d[['High', 'Low', 'Close']].apply(lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Close']), abs(x['Low'] - x['Close'])), axis=1)
        daily_atr = df_1d['tr'].tail(14).mean()
        ma_200_daily = df_1d['Close'].tail(200).mean() if len(df_1d) >= 200 else df_1d['Close'].mean()
    else:
        daily_atr = 0.85
        ma_200_daily = 73.50

    def calc_rsi(s):
        if len(s) < 15: return 50.0
        d = s.diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        l = (-d.where(d < 0, 0)).rolling(14).mean()
        return 100 - (100 / (1 + (g / l)))

    rsi_15m = calc_rsi(df_15m['Close']).iloc[-1] if not df_15m.empty else 50.0
    rsi_1d = calc_rsi(df_1d['Close']).iloc[-1] if not df_1d.empty else 50.0

    if not df_15m.empty:
        df_15m['tp'] = (df_15m['High'] + df_15m['Low'] + df_15m['Close']) / 3
        valid_v = df_15m[df_15m['Volume'] > 0]
        vwap = (valid_v['tp'] * valid_v['Volume']).sum() / valid_v['Volume'].sum() if not valid_v.empty else silver_price
    else:
        vwap = silver_price

    long_liqs_raw, short_liqs_raw = [], []
    if not df_1h.empty:
        for _, row in df_1h.iterrows():
            p, vol = row['Close'], row['Volume']
            if vol > 0:
                for lev in [20, 35, 50]:
                    l_zone = p * (1 - (1/lev) + 0.005)
                    s_zone = p * (1 + (1/lev) - 0.005)
                    if l_zone < silver_price: long_liqs_raw.append(l_zone)
                    if s_zone > silver_price: short_liqs_raw.append(s_zone)

    top_long_raw = sorted(long_liqs_raw, reverse=True)[:3] if long_liqs_raw else []
    top_short_raw = sorted(short_liqs_raw)[:3] if short_liqs_raw else []

    pattern = detect_high_confluence_patterns(df_15m, vwap, top_long_raw + top_short_raw, is_crypto=False)
    elliott_data = calculate_elliott_abc(df_1h, is_crypto=False, round_decimals=3)

    df_l_liq = pd.DataFrame([{"Target": f"${x:,.3f}"} for x in top_long_raw])
    df_s_liq = pd.DataFrame([{"Target": f"${x:,.3f}"} for x in top_short_raw])

    payload = f"""
--- REAL-TIME SILVER (COMEX / SPOT) MARKET SNAPSHOT ---
- Spot Price: ${silver_price:,.3f} | Gold Price: ${gold_price:,.2f}
- Gold/Silver Ratio (GSR): {gsr:.2f} | DXY: {dxy_val:.2f} ({dxy_chg:+.2f}%)
- Session VWAP: ${vwap:,.3f} | Daily ATR: ${daily_atr:.3f}
- RSI: 15m={rsi_15m:.1f} | Daily={rsi_1d:.1f} | 200 SMA: ${ma_200_daily:,.3f}
- 15m Pattern: {pattern}
- Elliott Wave Structure: {elliott_data['summary']}
- Downside Long Liquidation Pools: {", ".join([f"${x:,.2f}" for x in top_long_raw]) if top_long_raw else 'None'}
- Overhead Short Liquidation Pools: {", ".join([f"${x:,.2f}" for x in top_short_raw]) if top_short_raw else 'None'}
"""
    return {
        'price': silver_price, 'gold': gold_price, 'gsr': gsr, 'vwap': vwap,
        'atr': daily_atr, 'rsi_15m': rsi_15m, 'rsi_1d': rsi_1d, 'dxy': dxy_val,
        'dxy_chg': dxy_chg, 'ma_200': ma_200_daily, 'pattern': pattern,
        'long_raw': top_long_raw, 'short_raw': top_short_raw,
        'long_df': df_l_liq, 'short_df': df_s_liq,
        'elliott': elliott_data, 'df_1h': df_1h, 'payload': payload
    }

# --- Dynamic AI Brief Parser ---
def parse_ai_trade_parameters(brief_text, default_data, is_crypto=True):
    lower_text = brief_text.lower()
    
    is_short = False
    verdict_chunk = lower_text.split("5. execution verdict")[1] if "5. execution verdict" in lower_text else lower_text
    
    if "favor short" in verdict_chunk or "tactical short" in verdict_chunk or "bearish" in verdict_chunk:
        is_short = True
    elif "favor long" in verdict_chunk or "tactical long" in verdict_chunk or "bullish" in verdict_chunk:
        is_short = False
    else:
        is_short = default_data['rsi_4h'] > 70 if is_crypto else False

    if is_short:
        if "4. short setup" in lower_text:
            section_raw = brief_text.split("4. Short Setup")[1]
            section_text = section_raw.split("5. Execution Verdict")[0] if "5. Execution Verdict" in section_raw else section_raw
        else:
            section_text = brief_text
    else:
        if "3. Long Setup" in brief_text:
            section_raw = brief_text.split("3. Long Setup")[1]
            section_text = section_raw.split("4. Short Setup")[0] if "4. Short Setup" in section_raw else section_raw
        else:
            section_text = brief_text

    def extract_clean_number(pattern, fallback):
        match = re.search(pattern, section_text, re.IGNORECASE)
        if match:
            clean = match.group(1).replace(",", "").replace("$", "").strip()
            try: return float(clean)
            except: pass
        return float(fallback)

    if is_short:
        def_entry = default_data['short_raw'][0] if default_data['short_raw'] else default_data['vwap'] + default_data['atr']
        def_sl = def_entry + (default_data['atr'] * 1.2)
        def_tp = default_data['long_raw'][0] if default_data['long_raw'] else default_data['vwap']
    else:
        def_entry = default_data['long_raw'][0] if default_data['long_raw'] else default_data['vwap']
        def_sl = def_entry - (default_data['atr'] * 1.2)
        def_tp = default_data['short_raw'][0] if default_data['short_raw'] else default_data['vwap'] + default_data['atr']

    entry = extract_clean_number(r"Entry (?:Range|Zone):\s*[\$]?([\d,\.]+)", def_entry)
    sl = extract_clean_number(r"Stop Loss:\s*[\$]?([\d,\.]+)", def_sl)
    tp = extract_clean_number(r"Take Profit 1:\s*[\$]?([\d,\.]+)", def_tp)

    return {
        "side": "SHORT (Sell)" if is_short else "LONG (Buy)",
        "is_short": is_short,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "bias_tag": "short" if is_short else "long"
    }

# --- 1-Click BTCC Dynamic Launcher Widget ---
def render_btcc_direct_launcher(symbol, trade_params, round_decimals=2, brief_text=""):
    st.markdown("---")
    st.subheader(f"⚡ 1-Click BTCC Order Bridge ({symbol})")
    
    btcc_ticker_map = {
        "BTC/USDT": "BTCUSDT",
        "XAG/USDT": "SILVERUSDT",
        "ETH/USDT": "ETHUSDT",
        "SOL/USDT": "SOLUSDT",
        "XRP/USDT": "XRPUSDT"
    }
    contract_code = btcc_ticker_map.get(symbol, symbol.replace("/", ""))
    bias_tag = trade_params.get("bias_tag", "auto")

    with st.expander(f"🎯 AI-Synchronized Order Parameters ({trade_params['side']})", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            side_idx = 1 if trade_params['is_short'] else 0
            side = st.selectbox("Position Side", ["LONG (Buy)", "SHORT (Sell)"], index=side_idx, key=f"side_{symbol}_{bias_tag}")
            leverage = st.slider("Leverage Target", min_value=1, max_value=50, value=20, step=1, key=f"lev_{symbol}_{bias_tag}")
        with col2:
            entry_price = st.number_input("Limit Entry ($)", value=float(trade_params['entry']), format=f"%.{round_decimals}f", key=f"entry_{symbol}_{bias_tag}")
        with col3:
            tp_price = st.number_input("Take Profit Target ($)", value=float(trade_params['tp']), format=f"%.{round_decimals}f", key=f"tp_{symbol}_{bias_tag}")
        with col4:
            sl_price = st.number_input("Stop Loss Target ($)", value=float(trade_params['sl']), format=f"%.{round_decimals}f", key=f"sl_{symbol}_{bias_tag}")

        # Live Mathematical Risk/Reward Calculation
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        rr_ratio = (reward / risk) if risk > 0 else 0.0

        st.caption(f"📊 **Live Calculated Risk/Reward:** `1 : {rr_ratio:.2f}` (Risk: ${risk:,.{round_decimals}f} | Reward: ${reward:,.{round_decimals}f})")

        payload = json.dumps({
            "symbol": contract_code,
            "side": "sell" if "SHORT" in side else "buy",
            "entry": f"{entry_price:.{round_decimals}f}",
            "tp": f"{tp_price:.{round_decimals}f}",
            "sl": f"{sl_price:.{round_decimals}f}",
            "leverage": leverage
        })

        st.write("")
        c1, c2, c3 = st.columns([1.5, 1.5, 1.5])
        with c1:
            if st.button(f"📋 1. Copy {side.split()[0]} Setup", key=f"btn_cp_{symbol}_{bias_tag}", use_container_width=True):
                pyperclip.copy(payload)
                st.toast(f"✅ Copied {symbol} setup! Now click 'Fill BTCC Ticket' bookmarklet.")
        with c2:
            btcc_url = f"https://www.btcc.com/en-US/trade/perpetual/{contract_code}"
            st.link_button(
                f"🚀 2. Open {contract_code} on BTCC",
                btcc_url,
                use_container_width=True
            )
        with c3:
            if st.button(f"📲 Send Setup to Telegram", key=f"btn_tg_{symbol}_{bias_tag}", use_container_width=True):
                msg = f"⚡ LIVE SETUP: {symbol}\nSide: {side}\nEntry: ${entry_price:.{round_decimals}f}\nTP: ${tp_price:.{round_decimals}f}\nSL: ${sl_price:.{round_decimals}f}\nLev: {leverage}x\nCalculated R:R: 1:{rr_ratio:.2f}\n\nBrief Snapshot:\n{brief_text[:400]}..."
                ok, res = send_telegram_alert(msg)
                if ok:
                    st.toast("📲 Alert delivered to your phone!")
                else:
                    st.error(f"Telegram error: {res}")

# --- Generic Tab Renderer ---
def render_crypto_tab(tab_name, ticker_symbol, session_key, round_dec=2):
    col_ctrl, _ = st.columns([1, 4])
    with col_ctrl:
        refresh = st.button(f"🔄 Refresh {ticker_symbol}", key=f"btn_{session_key}")

    data_key = f"{session_key}_data"
    brief_key = f"{session_key}_brief"

    if data_key not in st.session_state or refresh:
        with st.spinner(f"Fetching {ticker_symbol} market structure & computing visual zones..."):
            try:
                st.session_state[data_key] = get_crypto_data(ticker_symbol, round_decimals=round_dec)
                prompt = f"""You are a disciplined derivatives desk strategist. Analyze this live snapshot for {ticker_symbol}, including Elliott Wave projections and liquidation pools. 

Structure your response strictly into these 5 sections:
1. Market Structure & Key Levels
2. Liquidity & Elliott Wave Alignment
3. Long Setup (Must include: Thesis, Execution Trigger, Entry Range, Stop Loss, Take Profit 1, Take Profit 2, Risk/Reward Ratio)
4. Short Setup (Must include: Thesis, Execution Trigger, Entry Range, Stop Loss, Take Profit 1, Take Profit 2, Risk/Reward Ratio)
5. Execution Verdict (Explicitly declare either 'Primary Stance: Favor Long' or 'Primary Stance: Favor Short', followed by Tactical Rationale).

CRITICAL FORMATTING INSTRUCTION: Write all prices, ratios, and text in plain markdown (e.g. $69,450, 1:2.5). NEVER use LaTeX math formatting (no dollar-sign enclosed formulas like $68,650$).

{st.session_state[data_key]['payload']}"""

                st.session_state[brief_key] = generate_strategy_brief(prompt)
            except Exception as e:
                st.error(f"Error fetching {ticker_symbol} data: {e}")
                return

    data = st.session_state.get(data_key)
    if not data:
        st.warning(f"Click Refresh {ticker_symbol} above to load.")
        return

    # Metric Cards
    m1, m2, m3, m4, m5 = st.columns([1.2, 1.0, 1.1, 1.0, 1.3])
    m1.metric(f"{ticker_symbol} Spot", f"${data['price']:,.{max(round_dec, 2)}f}", delta=f"VWAP: ${data['vwap']:,.{max(round_dec, 2)}f}", delta_color="off")
    m2.metric("4H ATR Buffer", f"${data['atr']:,.{max(round_dec, 2)}f}")
    m3.metric("RSI (15m / 4H)", f"{data['rsi_15m']:.1f} / {data['rsi_4h']:.1f}")
    m4.metric("DXY Index", f"{data['dxy']:.2f}", f"{data['dxy_chg']:+.2f}%")
    m5.metric("Candle Pattern", data['pattern'])

    st.write("")

    # Chart + Quick Level Strip
    col_chart, col_levels = st.columns([3.4, 1.1])
    with col_chart:
        if not data['df_1h'].empty:
            fig = build_pro_chart(
                data['df_1h'], data['vwap'], data['long_raw'], data['short_raw'], data['elliott'],
                is_crypto=True, symbol_label=ticker_symbol
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_levels:
        st.caption("🔴 **Overhead Liquidity**")
        st.dataframe(data['short_df'], use_container_width=True, hide_index=True)
        st.caption("🟢 **Downside Liquidity**")
        st.dataframe(data['long_df'], use_container_width=True, hide_index=True)
        st.caption("🌊 **Elliott Fib Targets**")
        st.dataframe(data['elliott']['table'], use_container_width=True, hide_index=True)

    # Gemini Strategy Brief
    st.subheader("🤖 Gemini Strategy Brief")
    brief_text = st.session_state.get(brief_key, "Analysis pending.")
    st.markdown(brief_text)

    # BTCC Direct Launcher
    trade_params = parse_ai_trade_parameters(brief_text, data, is_crypto=True)
    render_btcc_direct_launcher(ticker_symbol, trade_params, round_decimals=round_dec, brief_text=brief_text)

# --- Main App ---
st.title("⚡ Institutional Derivatives Execution Terminal")

tab_btc, tab_silver, tab_eth, tab_sol, tab_xrp = st.tabs([
    "₿ Bitcoin (BTC)",
    "🥈 Silver (XAG)",
    "Ξ Ethereum (ETH)",
    "🟣 Solana (SOL)",
    "✕ Ripple (XRP)"
])

with tab_btc:
    render_crypto_tab("Bitcoin", "BTC/USDT", "btc", round_dec=0)

with tab_silver:
    col_ctrl_s, _ = st.columns([1, 4])
    with col_ctrl_s:
        refresh_slv = st.button("🔄 Refresh Silver Data", key="slv_btn")

    if 'slv_data' not in st.session_state or refresh_slv:
        with st.spinner("Fetching Silver chart data, Elliott swings & stop pools..."):
            try:
                st.session_state['slv_data'] = get_silver_data()
                prompt_s = f"""You are a commodities strategist. Analyze this snapshot for Silver, including Elliott Wave projections and liquidation pools. 

Structure your response strictly into these 5 sections:
1. Macro & GSR Dynamics
2. Liquidity & Elliott Wave Alignment
3. Long Setup (Must include: Thesis, Execution Trigger, Entry Range, Stop Loss, Take Profit 1, Take Profit 2, Risk/Reward Ratio)
4. Short Setup (Must include: Thesis, Execution Trigger, Entry Range, Stop Loss, Take Profit 1, Take Profit 2, Risk/Reward Ratio)
5. Execution Verdict (Explicitly declare either 'Primary Stance: Favor Long' or 'Primary Stance: Favor Short', followed by Tactical Rationale).

CRITICAL FORMATTING INSTRUCTION: Write all prices, ratios, and text in plain markdown (e.g. $67.25, 1:2.5). NEVER use LaTeX math formatting (no dollar-sign enclosed formulas like $68.50$).

{st.session_state['slv_data']['payload']}"""

                st.session_state['slv_ai_brief'] = generate_strategy_brief(prompt_s)
            except Exception as e:
                st.error(f"Error fetching Silver data: {e}")

    sdata = st.session_state.get('slv_data')
    if sdata:
        sm1, sm2, sm3, sm4, sm5 = st.columns([1.2, 1.0, 1.1, 1.0, 1.3])
        sm1.metric("Silver Spot", f"${sdata['price']:,.3f}", delta=f"VWAP: ${sdata['vwap']:,.3f}", delta_color="off")
        sm2.metric("Gold/Silver Ratio", f"{sdata['gsr']:.2f}")
        sm3.metric("Daily ATR", f"${sdata['atr']:.3f}")
        sm4.metric("DXY Index", f"{sdata['dxy']:.2f}", f"{sdata['dxy_chg']:+.2f}%")
        sm5.metric("Candle Pattern", sdata['pattern'])

        st.write("")

        scol_chart, scol_levels = st.columns([3.4, 1.1])
        with scol_chart:
            if not sdata['df_1h'].empty:
                fig_slv = build_pro_chart(
                    sdata['df_1h'], sdata['vwap'], sdata['long_raw'], sdata['short_raw'], sdata['elliott'],
                    is_crypto=False, symbol_label="Silver (XAG/USD)"
                )
                st.plotly_chart(fig_slv, use_container_width=True)

        with scol_levels:
            st.caption("🔴 **Overhead Liquidity**")
            st.dataframe(sdata['short_df'], use_container_width=True, hide_index=True)
            st.caption("🟢 **Downside Liquidity**")
            st.dataframe(sdata['long_df'], use_container_width=True, hide_index=True)
            st.caption("🌊 **Elliott Fib Targets**")
            st.dataframe(sdata['elliott']['table'], use_container_width=True, hide_index=True)

        st.subheader("🤖 Gemini Strategy Brief")
        slv_brief_text = st.session_state.get('slv_ai_brief', "Analysis pending.")
        st.markdown(slv_brief_text)
        
        slv_trade_params = parse_ai_trade_parameters(slv_brief_text, sdata, is_crypto=False)
        render_btcc_direct_launcher("XAG/USDT", slv_trade_params, round_decimals=3, brief_text=slv_brief_text)

with tab_eth:
    render_crypto_tab("Ethereum", "ETH/USDT", "eth", round_dec=2)

with tab_sol:
    render_crypto_tab("Solana", "SOL/USDT", "sol", round_dec=2)

with tab_xrp:
    render_crypto_tab("Ripple", "XRP/USDT", "xrp", round_dec=4)