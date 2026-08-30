import requests
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_candles(symbol: str, interval: str = "15m", limit: int = 100):
    # Use MEXC as primary to avoid Binance 451 geo-blocking
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=10).json()
        if not isinstance(res, list):
            if symbol != "SILVERUSDT":
                print(f"Error: MEXC API returned non-list: {res}")
            return pd.DataFrame()
        columns_list = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"
        ]
        df = pd.DataFrame(res, columns=columns_list[:len(res[0])])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        # print(f"Fetched {len(df)} candles for {symbol}")
        return df
    except Exception:
        # Fallback to Binance (original) just in case
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            res = requests.get(url, timeout=10).json()
            df = pd.DataFrame(res, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df
        except Exception:
            return pd.DataFrame()

def evaluate_market(symbol: str = "BTCUSDT", asset_type: str = "crypto"):
    df = fetch_candles(symbol, interval="15m", limit=100)
    if df.empty or len(df) < 30:
        if asset_type == "cme_silver":
            return {"status": "closed"}
        return {"status": "error"}

    # Anchor calculations to closed candle to prevent repainting
    closed_df = df.iloc[:-1].copy()
    spot = float(df["close"].iloc[-1])

    # ATR (14) on closed bars
    high_low = closed_df["high"] - closed_df["low"]
    high_cp = (closed_df["high"] - closed_df["close"].shift()).abs()
    low_cp = (closed_df["low"] - closed_df["close"].shift()).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    # RSI (14)
    delta = closed_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs)).iloc[-1]

    # Session VWAP
    cum_vol = closed_df["volume"].cumsum()
    cum_pv = (closed_df["close"] * closed_df["volume"]).cumsum()
    vwap = (cum_pv / (cum_vol + 1e-9)).iloc[-1]

    # Structural Swing Zones (Fixed 20-bar lookback)
    recent_highs = closed_df["high"].tail(20)
    recent_lows = closed_df["low"].tail(20)

    supply_entry = float(recent_highs.max())
    demand_entry = float(recent_lows.min())

    # Invalidation & Take-Profit Offsets
    supply_sl = supply_entry + (1.2 * atr)
    demand_sl = demand_entry - (1.2 * atr)
    tp1 = vwap

    return {
        "symbol": symbol,
        "spot": spot,
        "supply_entry": round(supply_entry, 1 if "BTC" in symbol else 2),
        "supply_sl": round(supply_sl, 1 if "BTC" in symbol else 2),
        "demand_entry": round(demand_entry, 1 if "BTC" in symbol else 2),
        "demand_sl": round(demand_sl, 1 if "BTC" in symbol else 2),
        "vwap": round(vwap, 1 if "BTC" in symbol else 2),
        "rsi": round(rsi, 1),
        "atr": round(atr, 2),
        "status": "active"
    }

def calculate_clean_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Wilder's RSI, True Range ATR, and Session-Anchored VWAP."""
    df = df.copy()
    
    # 1. Wilder's RSI (14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)

    # 2. True Range & ATR (14)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(tr)

    # 3. Session-Anchored VWAP (UTC Daily Reset)
    if 'timestamp' in df.columns:
        dt_series = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        session_id = dt_series.dt.date
    else:
        session_id = pd.Series(0, index=df.index)
        
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    df['pv'] = typical_price * df['volume']
    
    df['cum_pv'] = df.groupby(session_id)['pv'].cumsum()
    df['cum_vol'] = df.groupby(session_id)['volume'].cumsum()
    df['vwap'] = (df['cum_pv'] / (df['cum_vol'] + 1e-9)).fillna(df['close'])
    
    # 4. Fair Value Gap (FVG) Imbalance Detection
    # Bullish FVG: Current candle Low > 2 candles ago High
    df['bullish_fvg'] = (df['low'] > df['high'].shift(2))
    df['bullish_fvg_top'] = df['low']
    df['bullish_fvg_bottom'] = df['high'].shift(2)
    
    # Bearish FVG: Current candle High < 2 candles ago Low
    df['bearish_fvg'] = (df['high'] < df['low'].shift(2))
    df['bearish_fvg_top'] = df['low'].shift(2)
    df['bearish_fvg_bottom'] = df['high']

    return df

def get_structural_levels(df: pd.DataFrame, symbol: str, spot_price: float):
    """
    Computes High-Conviction Supply/Demand boundaries, FVGs, and Execution Plans.
    Guarantees: Short SL > Short Entry >= Spot >= Long Entry > Long SL
    """
    sym = symbol.upper()
    is_silver = "XAG" in sym or "SILVER" in sym
    decimals = 3 if is_silver else (4 if "XRP" in sym else (2 if "SOL" in sym or "ETH" in sym else 1))

    current = float(spot_price)
    raw_atr = float(df['atr'].iloc[-1])
    min_atr_floor = current * 0.0015 if is_silver else current * 0.0030
    atr = max(raw_atr, min_atr_floor)

    vwap = float(df['vwap'].iloc[-1])
    rsi_val = float(df['rsi'].iloc[-1])

    # 50-bar Swing Structure
    lookback = min(len(df), 50)
    recent_highs = df['high'].iloc[-lookback:]
    recent_lows = df['low'].iloc[-lookback:]

    # Premium Supply / Discount Demand anchors
    above_spot = recent_highs[recent_highs > current]
    below_spot = recent_lows[recent_lows < current]

    swing_high = float(above_spot.max()) if not above_spot.empty else current + (1.5 * atr)
    swing_low = float(below_spot.min()) if not below_spot.empty else current - (1.5 * atr)

    # Core High-Conviction Structural Boundaries
    short_entry = round(max(swing_high, current + (0.4 * atr)), decimals)
    short_sl = round(short_entry + (1.2 * atr), decimals)
    short_tp = round(min(vwap if vwap < current else current - (1.0 * atr), short_entry - (2.0 * atr)), decimals)

    long_entry = round(min(swing_low, current - (0.4 * atr)), decimals)
    long_sl = round(long_entry - (1.2 * atr), decimals)
    long_tp = round(max(vwap if vwap > current else current + (1.0 * atr), long_entry + (2.0 * atr)), decimals)

    # Directional Invariant Enforcement
    if not (short_sl > short_entry >= current >= long_entry > long_sl):
        short_entry = round(current + (0.8 * atr), decimals)
        short_sl = round(short_entry + (1.0 * atr), decimals)
        short_tp = round(current - (0.8 * atr), decimals)
        long_entry = round(current - (0.8 * atr), decimals)
        long_sl = round(long_entry - (1.0 * atr), decimals)
        long_tp = round(current + (0.8 * atr), decimals)

    # 3-Tier Tactical Matrix
    l_trig_low = round(long_entry - (0.25 * atr), decimals)
    l_trig_high = round(long_entry + (0.25 * atr), decimals)
    l_sl = round(long_sl, decimals)
    l_tp1 = round(vwap if vwap > current else current + (0.5 * atr), decimals)
    l_tp2 = round(short_entry, decimals)
    l_rr = round(abs(l_trig_high - l_tp2) / (abs(l_trig_high - l_sl) + 1e-9), 2)

    s_trig_low = round(short_entry - (0.25 * atr), decimals)
    s_trig_high = round(short_entry + (0.25 * atr), decimals)
    s_sl = round(short_sl, decimals)
    s_tp1 = round(vwap if vwap < current else current - (0.5 * atr), decimals)
    s_tp2 = round(long_entry, decimals)
    s_rr = round(abs(s_trig_low - s_tp2) / (abs(s_trig_low - s_sl) + 1e-9), 2)

    tactical_matrix = [
        {
            "Setup": "🟢 Primary Long (Discount Liquidity Sweep)",
            "Trigger / Entry": f"${l_trig_low:.{decimals}f} – ${l_trig_high:.{decimals}f}",
            "Invalidation (SL)": f"${l_sl:.{decimals}f}",
            "Targets (TP)": f"TP1: ${l_tp1:.{decimals}f} | TP2: ${l_tp2:.{decimals}f}",
            "R:R & Rationale": f"1:{l_rr:.1f} R:R | Institutional demand absorption below Session VWAP."
        },
        {
            "Setup": "🔴 Primary Short (Supply Shelf Exhaustion)",
            "Trigger / Entry": f"${s_trig_low:.{decimals}f} – ${s_trig_high:.{decimals}f}",
            "Invalidation (SL)": f"${s_sl:.{decimals}f}",
            "Targets (TP)": f"TP1: ${s_tp1:.{decimals}f} | TP2: ${s_tp2:.{decimals}f}",
            "R:R & Rationale": f"1:{s_rr:.1f} R:R | Premium liquidity sweep mitigation with overbought exhaustion."
        }
    ]

    return {
        "symbol": symbol,
        "spot": current,
        "decimals": decimals,
        "vwap": round(vwap, decimals),
        "atr": round(atr, decimals),
        "rsi": round(rsi_val, 1),
        "short_plan": {"entry": short_entry, "sl": short_sl, "tp": short_tp},
        "long_plan": {"entry": long_entry, "sl": long_sl, "tp": long_tp},
        "tactical_matrix": tactical_matrix,
        "is_silver": is_silver
    }