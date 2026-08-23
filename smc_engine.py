"""
smc_engine.py - Smart Money Concepts & Liquidity Engine
Sanitized calculation core with strict directional bounds, asset routing, and directional bias scoring.
"""
import pandas as pd
import numpy as np

def calculate_clean_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes RSI, ATR, VWAP, and CVD without lookahead bias."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)

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
    return df

def compute_directional_bias(df: pd.DataFrame, spot: float, vwap: float, rsi: float) -> dict:
    """
    Computes a quantitative weighted directional edge.
    Pillars: VWAP Location (35%), CVD Slope (35%), RSI/Momentum Range (30%).
    """
    score = 0
    
    # 1. VWAP Location
    if spot > vwap:
        score += 1
    elif spot < vwap:
        score -= 1
        
    # 2. CVD Flow (last 4 candles)
    if len(df) >= 5 and df['cvd'].iloc[-1] > df['cvd'].iloc[-4]:
        score += 1
    else:
        score -= 1
        
    # 3. Momentum / Dealing Range Location
    if rsi > 65:
        score -= 1  # Overbought in premium -> favors mean reversion Short
    elif rsi < 35:
        score += 1  # Oversold in discount -> favors Long bounce
        
    if score >= 2:
        return {
            "verdict": "FAVOR LONG (BULLISH EDGE) 🟢",
            "reason": "Price holding value with positive delta flow; higher probability of upward expansion.",
            "color": "#089981",
            "bg": "rgba(8, 153, 129, 0.15)"
        }
    elif score <= -2:
        return {
            "verdict": "FAVOR SHORT (BEARISH EDGE) 🔴",
            "reason": "Price trading below equilibrium with net negative volume delta; favors mean reversion / lower rotation.",
            "color": "#f23645",
            "bg": "rgba(242, 54, 69, 0.15)"
        }
    else:
        return {
            "verdict": "NEUTRAL / RANGE ROTATION ⚖️",
            "reason": "Price consolidating near equilibrium; wait for an outer boundary sweep before executing.",
            "color": "#787b86",
            "bg": "rgba(120, 123, 134, 0.15)"
        }

def get_structural_levels(df: pd.DataFrame, symbol: str, spot_price: float):
    """
    Computes structural swing levels and execution plans.
    Strictly asserts: Short SL > Short Entry >= Spot >= Long Entry > Long SL
    """
    is_silver = "XAG" in symbol or "SILVER" in symbol
    decimals = 3 if is_silver else (4 if "XRP" in symbol else (2 if "SOL" in symbol or "ETH" in symbol else 1))
    
    current = float(spot_price)
    atr = max(float(df['atr'].iloc[-1]), current * 0.003)
    vwap = float(df['vwap'].iloc[-1])
    rsi_val = float(df['rsi'].iloc[-1])
    
    # 1. Structural Pivots
    recent_highs = df['high'].iloc[-30:]
    recent_lows = df['low'].iloc[-30:]
    
    swing_high = float(recent_highs[recent_highs > current].min()) if (recent_highs > current).any() else current + (1.2 * atr)
    swing_low = float(recent_lows[recent_lows < current].max()) if (recent_lows < current).any() else current - (1.2 * atr)
    
    # 2. Strict Directional Calculations
    short_entry = round(max(swing_high, current + (0.4 * atr)), decimals)
    short_sl = round(short_entry + (1.2 * atr), decimals)
    short_tp = round(min(vwap if vwap < current else swing_low, short_entry - (1.5 * atr)), decimals)
    
    long_entry = round(min(swing_low, current - (0.4 * atr)), decimals)
    long_sl = round(long_entry - (1.2 * atr), decimals)
    long_tp = round(max(vwap if vwap > current else swing_high, long_entry + (1.5 * atr)), decimals)

    # 3. Deterministic Inversion Guard
    if not (short_sl > short_entry >= current >= long_entry > long_sl):
        short_entry = round(current + (1.0 * atr), decimals)
        short_sl = round(short_entry + (1.0 * atr), decimals)
        short_tp = round(current, decimals)
        long_entry = round(current - (1.0 * atr), decimals)
        long_sl = round(long_entry - (1.0 * atr), decimals)
        long_tp = round(current, decimals)

    overhead_liq = [short_entry, round(short_entry + (0.8 * atr), decimals), round(short_entry + (1.5 * atr), decimals)]
    downside_liq = [long_entry, round(long_entry - (0.8 * atr), decimals), round(long_entry - (1.5 * atr), decimals)]
    
    bias_info = compute_directional_bias(df, current, vwap, rsi_val)

    return {
        "symbol": symbol,
        "spot": current,
        "decimals": decimals,
        "vwap": round(vwap, decimals),
        "atr": round(atr, decimals),
        "rsi": round(rsi_val, 1),
        "overhead": overhead_liq,
        "downside": downside_liq,
        "short_plan": {"entry": short_entry, "sl": short_sl, "tp": short_tp},
        "long_plan": {"entry": long_entry, "sl": long_sl, "tp": long_tp},
        "bias": bias_info,
        "is_silver": is_silver
    }
