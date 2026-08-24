"""
smc_engine.py - Institutional Smart Money Concepts & Execution Matrix Engine
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

def get_structural_levels(df: pd.DataFrame, symbol: str, spot_price: float):
    """
    Computes High-Conviction Supply/Demand boundaries and the 3-Tier Tactical Execution Matrix.
    Strictly asserts: Short SL > Short Entry >= Spot >= Long Entry > Long SL
    """
    is_silver = "XAG" in symbol or "SILVER" in symbol
    decimals = 3 if is_silver else (4 if "XRP" in symbol else (2 if "SOL" in symbol or "ETH" in symbol else 1))
    
    current = float(spot_price)
    
    # Asset-specific ATR and spread floor
    raw_atr = float(df['atr'].iloc[-1])
    min_atr_floor = current * 0.0015 if is_silver else current * 0.0035
    atr = max(raw_atr, min_atr_floor)
    
    vwap = float(df['vwap'].iloc[-1])
    rsi_val = float(df['rsi'].iloc[-1])
    
    # Structural Pivots
    lookback = 35
    recent_highs = df['high'].iloc[-lookback:]
    recent_lows = df['low'].iloc[-lookback:]
    
    # High Conviction Anchors (Premium Supply Ceiling / Discount Demand Floor)
    swing_high = float(recent_highs[recent_highs > current].max()) if (recent_highs > current).any() else current + (1.8 * atr)
    swing_low = float(recent_lows[recent_lows < current].min()) if (recent_lows < current).any() else current - (1.8 * atr)
    
    # High-Conviction Core Plans
    short_entry = round(max(swing_high, current + (0.5 * atr)), decimals)
    short_sl = round(short_entry + (1.2 * atr), decimals)
    short_tp = round(min(vwap if vwap < current else swing_low, short_entry - (2.0 * atr)), decimals)
    
    long_entry = round(min(swing_low, current - (0.5 * atr)), decimals)
    long_sl = round(long_entry - (1.2 * atr), decimals)
    long_tp = round(max(vwap if vwap > current else swing_high, long_entry + (2.0 * atr)), decimals)

    # Inversion Guard
    if not (short_sl > short_entry >= current >= long_entry > long_sl):
        short_entry = round(current + (1.0 * atr), decimals)
        short_sl = round(short_entry + (1.0 * atr), decimals)
        short_tp = round(current, decimals)
        long_entry = round(current - (1.0 * atr), decimals)
        long_sl = round(long_entry - (1.0 * atr), decimals)
        long_tp = round(current, decimals)

    # 3-Tier Tactical Execution Matrix
    # Setup 1: Primary Long (Discount Sweep)
    l_trig_low = round(long_entry - (0.3 * atr), decimals)
    l_trig_high = round(long_entry + (0.2 * atr), decimals)
    l_sl = round(l_trig_low - (0.8 * atr), decimals)
    l_tp1 = round(vwap if vwap > current else current + (0.5 * atr), decimals)
    l_tp2 = short_entry
    l_rr = round(abs(l_trig_high - l_tp2) / (abs(l_trig_high - l_sl) + 1e-9), 2)
    
    # Setup 2: Primary Short (Supply / Mitigation Fade)
    s_trig_low = round(short_entry - (0.2 * atr), decimals)
    s_trig_high = round(short_entry + (0.3 * atr), decimals)
    s_sl = round(s_trig_high + (0.8 * atr), decimals)
    s_tp1 = round(vwap if vwap < current else current - (0.5 * atr), decimals)
    s_tp2 = long_entry
    s_rr = round(abs(s_trig_low - s_tp2) / (abs(s_trig_low - s_sl) + 1e-9), 2)
    
    # Setup 3: Breakdown Short Continuation
    bd_entry = round(current - (0.3 * atr), decimals)
    bd_sl = round(current + (0.4 * atr), decimals)
    bd_tp1 = long_entry
    bd_tp2 = round(long_entry - (1.0 * atr), decimals)
    bd_rr = round(abs(bd_entry - bd_tp1) / (abs(bd_entry - bd_sl) + 1e-9), 2)

    tactical_matrix = [
        {
            "Setup": "🟢 Primary Long (Discount Floor Sweep)",
            "Trigger / Entry": f"${l_trig_low:.{decimals}f} – ${l_trig_high:.{decimals}f}",
            "Invalidation (SL)": f"${l_sl:.{decimals}f}",
            "Targets (TP)": f"TP1: ${l_tp1:.{decimals}f} | TP2: ${l_tp2:.{decimals}f}",
            "R:R & Rationale": f"1:{l_rr:.1f} R:R | Fading sell-side liquidity run into discount boundary; absorption expected."
        },
        {
            "Setup": "🔴 Primary Short (Supply Retest / Fade)",
            "Trigger / Entry": f"${s_trig_low:.{decimals}f} – ${s_trig_high:.{decimals}f}",
            "Invalidation (SL)": f"${s_sl:.{decimals}f}",
            "Targets (TP)": f"TP1: ${s_tp1:.{decimals}f} | TP2: ${s_tp2:.{decimals}f}",
            "R:R & Rationale": f"1:{s_rr:.1f} R:R | Exhaustion wick into upper supply shelf under heavy Point of Control."
        },
        {
            "Setup": "⚡ Aggressive Breakdown Short",
            "Trigger / Entry": f"${bd_entry:.{decimals}f} (Confirmed close)",
            "Invalidation (SL)": f"${bd_sl:.{decimals}f}",
            "Targets (TP)": f"TP1: ${bd_tp1:.{decimals}f} | TP2: ${bd_tp2:.{decimals}f}",
            "R:R & Rationale": f"1:{bd_rr:.1f} R:R | Momentum flush through thin intermediate volume air pocket."
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
