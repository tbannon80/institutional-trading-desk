import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def is_silver_market_open():
    """COMEX Silver: Sunday 5:00 PM CT (22:00 UTC) through Friday 4:00 PM CT (21:00 UTC)"""
    now = datetime.now(timezone.utc)
    weekday = now.weekday() # Monday = 0, Friday = 4, Saturday = 5, Sunday = 6
    hour = now.hour

    if weekday == 5: # Saturday (Closed all day)
        return False
    if weekday == 6 and hour < 22: # Sunday before 5 PM CT / 22:00 UTC
        return False
    if weekday == 4 and hour >= 21: # Friday after 4 PM CT / 21:00 UTC
        return False
    return True

def validate_candle_data(df: pd.DataFrame, is_silver: bool = False) -> tuple[bool, str]:
    """
    Strict Upstream Data Integrity Check:
    1. Ensures dataframe has at least 20 bars.
    2. Validates that the latest candle is fresh (<= 30 minutes old relative to UTC).
    3. Rejects data with 0 or NaN volume (unless COMEX market is closed).
    """
    if df is None or df.empty or len(df) < 20:
        return False, "insufficient_data"
        
    if is_silver and not is_silver_market_open():
        return True, "market_closed"
        
    # Timestamp freshness check
    now_epoch = time.time()
    if 'timestamp' in df.columns:
        latest_ts = float(df['timestamp'].iloc[-1])
        # If timestamp in milliseconds, convert to seconds
        if latest_ts > 1e11:
            latest_ts /= 1000.0
            
        age_seconds = now_epoch - latest_ts
        if age_seconds > 1800: # Older than 30 minutes
            if is_silver and not is_silver_market_open():
                return True, "market_closed"
            return False, f"stale_data_age_{int(age_seconds)}s"
            
    # Volume check
    latest_vol = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
    if latest_vol <= 0.0:
        if is_silver and not is_silver_market_open():
            return True, "market_closed"
        # If volume across last 3 bars is completely zero
        recent_vol = df['volume'].tail(3).sum() if 'volume' in df.columns else 0.0
        if recent_vol <= 0.0:
            return False, "zero_volume_detected"
            
    return True, "valid"

def calculate_clean_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Wilder's RSI, True Range ATR, Session-Anchored VWAP, and FVGs."""
    df = df.copy()
    
    # 1. Wilder's RSI (14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50.0)

    # 2. True Range & Wilder's Exponential ATR (14)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(tr)

    # 3. Session-Anchored VWAP and Variance Bands (UTC Daily Reset)
    if 'timestamp' in df.columns:
        dt_series = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        session_id = dt_series.dt.date
    else:
        session_id = pd.Series(0, index=df.index)
        
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume'].clip(lower=1e-6)
    pv = typical_price * vol
    pv2 = (typical_price ** 2) * vol
    
    df['cum_pv'] = pv.groupby(session_id).cumsum()
    df['cum_pv2'] = pv2.groupby(session_id).cumsum()
    df['cum_vol'] = vol.groupby(session_id).cumsum()
    df['vwap'] = (df['cum_pv'] / (df['cum_vol'] + 1e-9)).fillna(df['close'])

    variance = (df['cum_pv2'] / (df['cum_vol'] + 1e-9)) - (df['vwap'] ** 2)
    df['vwap_std'] = np.sqrt(variance.clip(lower=0.0)).fillna(df['atr'])
    df['vwap_std'] = np.maximum(df['vwap_std'], 0.5 * df['atr'])
    df['vwap_upper_2'] = df['vwap'] + (2.0 * df['vwap_std'])
    df['vwap_lower_2'] = df['vwap'] - (2.0 * df['vwap_std'])
    df['vwap_upper_25'] = df['vwap'] + (2.5 * df['vwap_std'])
    df['vwap_lower_25'] = df['vwap'] - (2.5 * df['vwap_std'])
    
    # 4. Fair Value Gap (FVG) Imbalance Detection
    # Bullish FVG: Current candle Low > 2 candles ago High
    df['bullish_fvg'] = (df['low'] > df['high'].shift(2))
    df['bullish_fvg_top'] = df['low']
    df['bullish_fvg_bottom'] = df['high'].shift(2)
    
    # Bearish FVG: Current candle High < 2 candles ago Low
    df['bearish_fvg'] = (df['high'] < df['low'].shift(2))
    df['bearish_fvg_top'] = df['low'].shift(2)
    df['bearish_fvg_bottom'] = df['high']

    # 5. Volume 20-SMA for Breakout vs Limit detection
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()

    return df

def find_structural_fvg_targets(df: pd.DataFrame, current_price: float) -> tuple[float | None, float | None]:
    """
    Identifies the nearest confirmed Fair Value Gap (FVG) structural imbalance boundaries
    across both 1-Hour resampled and 15-Minute resolution for realistic liquidity targeting.
    Returns: (nearest_overhead_bearish_fvg, nearest_discount_bullish_fvg)
    """
    overhead_targets = []
    discount_targets = []

    # 1. 15-Minute FVGs from indicators
    if 'bearish_fvg' in df.columns and any(df['bearish_fvg']):
        b_fvgs = df.loc[df['bearish_fvg'] == True, 'bearish_fvg_bottom']
        overhead_targets.extend(b_fvgs[b_fvgs > current_price].tolist())

    if 'bullish_fvg' in df.columns and any(df['bullish_fvg']):
        bull_fvgs = df.loc[df['bullish_fvg'] == True, 'bullish_fvg_top']
        discount_targets.extend(bull_fvgs[bull_fvgs < current_price].tolist())

    # 2. 1-Hour Confirmed FVGs if timestamps are present
    if 'timestamp' in df.columns and len(df) >= 8:
        try:
            df_1h = df.set_index(pd.to_datetime(df['timestamp'], unit='s', utc=True)).resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna().reset_index()

            if len(df_1h) >= 3:
                # 1H Bearish FVG: High < Low 2 bars ago
                bear_1h = df_1h['high'] < df_1h['low'].shift(2)
                if any(bear_1h):
                    h_vals = df_1h.loc[bear_1h, 'high']
                    overhead_targets.extend(h_vals[h_vals > current_price].tolist())

                # 1H Bullish FVG: Low > High 2 bars ago
                bull_1h = df_1h['low'] > df_1h['high'].shift(2)
                if any(bull_1h):
                    l_vals = df_1h.loc[bull_1h, 'low']
                    discount_targets.extend(l_vals[l_vals < current_price].tolist())
        except Exception:
            pass

    nearest_overhead = min(overhead_targets) if overhead_targets else None
    nearest_discount = max(discount_targets) if discount_targets else None

    return nearest_overhead, nearest_discount

def fetch_htf_regime(symbol: str = "BTC/USDT") -> dict:
    """
    Pulls 4-Hour candle history to determine overarching Macro Regime:
    - Calculates 4H EMA50 and 4H EMA200
    - Regime is BULLISH if Close > EMA200 and EMA50 > EMA200
    - Regime is BEARISH if Close < EMA200 and EMA50 < EMA200
    - Otherwise NEUTRAL
    """
    sym = symbol.upper()
    is_silver = "XAG" in sym or "SILVER" in sym
    headers = {"User-Agent": "Mozilla/5.0"}
    df_htf = None

    if is_silver:
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/SI=F?interval=1h&range=60d"
            r = requests.get(url, headers=headers, timeout=5).json()
            res = r['chart']['result'][0]
            timestamps = res['timestamp']
            quote = res['indicators']['quote'][0]
            df_htf = pd.DataFrame({
                'timestamp': timestamps,
                'close': quote['close']
            }).dropna().reset_index(drop=True)
        except Exception:
            pass
    else:
        # BTC from Coinbase Public Candles (1h granularity = 3600)
        try:
            url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                raw = r.json()
                if isinstance(raw, list) and len(raw) > 0:
                    df_temp = pd.DataFrame(raw, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                    df_htf = df_temp.sort_values('timestamp').reset_index(drop=True)
        except Exception:
            pass

    if df_htf is not None and not df_htf.empty and len(df_htf) >= 50:
        close_series = df_htf['close'].astype(float)
        ema50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(close_series.ewm(span=min(200, len(close_series)), adjust=False).mean().iloc[-1])
        current_close = float(close_series.iloc[-1])

        if current_close > ema200 and ema50 >= ema200:
            regime = "BULLISH"
        elif current_close < ema200 and ema50 <= ema200:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        return {
            "htf_regime": regime,
            "ema50_4h": round(ema50, 2 if not is_silver else 3),
            "ema200_4h": round(ema200, 2 if not is_silver else 3),
            "current_close": current_close
        }

    return {"htf_regime": "NEUTRAL", "ema50_4h": 0.0, "ema200_4h": 0.0, "current_close": 0.0}

def get_structural_levels(df: pd.DataFrame, symbol: str, spot_price: float):
    """
    Computes High-Conviction Structural Boundaries using Market Structure Invalidation (MSI).
    Strictly scoped to BTC/USDT and SILVER/USDT.
    
    MSI Rule:
    - Long SL = Absolute extreme wick of originating structural swing low minus (0.15 * ATR).
    - Short SL = Absolute extreme wick of originating structural swing high plus (0.15 * ATR).
    - TP1 = 1.0 R:R / Session VWAP (50% position exit + Move SL to Break-Even).
    - TP2 = 2.0+ R:R (Remaining 50% runner target).
    - Order Type = 'BREAKOUT' if expansion volume > 1.5x SMA20, else 'LIMIT'.
    """
    sym = symbol.upper()
    is_silver = "XAG" in sym or "SILVER" in sym
    decimals = 3 if is_silver else 1

    current = float(spot_price)
    raw_atr = float(df['atr'].iloc[-1])
    min_atr_floor = current * 0.0015 if is_silver else current * 0.0030
    atr = max(raw_atr, min_atr_floor)

    vwap = float(df['vwap'].iloc[-1])
    rsi_val = float(df['rsi'].iloc[-1])

    # 50-bar Swing Structure (Core lookback preserved)
    lookback = min(len(df), 50)
    recent_df = df.iloc[-lookback:].copy()
    recent_highs = recent_df['high'].astype(float)
    recent_lows = recent_df['low'].astype(float)

    raw_swing_high = float(recent_highs.max())
    raw_swing_low = float(recent_lows.min())

    # Candle body boundaries and localized consolidation shelves
    bodies_high = recent_df[['open', 'close']].max(axis=1).astype(float)
    bodies_low = recent_df[['open', 'close']].min(axis=1).astype(float)
    max_body_high = float(bodies_high.max())
    min_body_low = float(bodies_low.min())
    upper_shelf = float(np.percentile(bodies_high, 90))
    lower_shelf = float(np.percentile(bodies_low, 10))

    # Statistical dispersion metrics across 50-bar window
    high_mean = float(recent_highs.mean())
    high_std = float(recent_highs.std()) if float(recent_highs.std()) > 0 else atr
    z_score_high = (raw_swing_high - high_mean) / high_std

    low_mean = float(recent_lows.mean())
    low_std = float(recent_lows.std()) if float(recent_lows.std()) > 0 else atr
    z_score_low = (low_mean - raw_swing_low) / low_std

    # Statistical Outlier Filtering on Swing Extrema:
    # Filters freak volatility spikes / unconfirmed liquidity-sweep wicks > 2.5 * ATR from body boundary/shelf or VWAP
    is_high_outlier = (
        (raw_swing_high - max_body_high > 2.5 * atr) or
        (z_score_high > 2.5 and raw_swing_high - max_body_high > 1.5 * atr) or
        (raw_swing_high - vwap > 2.5 * atr and raw_swing_high - max_body_high > 1.8 * atr)
    )

    is_low_outlier = (
        (min_body_low - raw_swing_low > 2.5 * atr) or
        (z_score_low > 2.5 and min_body_low - raw_swing_low > 1.5 * atr) or
        (vwap - raw_swing_low > 2.5 * atr and min_body_low - raw_swing_low > 1.8 * atr)
    )

    if is_high_outlier:
        valid_highs = recent_highs[recent_highs <= max(max_body_high + 0.3 * atr, upper_shelf + 0.3 * atr)]
        swing_high_wick = float(valid_highs.max()) if not valid_highs.empty else max_body_high
    else:
        swing_high_wick = raw_swing_high

    if is_low_outlier:
        valid_lows = recent_lows[recent_lows >= min(min_body_low - 0.3 * atr, lower_shelf - 0.3 * atr)]
        swing_low_wick = float(valid_lows.min()) if not valid_lows.empty else min_body_low
    else:
        swing_low_wick = raw_swing_low

    # Ensure swing extrema remain strictly outside spot buffer
    swing_high_wick = max(swing_high_wick, current + (0.5 * atr))
    swing_low_wick = min(swing_low_wick, current - (0.5 * atr))

    # Premium Supply / Discount Demand entry anchors (protected from outlier corruption)
    filtered_highs = recent_highs[recent_highs <= swing_high_wick]
    filtered_lows = recent_lows[recent_lows >= swing_low_wick]
    above_spot = filtered_highs[filtered_highs > current]
    below_spot = filtered_lows[filtered_lows < current]

    swing_high_anchor = float(above_spot.max()) if not above_spot.empty else current + (1.5 * atr)
    swing_low_anchor = float(below_spot.min()) if not below_spot.empty else current - (1.5 * atr)

    # 1. Long Plan with Market Structure Invalidation (MSI)
    long_entry = round(min(swing_low_anchor, current - (0.3 * atr)), decimals)
    # MSI: Place SL strictly below originating structural swing low wick minus 0.15 ATR volatility buffer
    long_sl = round(swing_low_wick - (0.15 * atr), decimals)
    
    # Ensure positive risk distance
    long_risk = max(long_entry - long_sl, 0.8 * atr)
    long_sl = round(long_entry - long_risk, decimals)
    
    # TP1 (50% exit)
    long_tp1 = round(long_entry + (1.0 * long_risk), decimals)
    if vwap > long_entry and vwap < long_tp1 + (0.5 * atr):
        long_tp1 = round(vwap, decimals)
    long_tp1 = max(long_tp1, round(long_entry + (0.8 * long_risk), decimals))

    # Dynamic Structural Boundaries for Long TP2 Runner Target
    min_long_runner = long_entry + (2.0 * long_risk)
    vwap_std = float(df['vwap_std'].iloc[-1]) if 'vwap_std' in df.columns else atr
    nearest_overhead_fvg, nearest_discount_fvg = find_structural_fvg_targets(df, current)

    long_runner_candidates = []
    # 1. Multi-std deviation bands of Session-Anchored VWAP
    for band in [vwap, vwap + (1.0 * vwap_std), vwap + (2.0 * vwap_std), vwap + (2.5 * vwap_std)]:
        if band >= min_long_runner:
            long_runner_candidates.append(band)

    # 2. Confirmed overhead Fair Value Gaps (1H / 15m)
    if nearest_overhead_fvg is not None and nearest_overhead_fvg >= min_long_runner:
        long_runner_candidates.append(nearest_overhead_fvg)

    # 3. Filtered structural swing high wick
    if swing_high_wick >= min_long_runner:
        long_runner_candidates.append(swing_high_wick)

    # Select the nearest confirmed dynamic structural boundary (preventing outlier inflation)
    if long_runner_candidates:
        long_tp2 = min(long_runner_candidates)
    else:
        long_tp2 = min_long_runner

    # Guarantee TP2 is strictly greater than TP1 with positive spacing
    long_tp2 = round(max(long_tp2, long_tp1 + (0.4 * atr), min_long_runner), decimals)

    # 2. Short Plan with Market Structure Invalidation (MSI)
    short_entry = round(max(swing_high_anchor, current + (0.3 * atr)), decimals)
    # MSI: Place SL strictly above originating structural swing high wick plus 0.15 ATR volatility buffer
    short_sl = round(swing_high_wick + (0.15 * atr), decimals)
    
    # Ensure positive risk distance
    short_risk = max(short_sl - short_entry, 0.8 * atr)
    short_sl = round(short_entry + short_risk, decimals)
    
    # TP1 (50% exit)
    short_tp1 = round(short_entry - (1.0 * short_risk), decimals)
    if vwap < short_entry and vwap > short_tp1 - (0.5 * atr):
        short_tp1 = round(vwap, decimals)
    short_tp1 = min(short_tp1, round(short_entry - (0.8 * short_risk), decimals))

    # Dynamic Structural Boundaries for Short TP2 Runner Target
    min_short_runner = short_entry - (2.0 * short_risk)

    short_runner_candidates = []
    # 1. Multi-std deviation bands of Session-Anchored VWAP
    for band in [vwap, vwap - (1.0 * vwap_std), vwap - (2.0 * vwap_std), vwap - (2.5 * vwap_std)]:
        if band <= min_short_runner:
            short_runner_candidates.append(band)

    # 2. Confirmed discount Fair Value Gaps (1H / 15m)
    if nearest_discount_fvg is not None and nearest_discount_fvg <= min_short_runner:
        short_runner_candidates.append(nearest_discount_fvg)

    # 3. Filtered structural swing low wick
    if swing_low_wick <= min_short_runner:
        short_runner_candidates.append(swing_low_wick)

    # Select the nearest confirmed dynamic structural boundary (preventing outlier inflation)
    if short_runner_candidates:
        short_tp2 = max(short_runner_candidates)
    else:
        short_tp2 = min_short_runner

    # Guarantee TP2 is strictly lower than TP1 with positive spacing
    short_tp2 = round(min(short_tp2, short_tp1 - (0.4 * atr), min_short_runner), decimals)

    # Directional Invariant Enforcement: Short SL > Short Entry >= Spot >= Long Entry > Long SL and TP ordering
    if not (short_sl > short_entry >= current >= long_entry > long_sl and long_tp2 > long_tp1 and short_tp2 < short_tp1):
        short_risk = 1.0 * atr
        short_entry = round(current + (0.6 * atr), decimals)
        short_sl = round(short_entry + short_risk, decimals)
        short_tp1 = round(current - (0.6 * atr), decimals)
        short_tp2 = round(short_entry - (2.0 * short_risk), decimals)
        
        long_risk = 1.0 * atr
        long_entry = round(current - (0.6 * atr), decimals)
        long_sl = round(long_entry - long_risk, decimals)
        long_tp1 = round(current + (0.6 * atr), decimals)
        long_tp2 = round(long_entry + (2.0 * long_risk), decimals)

    # 3. Order Differentiation (LIMIT vs BREAKOUT)
    vol_sma20 = float(df['vol_sma20'].iloc[-1]) if 'vol_sma20' in df.columns else 1000.0
    latest_vol = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
    
    # If candle volume expands > 1.5x 20-period average during active push
    if latest_vol > 1.5 * vol_sma20 and (current >= swing_high_wick or current <= swing_low_wick):
        entry_order_type = "BREAKOUT"
    else:
        entry_order_type = "LIMIT"

    # Tactical Execution Matrix
    l_rr1 = round(abs(long_entry - long_tp1) / (abs(long_entry - long_sl) + 1e-9), 2)
    l_rr2 = round(abs(long_entry - long_tp2) / (abs(long_entry - long_sl) + 1e-9), 2)
    s_rr1 = round(abs(short_entry - short_tp1) / (abs(short_entry - short_sl) + 1e-9), 2)
    s_rr2 = round(abs(short_entry - short_tp2) / (abs(short_entry - short_sl) + 1e-9), 2)

    tactical_matrix = [
        {
            "Setup": f"🟢 Long ({entry_order_type})",
            "Entry Target": f"${long_entry:.{decimals}f}",
            "Invalidation (MSI SL)": f"${long_sl:.{decimals}f} (Below {swing_low_wick:.{decimals}f} - 0.15 ATR)",
            "Targets (TP1 / TP2)": f"TP1: ${long_tp1:.{decimals}f} (50%) | TP2: ${long_tp2:.{decimals}f} (Runner)",
            "R:R & Rationale": f"TP1: 1:{l_rr1:.1f} | TP2: 1:{l_rr2:.1f} | Demand absorption below VWAP."
        },
        {
            "Setup": f"🔴 Short ({entry_order_type})",
            "Entry Target": f"${short_entry:.{decimals}f}",
            "Invalidation (MSI SL)": f"${short_sl:.{decimals}f} (Above {swing_high_wick:.{decimals}f} + 0.15 ATR)",
            "Targets (TP1 / TP2)": f"TP1: ${short_tp1:.{decimals}f} (50%) | TP2: ${short_tp2:.{decimals}f} (Runner)",
            "R:R & Rationale": f"TP1: 1:{s_rr1:.1f} | TP2: 1:{s_rr2:.1f} | Supply exhaustion above VWAP."
        }
    ]

    return {
        "symbol": symbol,
        "spot": current,
        "decimals": decimals,
        "vwap": round(vwap, decimals),
        "atr": round(atr, decimals),
        "rsi": round(rsi_val, 1),
        "entry_order_type": entry_order_type,
        "short_plan": {
            "entry": short_entry, 
            "sl": short_sl, 
            "tp": short_tp2,
            "tp1": short_tp1, 
            "tp2": short_tp2,
            "msi_anchor": swing_high_wick
        },
        "long_plan": {
            "entry": long_entry, 
            "sl": long_sl, 
            "tp": long_tp2,
            "tp1": long_tp1, 
            "tp2": long_tp2,
            "msi_anchor": swing_low_wick
        },
        "tactical_matrix": tactical_matrix,
        "is_silver": is_silver
    }