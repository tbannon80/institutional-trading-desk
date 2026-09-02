import logging
from brain_db import get_setting

logger = logging.getLogger(__name__)

def calculate_conviction_score(symbol, setup_type, entry_price, vwap, rsi, fvg_aligned, 
                               htf_regime="NEUTRAL", funding_rate=None, oi_trend=None, dxy_trend=None):
    """
    Calculates dynamic conviction score (0-100) using dynamic weights from SQLite settings.
    Strictly scoped to BTC (funding/OI) and Silver (FRED DXY).
    
    HTF Regime Filtering (Specification 5):
    - If 4H Regime is BEARISH (Close < EMA200) and setup is LONG:
      Applies a heavy 0.2x penalty multiplier and disables oversold RSI point bonus (Falling knife danger).
    - If 4H Regime is BULLISH (Close > EMA200) and setup is SHORT:
      Applies a heavy 0.2x penalty multiplier and disables overbought RSI point bonus.
    """
    # 1. Load active weights from settings (defaults: 25, 25, 15, 35)
    try:
        w_rsi = float(get_setting("w_rsi", 25.0))
        w_vwap = float(get_setting("w_vwap", 25.0))
        w_fvg = float(get_setting("w_fvg", 15.0))
        w_external = float(get_setting("w_external", 35.0))
    except Exception as e:
        logger.error(f"Error loading weights from DB, using defaults: {e}")
        w_rsi, w_vwap, w_fvg, w_external = 25.0, 25.0, 15.0, 35.0

    reasons = []
    htf_regime_upper = (htf_regime or "NEUTRAL").upper()
    setup_type_upper = setup_type.upper()

    # 2. HTF Regime Contradiction Check
    htf_penalty = 1.0
    if setup_type_upper == "LONG" and htf_regime_upper == "BEARISH":
        htf_penalty = 0.2
        reasons.append("⚠️ HTF 4H Bear Regime active (Close < EMA200): Severe downward trend momentum penalty (0.2x)")
    elif setup_type_upper == "SHORT" and htf_regime_upper == "BULLISH":
        htf_penalty = 0.2
        reasons.append("⚠️ HTF 4H Bull Regime active (Close > EMA200): Strong upward breakout momentum penalty (0.2x)")
    elif (setup_type_upper == "LONG" and htf_regime_upper == "BULLISH") or (setup_type_upper == "SHORT" and htf_regime_upper == "BEARISH"):
        reasons.append(f"✅ HTF 4H Trend Alignment confirmed ({htf_regime_upper})")

    # 3. Compute Base RSI Factor (max 25 pts)
    rsi_pts = 5.0
    if setup_type_upper == "LONG":
        if htf_regime_upper == "BEARISH":
            # In a macro downtrend, oversold is a trap, not a buy signal
            rsi_pts = 0.0
            if rsi <= 30.0:
                reasons.append(f"⚠️ 15m Oversold RSI ({rsi:.1f}) in 4H Bear Regime indicates falling knife danger")
        else:
            if rsi <= 30.0:
                rsi_pts = 25.0
                reasons.append(f"15m Oversold RSI ({rsi:.1f}) with supportive macro regime")
            elif 30.0 < rsi < 50.0:
                rsi_pts = 25.0 - (rsi - 30.0) * 1.0
    else: # SHORT
        if htf_regime_upper == "BULLISH":
            # In a macro uptrend, overbought is momentum, not a short signal
            rsi_pts = 0.0
            if rsi >= 70.0:
                reasons.append(f"⚠️ 15m Overbought RSI ({rsi:.1f}) in 4H Bull Regime indicates strong breakout continuation")
        else:
            if rsi >= 70.0:
                rsi_pts = 25.0
                reasons.append(f"15m Overbought RSI ({rsi:.1f}) with supportive macro regime")
            elif 50.0 < rsi < 70.0:
                rsi_pts = 5.0 + (rsi - 50.0) * 1.0

    # 4. Compute Base VWAP Alignment Factor (max 25 pts)
    vwap_pts = 0.0
    if setup_type_upper == "LONG":
        if entry_price < vwap:
            vwap_pts = 25.0
            reasons.append("SMC Entry anchored in deep Discount beneath Session VWAP")
    else: # SHORT
        if entry_price > vwap:
            vwap_pts = 25.0
            reasons.append("SMC Entry anchored in Premium Supply above Session VWAP")

    # 5. Compute Base FVG Factor (max 15 pts)
    fvg_pts = 15.0 if fvg_aligned else 0.0
    if fvg_aligned:
        reasons.append("Fair Value Gap (FVG) structural imbalance confirmed")

    # 6. Compute Base External Factor (max 35 pts)
    ext_pts = 0.0
    is_btc = "BTC" in symbol.upper()
    is_silver = "XAG" in symbol.upper() or "SILVER" in symbol.upper()

    if is_btc:
        # Funding rate (max 20 pts)
        fund_pts = 5.0
        f_rate = funding_rate if funding_rate is not None else 0.0001
        
        if setup_type_upper == "LONG":
            if f_rate <= 0.0001:
                fund_pts = 20.0
                reasons.append(f"Low funding rate ({f_rate*100:.3f}%) eliminates long squeeze risk")
            elif 0.0001 < f_rate < 0.0005:
                fund_pts = 20.0 - (f_rate - 0.0001) * 37500.0
            else:
                fund_pts = 0.0
                reasons.append("⚠️ Overheated funding longs")
        else: # SHORT
            if f_rate >= 0.0003:
                fund_pts = 20.0
                reasons.append(f"High funding rate ({f_rate*100:.3f}%) provides shorting premium")
            elif 0.0001 < f_rate < 0.0003:
                fund_pts = 5.0 + (f_rate - 0.0001) * 75000.0
            else:
                fund_pts = 5.0
                
        # OI Trend (max 15 pts)
        oi_pts = 5.0
        trend = oi_trend if oi_trend is not None else "FLAT"
        if setup_type_upper == "LONG":
            if trend in ("DOWN", "FLAT"):
                oi_pts = 15.0
                reasons.append("OI flush indicates long liquidation wash-out complete")
            else:
                reasons.append("⚠️ High OI buildup on price drop (aggressive positioning)")
        else: # SHORT
            if trend in ("DOWN", "FLAT"):
                oi_pts = 15.0
                reasons.append("Muted OI on rise indicates weak buying participation")
            else:
                reasons.append("⚠️ Strong buyer commitment (OI rising)")
                
        ext_pts = fund_pts + oi_pts

    elif is_silver:
        trend = dxy_trend if dxy_trend is not None else "FLAT"
        if setup_type_upper == "LONG":
            if trend == "DOWN":
                ext_pts = 35.0
                reasons.append("Strong DXY downtrend supports precious metals upside")
            elif trend == "FLAT":
                ext_pts = 20.0
                reasons.append("Flat DXY provides neutral macro backdrop")
            else:
                ext_pts = 5.0
                reasons.append("⚠️ Strong dollar headwind for silver")
        else: # SHORT
            if trend == "UP":
                ext_pts = 35.0
                reasons.append("Strong dollar trend pressures silver lower")
            elif trend == "FLAT":
                ext_pts = 15.0
            else:
                ext_pts = 5.0

    # 7. Apply Weights and calculate final score
    rsi_weighted = (rsi_pts / 25.0) * w_rsi
    vwap_weighted = (vwap_pts / 25.0) * w_vwap
    fvg_weighted = (fvg_pts / 15.0) * w_fvg
    ext_weighted = (ext_pts / 35.0) * w_external
    
    raw_score = rsi_weighted + vwap_weighted + fvg_weighted + ext_weighted
    
    # Apply HTF Regime Multiplier Penalty
    final_score = raw_score * htf_penalty

    return round(max(0.0, min(100.0, final_score)), 1), reasons
