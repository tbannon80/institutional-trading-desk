import logging
from brain_db import get_setting

logger = logging.getLogger(__name__)

def calculate_conviction_score(symbol, setup_type, entry_price, vwap, rsi, fvg_aligned, funding_rate=None, oi_trend=None, dxy_trend=None):
    """
    Calculates a dynamic conviction score (0-100) using weights loaded from SQLite settings.
    Surgically handles BTC (funding/OI), Silver (FRED DXY), and fallbacks for other assets.
    """
    # 1. Load active weights from settings (default to 25, 25, 15, 35)
    try:
        w_rsi = float(get_setting("w_rsi", 25.0))
        w_vwap = float(get_setting("w_vwap", 25.0))
        w_fvg = float(get_setting("w_fvg", 15.0))
        w_external = float(get_setting("w_external", 35.0))
    except Exception as e:
        logger.error(f"Error loading weights from DB, using defaults: {e}")
        w_rsi, w_vwap, w_fvg, w_external = 25.0, 25.0, 15.0, 35.0

    # 2. Compute Base RSI Factor (max 25 pts)
    rsi_pts = 5.0
    if setup_type.upper() == "LONG":
        if rsi <= 30.0:
            rsi_pts = 25.0
        elif 30.0 < rsi < 50.0:
            rsi_pts = 25.0 - (rsi - 30.0) * 1.0 # Linear scale from 25 to 5
    else: # SHORT
        if rsi >= 70.0:
            rsi_pts = 25.0
        elif 50.0 < rsi < 70.0:
            rsi_pts = 5.0 + (rsi - 50.0) * 1.0 # Linear scale from 5 to 25

    # 3. Compute Base VWAP Alignment Factor (max 25 pts)
    vwap_pts = 0.0
    if setup_type.upper() == "LONG":
        if entry_price < vwap:
            vwap_pts = 25.0
    else: # SHORT
        if entry_price > vwap:
            vwap_pts = 25.0

    # 4. Compute Base FVG Factor (max 15 pts)
    fvg_pts = 15.0 if fvg_aligned else 0.0

    # 5. Compute Base External Factor (max 35 pts)
    ext_pts = 0.0
    is_btc = "BTC" in symbol.upper()
    is_silver = "XAG" in symbol.upper() or "SILVER" in symbol.upper()

    reasons = []

    if is_btc:
        # Funding rate (max 20 pts)
        fund_pts = 5.0
        f_rate = funding_rate if funding_rate is not None else 0.0001
        
        if setup_type.upper() == "LONG":
            if f_rate <= 0.0001: # Neutral to low funding
                fund_pts = 20.0
                reasons.append("Low funding rate reduces squeeze risk")
            elif 0.0001 < f_rate < 0.0005:
                # Interpolate between 20 and 5
                fund_pts = 20.0 - (f_rate - 0.0001) * 37500.0
            else:
                fund_pts = 0.0
                reasons.append("⚠️ Overheated funding longs")
        else: # SHORT
            if f_rate >= 0.0003:
                fund_pts = 20.0
                reasons.append("High funding rate premium for shorts")
            elif 0.0001 < f_rate < 0.0003:
                fund_pts = 5.0 + (f_rate - 0.0001) * 75000.0
            else:
                fund_pts = 5.0
                
        # OI Trend (max 15 pts)
        oi_pts = 5.0
        trend = oi_trend if oi_trend is not None else "FLAT"
        if setup_type.upper() == "LONG":
            if trend in ("DOWN", "FLAT"):
                oi_pts = 15.0
                reasons.append("OI flush indicates long wash-out complete")
            else:
                reasons.append("⚠️ High OI buildup on price drop")
        else: # SHORT
            if trend in ("DOWN", "FLAT"):
                oi_pts = 15.0
                reasons.append("Muted OI on rise indicates weak buying momentum")
            else:
                reasons.append("⚠️ Strong buying momentum (OI rising)")
                
        ext_pts = fund_pts + oi_pts

    elif is_silver:
        trend = dxy_trend if dxy_trend is not None else "FLAT"
        if setup_type.upper() == "LONG":
            if trend == "DOWN":
                ext_pts = 35.0
                reasons.append("Strong DXY downtrend supports silver breakouts")
            elif trend == "FLAT":
                ext_pts = 20.0
                reasons.append("Flat DXY provides neutral macro backdrop")
            else:
                ext_pts = 5.0
                reasons.append("⚠️ Strong dollar headwind for silver")
        else: # SHORT
            if trend == "UP":
                ext_pts = 35.0
                reasons.append("Strong dollar trend pressures silver")
            elif trend == "FLAT":
                ext_pts = 15.0
            else:
                ext_pts = 5.0

    # 6. Apply Weights and calculate final score
    if is_btc or is_silver:
        rsi_weighted = (rsi_pts / 25.0) * w_rsi
        vwap_weighted = (vwap_pts / 25.0) * w_vwap
        fvg_weighted = (fvg_pts / 15.0) * w_fvg
        ext_weighted = (ext_pts / 35.0) * w_external
        
        score = rsi_weighted + vwap_weighted + fvg_weighted + ext_weighted
    else:
        # Asset has no external metrics; scale first three factors proportionally
        total_active_weight = w_rsi + w_vwap + w_fvg
        if total_active_weight == 0:
            return 50.0, []
            
        rsi_weighted = (rsi_pts / 25.0) * w_rsi
        vwap_weighted = (vwap_pts / 25.0) * w_vwap
        fvg_weighted = (fvg_pts / 15.0) * w_fvg
        
        score = (rsi_weighted + vwap_weighted + fvg_weighted) * (100.0 / total_active_weight)

    # Clean display reasons
    if rsi_pts >= 20.0:
        reasons.append(f"RSI highly favorable ({rsi})")
    if vwap_pts >= 25.0:
        reasons.append("SMC entry aligned with daily Session VWAP")
    if fvg_aligned:
        reasons.append("Fair Value Gap (FVG) structural alignment confirmed")

    return round(max(0.0, min(100.0, score)), 1), reasons
