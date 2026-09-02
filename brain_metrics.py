import requests
import logging

logger = logging.getLogger(__name__)

def fetch_binance_funding_and_oi(symbol="BTCUSDT"):
    """
    Fetches USDS-M Futures current funding rate and recent open interest trend.
    Returns dict: { "funding_rate": float, "current_oi": float, "oi_trend": "UP"|"DOWN"|"FLAT" }
    """
    res = {
        "funding_rate": 0.0,
        "current_oi": 0.0,
        "oi_trend": "FLAT"
    }
    
    headers = {"User-Agent": "Mozilla/5.0"}
    sym = symbol.upper().replace("/", "")
    if "BTC" not in sym:
        return res # Only track metrics for BTC
        
    # 1. Fetch Funding Rate and Mark Price
    funding_url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={sym}"
    try:
        r = requests.get(funding_url, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            res["funding_rate"] = float(data.get("lastFundingRate", 0.0))
    except Exception as e:
        logger.error(f"Error fetching Binance funding rate: {e}")
        
    # 2. Fetch Open Interest Trend (1-hour window, 15m intervals)
    oi_url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={sym}&period=15m&limit=5"
    try:
        r = requests.get(oi_url, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) >= 2:
                # Sort by timestamp ascending
                observations = sorted(data, key=lambda x: x.get("timestamp", 0))
                
                latest_oi = float(observations[-1].get("sumOpenInterest", 0.0))
                oldest_oi = float(observations[0].get("sumOpenInterest", 0.0))
                
                res["current_oi"] = latest_oi
                
                # Check 1-hour change threshold (e.g. 0.05% difference for flatness)
                pct_change = (latest_oi - oldest_oi) / (oldest_oi + 1e-9)
                if pct_change > 0.001: # 0.1% increase
                    res["oi_trend"] = "UP"
                elif pct_change < -0.001: # 0.1% decrease
                    res["oi_trend"] = "DOWN"
                else:
                    res["oi_trend"] = "FLAT"
    except Exception as e:
        logger.error(f"Error fetching Binance open interest: {e}")
        
    return res

def fetch_fred_dxy(api_key):
    """
    Fetches the U.S. Dollar Index proxy from FRED (Nominal Advanced Foreign Economies, DTWEXAFEGS)
    and computes the 5-day SMA trend.
    Returns dict: { "dxy_value": float, "dxy_trend": "UP"|"DOWN"|"FLAT", "dxy_sma_5d": float }
    """
    res = {
        "dxy_value": 100.0,
        "dxy_trend": "FLAT",
        "dxy_sma_5d": 100.0
    }
    if not api_key:
        return res
        
    # Fetch 10 observations to account for weekends/holidays where values might be empty
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DTWEXAFEGS&api_key={api_key}&file_type=json&sort_order=desc&limit=10"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            obs_list = data.get("observations", [])
            
            # Filter out non-numeric values (e.g., "." placeholders during holidays)
            values = []
            for obs in obs_list:
                val_str = obs.get("value", "")
                try:
                    values.append(float(val_str))
                except ValueError:
                    continue
                    
            if len(values) >= 5:
                current_dxy = values[0]
                sma_5d = sum(values[:5]) / 5.0
                
                res["dxy_value"] = current_dxy
                res["dxy_sma_5d"] = sma_5d
                
                diff = (current_dxy - sma_5d) / (sma_5d + 1e-9)
                if diff > 0.0005: # 0.05% above average
                    res["dxy_trend"] = "UP"
                elif diff < -0.0005: # 0.05% below average
                    res["dxy_trend"] = "DOWN"
                else:
                    res["dxy_trend"] = "FLAT"
    except Exception as e:
        logger.error(f"Error fetching FRED DXY index: {e}")
        
    return res
