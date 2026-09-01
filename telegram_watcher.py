"""
telegram_watcher.py - 24/7 Autonomous Market Structure Watcher
Watches 24/7 crypto pairs + CME-hours silver with multi-exchange public feeds.
"""
import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from smc_engine import calculate_clean_indicators, get_structural_levels
from brain_db import init_db, save_setup, get_setting
from brain_metrics import fetch_binance_funding_and_oi, fetch_fred_dxy
from brain_scorer import calculate_conviction_score

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8989112896")
STATE_FILE = os.path.join(os.path.dirname(__file__), "alert_state.json")

WATCHLIST = [
    {"symbol": "BTC/USDT", "coinbase_pair": "BTC-USD", "kraken_pair": "XBTUSD", "fsym": "BTC", "is_silver": False},
    {"symbol": "ETH/USDT", "coinbase_pair": "ETH-USD", "kraken_pair": "ETHUSD", "fsym": "ETH", "is_silver": False},
    {"symbol": "SOL/USDT", "coinbase_pair": "SOL-USD", "kraken_pair": "SOLUSD", "fsym": "SOL", "is_silver": False},
    {"symbol": "XRP/USDT", "coinbase_pair": "XRP-USD", "kraken_pair": "XRPUSD", "fsym": "XRP", "is_silver": False},
    {"symbol": "SILVER/USDT", "coinbase_pair": None, "kraken_pair": None, "fsym": "XAG", "is_silver": True}
]

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

def load_alert_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_alert_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Error saving state: {e}")

def send_telegram_alert(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        r = requests.post(url, json=payload, timeout=8)
        return r.status_code == 200
    except Exception as e:
        print(f"Failed to post to Telegram: {e}")
        return False

def fetch_klines_and_spot(asset):
    headers = {"User-Agent": "Mozilla/5.0"}
    df = None
    spot = 0.0

    if asset["is_silver"]:
        if not is_silver_market_open():
            return None, -1.0 # Signal market closed
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/SI=F?interval=15m&range=2d"
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
            df = df_temp.dropna().reset_index(drop=True)
            if not df.empty:
                spot = float(df['close'].iloc[-1])
        except Exception as e:
            print(f"Error fetching Silver from Yahoo Finance: {e}")
            pass
    else:
        # 1. Primary: Coinbase Public Candles (15m)
        if asset["coinbase_pair"]:
            try:
                pair = asset["coinbase_pair"]
                url = f"https://api.exchange.coinbase.com/products/{pair}/candles?granularity=900"
                r = requests.get(url, headers=headers, timeout=5)
                if r.status_code == 200:
                    raw = r.json()
                    if isinstance(raw, list) and len(raw) > 0:
                        df_temp = pd.DataFrame(raw, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                        df = df_temp.sort_values('timestamp').reset_index(drop=True)
                        spot = float(df['close'].iloc[-1])
            except Exception:
                pass

        # 2. Secondary: Kraken Public OHLC (15m)
        if (df is None or df.empty) and asset["kraken_pair"]:
            try:
                k_pair = asset["kraken_pair"]
                url = f"https://api.kraken.com/0/public/OHLC?pair={k_pair}&interval=15"
                r = requests.get(url, headers=headers, timeout=5).json()
                result = r.get("result", {})
                for k in result:
                    if k != "last":
                        raw = result[k]
                        rows = [{'timestamp': int(row[0]), 'open': float(row[1]), 'high': float(row[2]), 'low': float(row[3]), 'close': float(row[4]), 'volume': float(row[6])} for row in raw]
                        df = pd.DataFrame(rows)
                        spot = float(df['close'].iloc[-1])
                        break
            except Exception:
                pass

    if df is not None and not df.empty and spot > 0:
        df = calculate_clean_indicators(df)
        return df, spot

    return None, spot

def run_watcher():
    print(f"🚀 Running Multi-Asset Market Watcher at {datetime.now(timezone.utc).isoformat()}...")
    init_db() # Ensure the SQLite database is initialized
    state = load_alert_state()
    now_ts = time.time()
    COOLDOWN_SECS = 3 * 3600

    # 1. Fetch pre-requisite macro/external data
    # Fetch FRED DXY proxy
    fred_key = get_setting("fred_api_key")
    fred_data = fetch_fred_dxy(fred_key) if fred_key else {"dxy_value": 100.0, "dxy_trend": "FLAT", "dxy_sma_5d": 100.0}
    
    # Fetch Binance BTC metrics
    btc_metrics = fetch_binance_funding_and_oi("BTCUSDT")

    for asset in WATCHLIST:
        sym = asset["symbol"]
        try:
            df, spot = fetch_klines_and_spot(asset)
            
            # Handle Closed Commodity Session
            if asset["is_silver"] and spot == -1.0:
                print(f"[{sym:11}] ⏸️  Market Closed (COMEX Weekend Session)")
                continue

            if df is None or spot <= 0.0:
                print(f"⚠️ Could not fetch market data for {sym}")
                continue

            levels = get_structural_levels(df, sym, spot)
            s_plan = levels['short_plan']
            l_plan = levels['long_plan']
            atr = levels['atr']
            dec = levels['decimals']
            vwap = levels['vwap']
            rsi = levels['rsi']

            dist_to_short = s_plan['entry'] - spot
            dist_to_long = spot - l_plan['entry']
            threshold = 0.8 * atr

            print(f"[{sym:11}] Spot: ${spot:<10.{dec}f} | Supply: ${s_plan['entry']:<10.{dec}f} (Δ ${dist_to_short:<7.{dec}f}) | Demand: ${l_plan['entry']:<10.{dec}f} (Δ ${dist_to_long:<7.{dec}f}) | RSI: {rsi:<4} | VWAP: ${vwap:.{dec}f}")

            # FVG alignments in last 5 candles
            bullish_fvg_present = any(df['bullish_fvg'].iloc[-5:]) if 'bullish_fvg' in df.columns else False
            bearish_fvg_present = any(df['bearish_fvg'].iloc[-5:]) if 'bearish_fvg' in df.columns else False

            # Short Trigger
            if 0 <= dist_to_short <= threshold:
                zone_key = f"{sym}_SHORT"
                if now_ts - state.get(zone_key, 0) > COOLDOWN_SECS:
                    # Gather setup specific metrics
                    f_rate = btc_metrics["funding_rate"] if "BTC" in sym.upper() else None
                    oi_t = btc_metrics["oi_trend"] if "BTC" in sym.upper() else None
                    dxy_t = fred_data["dxy_trend"] if "SILVER" in sym.upper() or "XAG" in sym.upper() else None
                    
                    score, reasons = calculate_conviction_score(
                        sym, "SHORT", s_plan['entry'], vwap, rsi, bearish_fvg_present,
                        funding_rate=f_rate, oi_trend=oi_t, dxy_trend=dxy_t
                    )
                    
                    # Log to database
                    save_setup(
                        sym, "SHORT", spot, s_plan['entry'], s_plan['sl'], s_plan['tp'],
                        atr, rsi, vwap, funding_rate=f_rate, open_interest=btc_metrics["current_oi"] if "BTC" in sym.upper() else None,
                        dxy_index=fred_data["dxy_value"] if "SILVER" in sym.upper() or "XAG" in sym.upper() else None,
                        conviction_score=score
                    )
                    
                    reasons_str = "\n".join([f"▫️ *Brain Sentiment:* {r}" for r in reasons])
                    if reasons_str:
                        reasons_str = f"\n{reasons_str}\n"
                    
                    msg = (
                        f"🚨 *TRADE ALERT: {sym} Approaching Supply Ceiling*\n\n"
                        f"▫️ *Spot Price:* `${spot:.{dec}f}`\n"
                        f"▫️ *Supply Entry:* `${s_plan['entry']:.{dec}f}` (Within `${dist_to_short:.{dec}f}`)\n"
                        f"▫️ *Invalidation (SL):* `${s_plan['sl']:.{dec}f}`\n"
                        f"▫️ *Target (TP):* `${s_plan['tp']:.{dec}f}`\n"
                        f"▫️ *Wilder RSI:* `{rsi}` | *Session VWAP:* `${vwap:.{dec}f}`\n"
                        f"▫️ *Conviction Score:* `🔥 {score}%`\n"
                        f"{reasons_str}\n"
                        f"⚡ *Action:* Prepare Short Playbook on BTCC."
                    )
                    if send_telegram_alert(msg):
                        state[zone_key] = now_ts

            # Long Trigger
            if 0 <= dist_to_long <= threshold:
                zone_key = f"{sym}_LONG"
                if now_ts - state.get(zone_key, 0) > COOLDOWN_SECS:
                    # Gather setup specific metrics
                    f_rate = btc_metrics["funding_rate"] if "BTC" in sym.upper() else None
                    oi_t = btc_metrics["oi_trend"] if "BTC" in sym.upper() else None
                    dxy_t = fred_data["dxy_trend"] if "SILVER" in sym.upper() or "XAG" in sym.upper() else None
                    
                    score, reasons = calculate_conviction_score(
                        sym, "LONG", l_plan['entry'], vwap, rsi, bullish_fvg_present,
                        funding_rate=f_rate, oi_trend=oi_t, dxy_trend=dxy_t
                    )
                    
                    # Log to database
                    save_setup(
                        sym, "LONG", spot, l_plan['entry'], l_plan['sl'], l_plan['tp'],
                        atr, rsi, vwap, funding_rate=f_rate, open_interest=btc_metrics["current_oi"] if "BTC" in sym.upper() else None,
                        dxy_index=fred_data["dxy_value"] if "SILVER" in sym.upper() or "XAG" in sym.upper() else None,
                        conviction_score=score
                    )
                    
                    reasons_str = "\n".join([f"▫️ *Brain Sentiment:* {r}" for r in reasons])
                    if reasons_str:
                        reasons_str = f"\n{reasons_str}\n"
                    
                    msg = (
                        f"🚨 *TRADE ALERT: {sym} Approaching Demand Floor*\n\n"
                        f"▫️ *Spot Price:* `${spot:.{dec}f}`\n"
                        f"▫️ *Demand Entry:* `${l_plan['entry']:.{dec}f}` (Within `${dist_to_long:.{dec}f}`)\n"
                        f"▫️ *Invalidation (SL):* `${l_plan['sl']:.{dec}f}`\n"
                        f"▫️ *Target (TP):* `${l_plan['tp']:.{dec}f}`\n"
                        f"▫️ *Wilder RSI:* `{rsi}` | *Session VWAP:* `${vwap:.{dec}f}`\n"
                        f"▫️ *Conviction Score:* `🔥 {score}%`\n"
                        f"{reasons_str}\n"
                        f"⚡ *Action:* Prepare Long Playbook on BTCC."
                    )
                    if send_telegram_alert(msg):
                        state[zone_key] = now_ts

        except Exception as e:
            print(f"Error checking {sym}: {e}")

    save_alert_state(state)
    print("✅ Multi-Asset Watcher scan complete.")

if __name__ == "__main__":
    run_watcher()
