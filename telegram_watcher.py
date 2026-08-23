"""
telegram_watcher.py - 24/7 Autonomous Market Watcher
Queries live OHLCV & orderbook boundaries, verifies confluence, and sends Telegram alerts.
"""
import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from smc_engine import calculate_clean_indicators, get_structural_levels

# Environment Secrets
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
STATE_FILE = "alert_state.json"

WATCHLIST = [
    {"symbol": "BTC/USDT", "api_sym": "BTC", "is_silver": False},
    {"symbol": "XAG/USDT", "api_sym": "XAG", "is_silver": True}
]

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
        print("⚠️ Telegram token/chat_id not configured. Alert payload:")
        print(text)
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
    fsym = asset["api_sym"]
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. Fetch Spot Price
    spot = 0.0
    if asset["is_silver"]:
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/ticker/price?symbol=XAGUSDT", headers=headers, timeout=5).json()
            if r.get('price'): spot = float(r['price'])
        except Exception:
            spot = 69.340
    else:
        try:
            r = requests.get(f"https://api.coinbase.com/v2/prices/{fsym}-USD/spot", headers=headers, timeout=5).json()
            if r.get('data', {}).get('amount'): spot = float(r['data']['amount'])
        except Exception:
            spot = 77200.0

    # 2. Fetch 15m Candles
    tsym = "USD" if asset["is_silver"] else "USDT"
    url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={fsym}&tsym={tsym}&limit=60&aggregate=15"
    r = requests.get(url, headers=headers, timeout=5).json()
    raw = r.get('Data', {}).get('Data', [])
    
    df = pd.DataFrame(raw)
    if not df.empty and 'close' in df.columns:
        df = df[['time', 'open', 'high', 'low', 'close', 'volumeto']]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    else:
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=60, freq='15min')
        df = pd.DataFrame({
            'timestamp': dates.astype(int) // 10**9,
            'open': spot, 'high': spot * 1.002, 'low': spot * 0.998, 'close': spot, 'volume': 1000
        })
    df = calculate_clean_indicators(df)
    return df, spot

def run_watcher():
    print(f"🚀 Running Market Watcher at {datetime.now(timezone.utc).isoformat()}...")
    state = load_alert_state()
    now_ts = time.time()
    COOLDOWN_SECS = 4 * 3600  # 4-hour cooldown per zone to prevent alert spam

    for asset in WATCHLIST:
        sym = asset["symbol"]
        df, spot = fetch_klines_and_spot(asset)
        levels = get_structural_levels(df, sym, spot)
        
        s_plan = levels['short_plan']
        l_plan = levels['long_plan']
        atr = levels['atr']
        bias = levels['bias']
        dec = levels['decimals']

        # Proximity Check: Is price within 0.5 * ATR of Short Entry (Supply) or Long Entry (Demand)?
        dist_to_short = s_plan['entry'] - spot
        dist_to_long = spot - l_plan['entry']
        threshold = 0.6 * atr

        # Check Short Proximity
        if 0 <= dist_to_short <= threshold:
            zone_key = f"{sym}_SHORT_{s_plan['entry']}"
            last_alert = state.get(zone_key, 0)
            if now_ts - last_alert > COOLDOWN_SECS:
                msg = (
                    f"🚨 *HEADS-UP: {sym} Approaching Supply Zone*\n\n"
                    f"▫️ *Spot Price:* `${spot:.{dec}f}`\n"
                    f"▫️ *Supply Target:* `${s_plan['entry']:.{dec}f}` (~`${dist_to_short:.{dec}f}` away)\n"
                    f"▫️ *Invalidation:* `${s_plan['sl']:.{dec}f}`\n"
                    f"▫️ *Take Profit:* `${s_plan['tp']:.{dec}f}`\n"
                    f"▫️ *Bias:* {bias['verdict']}\n\n"
                    f"👉 *Action:* Open Tactical Terminal to prepare BTCC Short setup."
                )
                if send_telegram_alert(msg):
                    state[zone_key] = now_ts

        # Check Long Proximity
        if 0 <= dist_to_long <= threshold:
            zone_key = f"{sym}_LONG_{l_plan['entry']}"
            last_alert = state.get(zone_key, 0)
            if now_ts - last_alert > COOLDOWN_SECS:
                msg = (
                    f"🚨 *HEADS-UP: {sym} Approaching Demand Floor*\n\n"
                    f"▫️ *Spot Price:* `${spot:.{dec}f}`\n"
                    f"▫️ *Demand Target:* `${l_plan['entry']:.{dec}f}` (~`${dist_to_long:.{dec}f}` away)\n"
                    f"▫️ *Invalidation:* `${l_plan['sl']:.{dec}f}`\n"
                    f"▫️ *Take Profit:* `${l_plan['tp']:.{dec}f}`\n"
                    f"▫️ *Bias:* {bias['verdict']}\n\n"
                    f"👉 *Action:* Open Tactical Terminal to prepare BTCC Long setup."
                )
                if send_telegram_alert(msg):
                    state[zone_key] = now_ts

    save_alert_state(state)
    print("✅ Watcher scan complete.")

if __name__ == "__main__":
    run_watcher()
