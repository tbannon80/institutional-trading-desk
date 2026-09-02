import os
import sys
import time
import requests
import logging
import pandas as pd
from datetime import datetime, timezone

# Add parent path to import sibling files easily
sys.path.append(os.path.dirname(__file__))

from brain_db import (
    get_connection, get_active_setups, update_setup_status, 
    get_resolved_setups_count, get_setting, init_db
)
from smc_engine import is_silver_market_open

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8989112896")

def send_telegram_notification(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing in brain tracker.")
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
        logger.error(f"Failed to send telegram notification: {e}")
        return False

def get_current_price(symbol):
    """
    Fetches reliable spot price from primary verified endpoints.
    Guarantees no 0.0 returns for Silver or BTC.
    """
    sym = symbol.upper()
    headers = {"User-Agent": "Mozilla/5.0"}
    
    if "XAG" in sym or "SILVER" in sym:
        # 1. Primary Yahoo Finance COMEX Silver (SI=F)
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/SI=F?interval=1m&range=1d"
            r = requests.get(url, headers=headers, timeout=5).json()
            meta = r['chart']['result'][0]['meta']
            price = meta.get('regularMarketPrice') or meta.get('chartPreviousClose')
            if price and float(price) > 0.0:
                return float(price)
        except Exception:
            pass
            
        # 2. Fallback to GoldAPI or Cryptocompare only if > 0
        try:
            url = "https://min-api.cryptocompare.com/data/price?fsym=XAG&tsyms=USD"
            r = requests.get(url, headers=headers, timeout=5).json()
            p = float(r.get("USD", 0.0))
            if p > 0.0:
                return p
        except Exception:
            pass
            
        return None
    else:
        # Crypto spot from Coinbase Public API
        coinbase_pair = sym.replace("/", "-")
        url = f"https://api.exchange.coinbase.com/products/{coinbase_pair}/ticker"
        try:
            r = requests.get(url, headers=headers, timeout=5).json()
            price = float(r.get("price", 0.0))
            if price > 0.0:
                return price
        except Exception:
            pass
            
        # Fallback to Kraken public ticker
        kraken_pair = sym.replace("/", "")
        if kraken_pair == "BTCUSDT":
            kraken_pair = "XBTUSD"
        url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_pair}"
        try:
            r = requests.get(url, headers=headers, timeout=5).json()
            result = r.get("result", {})
            for k in result:
                p = float(result[k]["c"][0])
                if p > 0.0:
                    return p
        except Exception:
            pass
            
    return None

def fetch_recent_15m_bars(symbol: str, limit: int = 40) -> pd.DataFrame:
    """
    Pulls complete 15-minute OHLC bar arrays for full candle wick evaluation.
    """
    sym = symbol.upper()
    is_silver = "XAG" in sym or "SILVER" in sym
    headers = {"User-Agent": "Mozilla/5.0"}
    
    if is_silver:
        if not is_silver_market_open():
            return pd.DataFrame()
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/SI=F?interval=15m&range=5d"
            r = requests.get(url, headers=headers, timeout=5).json()
            res = r['chart']['result'][0]
            timestamps = res['timestamp']
            quote = res['indicators']['quote'][0]
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': quote['open'],
                'high': quote['high'],
                'low': quote['low'],
                'close': quote['close'],
                'volume': quote['volume']
            }).dropna().reset_index(drop=True)
            return df.tail(limit)
        except Exception as e:
            logger.error(f"Error fetching Silver 15m bars: {e}")
            return pd.DataFrame()
    else:
        # BTC from Coinbase (granularity 900 = 15m)
        try:
            url = f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=900"
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                raw = r.json()
                if isinstance(raw, list) and len(raw) > 0:
                    df = pd.DataFrame(raw, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    return df.tail(limit)
        except Exception as e:
            logger.error(f"Error fetching BTC 15m bars from Coinbase: {e}")
            
        # Fallback to MEXC
        try:
            url = "https://api.mexc.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=40"
            r = requests.get(url, headers=headers, timeout=5).json()
            if isinstance(r, list) and len(r) > 0:
                rows = [{'timestamp': int(row[0]) // 1000, 'open': float(row[1]), 'high': float(row[2]), 'low': float(row[3]), 'close': float(row[4]), 'volume': float(row[5])} for row in r]
                return pd.DataFrame(rows)
        except Exception:
            pass
            
    return pd.DataFrame()

def check_resolved_setups_and_evaluate():
    """
    Periodically triggered when a trade resolves.
    Proposes dynamic weight optimization adjustments using Gemini.
    """
    try:
        resolved_count = get_resolved_setups_count()
        if resolved_count > 0 and resolved_count % 10 == 0:
            trigger_gemini_brain_evaluation(resolved_count)
    except Exception as e:
        logger.error(f"Error checking resolved setups for evaluation: {e}")

def trigger_gemini_brain_evaluation(resolved_count):
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        logger.warning("Gemini API key missing, skipping weight optimization evaluation.")
        return

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, setup_type, entry_price, sl_price, tp_price, status, conviction_score, rsi, vwap, funding_rate, dxy_index
        FROM setups 
        WHERE status IN ('WIN', 'FULL_WIN', 'HALF_WIN_BE', 'LOSS') 
        ORDER BY timestamp DESC LIMIT 30
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows or len(rows) < 5:
        return

    resolved_setups = [dict(r) for r in rows]

    w_rsi = float(get_setting("w_rsi", 25.0))
    w_vwap = float(get_setting("w_vwap", 25.0))
    w_fvg = float(get_setting("w_fvg", 15.0))
    w_external = float(get_setting("w_external", 35.0))

    prompt = (
        f"You are the Statistical Evaluator of Bannon Trading Desk Brain.\n"
        f"Your task is to analyze the performance of our recent {len(resolved_setups)} trade alerts "
        f"(Full Wins, Break-Even Half-Wins, and Losses) and propose optimization adjustments for dynamic factor weights: "
        f"RSI, Session VWAP, FVG, and External Macro (Funding/DXY).\n\n"
        f"Current weights:\n"
        f"- RSI Weight (w_rsi): {w_rsi}\n"
        f"- VWAP Weight (w_vwap): {w_vwap}\n"
        f"- FVG Weight (w_fvg): {w_fvg}\n"
        f"- External Weight (w_external): {w_external}\n"
        f"(Total weights must sum to 100.0).\n\n"
        f"Recent Resolved Trade Setups Data:\n"
        f"{resolved_setups}\n\n"
        f"Instructions:\n"
        f"1. Perform statistical analysis of indicator alignment with Wins vs Losses.\n"
        f"2. Suggest new optimized weights (summing to 100.0) that would maximize conviction for wins and filter out losses.\n"
        f"3. Include a single JSON block at the bottom:\n"
        f"```json\n"
        f"{{\n"
        f"  \"w_rsi\": float,\n"
        f"  \"w_vwap\": float,\n"
        f"  \"w_fvg\": float,\n"
        f"  \"w_external\": float\n"
        f"}}\n"
        f"```"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={gemini_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        if r.status_code == 200:
            resp_json = r.json()
            if "candidates" in resp_json and resp_json["candidates"]:
                eval_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                msg = (
                    f"🧠 *BANNON TRADING DESK: BRAIN OPTIMIZATION PROPOSAL*\n\n"
                    f"The Brain has analyzed the last {resolved_count} resolved setups.\n\n"
                    f"{eval_text[:3000]}\n\n"
                    f"💡 *Action:* Click a button below to apply this optimized parameter adjustment proposal."
                )
                send_proposal_with_buttons(msg)
    except Exception as e:
        logger.error(f"Error calling Gemini for optimization evaluation: {e}")

def send_proposal_with_buttons(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [
                [
                    {"text": "✅ Apply Proposal", "callback_data": "brain_apply_weights"},
                    {"text": "❌ Dismiss", "callback_data": "brain_dismiss_weights"}
                ]
            ]
        }
    }
    try:
        requests.post(url, json=payload, timeout=8)
    except Exception as e:
        logger.error(f"Error sending proposal telegram buttons: {e}")

def track_active_setups_tick():
    """
    Core polling function called every 15 minutes.
    Evaluates complete 15-minute OHLC candle wick arrays (High_t, Low_t, Close_t).
    
    Multi-Tier Risk Rules (Specifications 3 & 4):
    - Activation: Low_t <= Entry (for Longs) or High_t >= Entry (for Shorts).
    - TP1 Hit: Closes 50% position, moves Stop-Loss to Entry (Break-Even).
    - TP2 Hit: Resolves as FULL_WIN (100% target achieved).
    - Conservative Priority Rule: If both SL and TP thresholds breach on the same 15m candle,
      Stop-Loss takes precedence (worst-case path sequencing).
    """
    logger.info("Executing periodic brain active setup tracking tick (Complete OHLC Wick Tracking)...")
    active_setups = get_active_setups()
    if not active_setups:
        return
        
    now_ts = datetime.now(timezone.utc)
    
    for setup in active_setups:
        symbol = setup["symbol"]
        setup_id = setup["setup_id"]
        setup_type = setup["setup_type"]
        entry = float(setup["entry_price"])
        sl = float(setup["sl_price"])
        tp1 = float(setup["tp1_price"]) if setup["tp1_price"] and setup["tp1_price"] > 0 else float(setup["tp_price"])
        tp2 = float(setup["tp2_price"]) if setup["tp2_price"] and setup["tp2_price"] > 0 else float(setup["tp_price"])
        status = setup["status"]
        tp1_hit = bool(setup["tp1_hit"])
        be_price = float(setup["be_price"]) if setup["be_price"] else None
        highest = setup["highest_reached"]
        lowest = setup["lowest_reached"]
        atr = float(setup["atr"]) if setup["atr"] else 1.0
        
        # Pull recent 15-minute OHLC bars
        bars_df = fetch_recent_15m_bars(symbol, limit=20)
        if bars_df.empty:
            # If historical bars unavailable, fallback to single spot check
            spot = get_current_price(symbol)
            if spot is None or spot <= 0.0:
                continue
            bars_df = pd.DataFrame([{
                'timestamp': time.time(),
                'open': spot,
                'high': spot,
                'low': spot,
                'close': spot,
                'volume': 100.0
            }])

        setup_start_dt = datetime.fromisoformat(setup["timestamp"])
        setup_start_epoch = setup_start_dt.timestamp()

        # Filter bars that occurred at or after setup creation
        relevant_bars = bars_df[bars_df['timestamp'] >= (setup_start_epoch - 900)]
        if relevant_bars.empty:
            relevant_bars = bars_df.tail(1)

        dec = 3 if "XAG" in symbol or "SILVER" in symbol else 1

        for _, bar in relevant_bars.iterrows():
            bar_high = float(bar['high'])
            bar_low = float(bar['low'])
            bar_close = float(bar['close'])
            
            highest = max(highest if highest is not None else bar_high, bar_high)
            lowest = min(lowest if lowest is not None else bar_low, bar_low)

            # 1. PENDING Status Check -> Wick touches Entry
            if status == "PENDING":
                activated = False
                if setup_type == "LONG":
                    if bar_low <= entry:
                        activated = True
                else: # SHORT
                    if bar_high >= entry:
                        activated = True

                if activated:
                    status = "ACTIVE"
                    update_setup_status(
                        setup_id, "ACTIVE", 
                        highest=highest, lowest=lowest
                    )
                    send_telegram_notification(
                        f"🟢 *Setup Activated: {symbol} {setup_type}*\n"
                        f"▫️ Candle wick filled entry target at `${entry:.{dec}f}`.\n"
                        f"▫️ *Active Management Targets:*\n"
                        f"🎯 *TP1 (50% Close + BE):* `${tp1:.{dec}f}`\n"
                        f"🎯 *TP2 (Runner Target):* `${tp2:.{dec}f}`\n"
                        f"🛡️ *Market Structure Invalidation (SL):* `${sl:.{dec}f}`"
                    )

            # 2. ACTIVE or TP1_HIT Status -> Check Resolution Targets
            if status in ("ACTIVE", "TP1_HIT"):
                current_sl = be_price if (tp1_hit and be_price is not None) else sl
                resolved = False
                new_status = status
                msg = ""

                if setup_type == "LONG":
                    # Conservative Collision Rule: If both SL and TP are wicked on the same 15m bar, prioritize SL
                    if bar_low <= current_sl and bar_high >= tp2:
                        resolved = True
                        if tp1_hit:
                            new_status = "HALF_WIN_BE"
                            msg = (
                                f"🛡️ *BRAIN TRADE RESOLVED: {symbol} LONG RUNNER AT BREAK-EVEN*\n\n"
                                f"▫️ *Entry Price:* `${entry:.{dec}f}` | *TP1 Secured:* `${tp1:.{dec}f}`\n"
                                f"▫️ *Exit:* Remaining 50% stopped at Break-Even `${current_sl:.{dec}f}`.\n"
                                f"💰 Trade resolved with net partial profit (50% TP1 win)."
                            )
                        else:
                            new_status = "LOSS"
                            msg = (
                                f"🛡️ *BRAIN ALERT: {symbol} LONG STOPPED OUT (MSI)*\n\n"
                                f"▫️ *Entry Price:* `${entry:.{dec}f}`\n"
                                f"▫️ *Invalidation SL Hit:* `${current_sl:.{dec}f}` (Bar Low: `${bar_low:.{dec}f}`)\n"
                                f"📉 Setup resolved as Loss."
                            )

                    # Check Stop Loss
                    elif bar_low <= current_sl:
                        resolved = True
                        if tp1_hit:
                            new_status = "HALF_WIN_BE"
                            msg = (
                                f"🛡️ *BRAIN TRADE RESOLVED: {symbol} LONG RUNNER AT BREAK-EVEN*\n\n"
                                f"▫️ *Entry Price:* `${entry:.{dec}f}` | *TP1 Secured:* `${tp1:.{dec}f}`\n"
                                f"▫️ *Exit:* Remaining 50% stopped at Break-Even `${current_sl:.{dec}f}`.\n"
                                f"💰 Trade resolved with net partial profit (50% TP1 win)."
                            )
                        else:
                            new_status = "LOSS"
                            msg = (
                                f"🛡️ *BRAIN ALERT: {symbol} LONG STOPPED OUT (MSI)*\n\n"
                                f"▫️ *Entry Price:* `${entry:.{dec}f}`\n"
                                f"▫️ *Invalidation SL Hit:* `${current_sl:.{dec}f}` (Bar Low: `${bar_low:.{dec}f}`)\n"
                                f"📉 Setup resolved as Loss."
                            )

                    # Check TP2 (Runner Target)
                    elif bar_high >= tp2:
                        resolved = True
                        new_status = "FULL_WIN"
                        msg = (
                            f"🎯 *BRAIN RUNNER TARGET REACHED: {symbol} LONG FULL WIN (TP2)*\n\n"
                            f"▫️ *Entry Price:* `${entry:.{dec}f}`\n"
                            f"▫️ *Runner TP2 Hit:* `${tp2:.{dec}f}` (Bar High: `${bar_high:.{dec}f}`)\n"
                            f"▫️ *Conviction Score:* `{setup['conviction_score']}%`\n"
                            f"💰 Full trade target executed successfully!"
                        )

                    # Check TP1 (Split-Position 50% Exit & Break-Even Move)
                    elif (not tp1_hit) and bar_high >= tp1:
                        tp1_hit = True
                        be_price = round(entry + (0.02 * atr), dec) # Move SL to Break-Even with fee buffer
                        status = "TP1_HIT"
                        update_setup_status(
                            setup_id, "TP1_HIT", 
                            highest=highest, lowest=lowest, 
                            be_price=be_price, tp1_hit=1
                        )
                        send_telegram_notification(
                            f"🎯 *BRAIN TARGET REACHED: {symbol} LONG TP1 HIT*\n\n"
                            f"▫️ *Entry Price:* `${entry:.{dec}f}` | *TP1 Level:* `${tp1:.{dec}f}`\n"
                            f"▫️ *Execution Action:* 50% position closed at profit.\n"
                            f"▫️ *Risk Management:* Stop-Loss moved to Break-Even `${be_price:.{dec}f}`.\n"
                            f"🚀 Remaining 50% runner targeting TP2: `${tp2:.{dec}f}`."
                        )

                else: # SHORT
                    # Conservative Collision Rule
                    if bar_high >= current_sl and bar_low <= tp2:
                        resolved = True
                        if tp1_hit:
                            new_status = "HALF_WIN_BE"
                            msg = (
                                f"🛡️ *BRAIN TRADE RESOLVED: {symbol} SHORT RUNNER AT BREAK-EVEN*\n\n"
                                f"▫️ *Entry Price:* `${entry:.{dec}f}` | *TP1 Secured:* `${tp1:.{dec}f}`\n"
                                f"▫️ *Exit:* Remaining 50% stopped at Break-Even `${current_sl:.{dec}f}`.\n"
                                f"💰 Trade resolved with net partial profit (50% TP1 win)."
                            )
                        else:
                            new_status = "LOSS"
                            msg = (
                                f"🛡️ *BRAIN ALERT: {symbol} SHORT STOPPED OUT (MSI)*\n\n"
                                f"▫️ *Entry Price:* `${entry:.{dec}f}`\n"
                                f"▫️ *Invalidation SL Hit:* `${current_sl:.{dec}f}` (Bar High: `${bar_high:.{dec}f}`)\n"
                                f"📉 Setup resolved as Loss."
                            )

                    # Check Stop Loss
                    elif bar_high >= current_sl:
                        resolved = True
                        if tp1_hit:
                            new_status = "HALF_WIN_BE"
                            msg = (
                                f"🛡️ *BRAIN TRADE RESOLVED: {symbol} SHORT RUNNER AT BREAK-EVEN*\n\n"
                                f"▫️ *Entry Price:* `${entry:.{dec}f}` | *TP1 Secured:* `${tp1:.{dec}f}`\n"
                                f"▫️ *Exit:* Remaining 50% stopped at Break-Even `${current_sl:.{dec}f}`.\n"
                                f"💰 Trade resolved with net partial profit (50% TP1 win)."
                            )
                        else:
                            new_status = "LOSS"
                            msg = (
                                f"🛡️ *BRAIN ALERT: {symbol} SHORT STOPPED OUT (MSI)*\n\n"
                                f"▫️ *Entry Price:* `${entry:.{dec}f}`\n"
                                f"▫️ *Invalidation SL Hit:* `${current_sl:.{dec}f}` (Bar High: `${bar_high:.{dec}f}`)\n"
                                f"📉 Setup resolved as Loss."
                            )

                    # Check TP2 (Runner Target)
                    elif bar_low <= tp2:
                        resolved = True
                        new_status = "FULL_WIN"
                        msg = (
                            f"🎯 *BRAIN RUNNER TARGET REACHED: {symbol} SHORT FULL WIN (TP2)*\n\n"
                            f"▫️ *Entry Price:* `${entry:.{dec}f}`\n"
                            f"▫️ *Runner TP2 Hit:* `${tp2:.{dec}f}` (Bar Low: `${bar_low:.{dec}f}`)\n"
                            f"▫️ *Conviction Score:* `{setup['conviction_score']}%`\n"
                            f"💰 Full trade target executed successfully!"
                        )

                    # Check TP1 (Split-Position 50% Exit & Break-Even Move)
                    elif (not tp1_hit) and bar_low <= tp1:
                        tp1_hit = True
                        be_price = round(entry - (0.02 * atr), dec) # Move SL to Break-Even
                        status = "TP1_HIT"
                        update_setup_status(
                            setup_id, "TP1_HIT", 
                            highest=highest, lowest=lowest, 
                            be_price=be_price, tp1_hit=1
                        )
                        send_telegram_notification(
                            f"🎯 *BRAIN TARGET REACHED: {symbol} SHORT TP1 HIT*\n\n"
                            f"▫️ *Entry Price:* `${entry:.{dec}f}` | *TP1 Level:* `${tp1:.{dec}f}`\n"
                            f"▫️ *Execution Action:* 50% position closed at profit.\n"
                            f"▫️ *Risk Management:* Stop-Loss moved to Break-Even `${be_price:.{dec}f}`.\n"
                            f"🚀 Remaining 50% runner targeting TP2: `${tp2:.{dec}f}`."
                        )

                if resolved:
                    update_setup_status(
                        setup_id, new_status, 
                        highest=highest, lowest=lowest, 
                        resolution_time=now_ts.isoformat()
                    )
                    send_telegram_notification(msg)
                    check_resolved_setups_and_evaluate()
                    break # Cease evaluating bars for this resolved setup

        # 3. Time Expirations
        if status == "PENDING" and (now_ts - setup_start_dt).total_seconds() > 24 * 3600:
            update_setup_status(setup_id, "EXPIRED", highest=highest, lowest=lowest, resolution_time=now_ts.isoformat())
            logger.info(f"Setup #{setup_id} EXPIRED in PENDING status after 24h.")
        elif status in ("ACTIVE", "TP1_HIT") and (now_ts - setup_start_dt).total_seconds() > 48 * 3600:
            final_status = "HALF_WIN_BE" if tp1_hit else "EXPIRED"
            update_setup_status(setup_id, final_status, highest=highest, lowest=lowest, resolution_time=now_ts.isoformat())
            send_telegram_notification(
                f"⏳ *Setup Closed after 48h: {symbol} {setup_type}*\n"
                f"▫️ Position resolved as `{final_status}` after reaching 48h max hold time."
            )

if __name__ == "__main__":
    init_db()
    track_active_setups_tick()

