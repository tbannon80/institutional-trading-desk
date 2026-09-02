import os
import sys
import time
import requests
import logging
from datetime import datetime, timezone

# Add parent path to import sibling files easily
sys.path.append(os.path.dirname(__file__))

from brain_db import (
    get_connection, get_active_setups, update_setup_status, 
    get_resolved_setups_count, get_setting, init_db
)

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
    Fetches current spot price from standard public endpoints.
    """
    sym = symbol.upper()
    headers = {"User-Agent": "Mozilla/5.0"}
    
    if "XAG" in sym or "SILVER" in sym:
        # Silver COMEX hourly/min feed from cryptocompare
        url = "https://min-api.cryptocompare.com/data/price?fsym=XAG&tsyms=USD"
        try:
            r = requests.get(url, headers=headers, timeout=5).json()
            return float(r.get("USD", 0.0))
        except Exception:
            pass
    else:
        # Crypto spot from Coinbase Public API
        coinbase_pair = sym.replace("/", "-")
        url = f"https://api.exchange.coinbase.com/products/{coinbase_pair}/ticker"
        try:
            r = requests.get(url, headers=headers, timeout=5).json()
            return float(r.get("price", 0.0))
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
                return float(result[k]["c"][0])
        except Exception:
            pass
            
    return None

def check_resolved_setups_and_evaluate():
    """
    Periodically triggered when a new trade resolves.
    Proposes optimizing adjustments using Gemini Model.
    """
    try:
        resolved_count = get_resolved_setups_count()
        # Trigger evaluation every 10 resolved trades (keeps user updated regularly)
        if resolved_count > 0 and resolved_count % 10 == 0:
            trigger_gemini_brain_evaluation(resolved_count)
    except Exception as e:
        logger.error(f"Error checking resolved setups for evaluation: {e}")

def trigger_gemini_brain_evaluation(resolved_count):
    """
    Queries SQLite database, grabs resolved setups, and sends them to Gemini to evaluate
    and propose weight optimization parameter modifications.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        logger.warning("Gemini API key missing, skipping weight optimization evaluation.")
        return

    # Load resolved setups
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, setup_type, entry_price, sl_price, tp_price, status, conviction_score, rsi, vwap, funding_rate, dxy_index
        FROM setups 
        WHERE status IN ('WIN', 'LOSS') 
        ORDER BY timestamp DESC LIMIT 30
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows or len(rows) < 5:
        return # Not enough samples to optimize

    resolved_setups = [dict(r) for r in rows]

    # Get current weights
    w_rsi = float(get_setting("w_rsi", 25.0))
    w_vwap = float(get_setting("w_vwap", 25.0))
    w_fvg = float(get_setting("w_fvg", 15.0))
    w_external = float(get_setting("w_external", 35.0))

    prompt = (
        f"You are the Statistical Evaluator of Bannon Trading Desk Brain.\n"
        f"Your task is to analyze the performance of our recent {len(resolved_setups)} trade alerts (Wins and Losses) "
        f"and propose optimization adjustments for the dynamic factor weights: RSI weight, Session VWAP weight, "
        f"Fair Value Gap (FVG) weight, and External Market/Macro weight (Funding/DXY).\n\n"
        f"Current weights:\n"
        f"- RSI Weight (w_rsi): {w_rsi}\n"
        f"- VWAP Weight (w_vwap): {w_vwap}\n"
        f"- FVG Weight (w_fvg): {w_fvg}\n"
        f"- External Weight (w_external): {w_external}\n"
        f"(Total weights must always sum to 100.0).\n\n"
        f"Recent Resolved Trade Setups Data:\n"
        f"{resolved_setups}\n\n"
        f"Instructions:\n"
        f"1. Perform statistical analysis of which indicators aligned best with Wins vs Losses.\n"
        f"2. Suggest new optimized weights (summing to exactly 100.0) that would have increased the average conviction score "
        f"of successful trades and decreased the score of failures.\n"
        f"3. Format your response clearly. You MUST include a single JSON block at the bottom of your output that the "
        f"system can parse to update settings if approved. The JSON block format MUST be exactly:\n"
        f"```json\n"
        f"{{\n"
        f"  \"w_rsi\": float,\n"
        f"  \"w_vwap\": float,\n"
        f"  \"w_fvg\": float,\n"
        f"  \"w_external\": float\n"
        f"}}\n"
        f"```\n\n"
        f"Explain your logic first to the user, comparing wins/losses, and end with the JSON block code."
    )

    # Call Gemini Model
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
    if gemini_key:
        url += f"?key={gemini_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        if r.status_code == 200:
            resp_json = r.json()
            if "candidates" in resp_json and resp_json["candidates"]:
                eval_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                
                # Send the analysis and adjustment proposal to the user's Telegram!
                msg = (
                    f"🧠 *BANNON TRADING DESK: BRAIN OPTIMIZATION PROPOSAL*\n\n"
                    f"The Brain has analyzed the last {resolved_count} resolved setups.\n\n"
                    f"{eval_text[:3000]}\n\n"
                    f"💡 *Action:* Click a button below to apply this optimized parameter adjustment proposal."
                )
                
                # We'll send this with interactive yes/no callback buttons!
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
    Tracks entry/TP/SL hit events for active and pending setups.
    """
    logger.info("Executing periodic brain active setup tracking tick...")
    active_setups = get_active_setups()
    if not active_setups:
        return
        
    now_ts = datetime.now(timezone.utc)
    
    for setup in active_setups:
        symbol = setup["symbol"]
        setup_id = setup["setup_id"]
        setup_type = setup["setup_type"]
        entry = setup["entry_price"]
        sl = setup["sl_price"]
        tp = setup["tp_price"]
        status = setup["status"]
        highest = setup["highest_reached"]
        lowest = setup["lowest_reached"]
        
        spot = get_current_price(symbol)
        if spot is None:
            logger.warning(f"Could not get spot price for {symbol} during tracking.")
            continue
            
        logger.info(f"Tracking setup #{setup_id} {symbol} {setup_type} | Status: {status} | Spot: {spot} | Entry: {entry} | SL: {sl} | TP: {tp}")
        
        # 1. PENDING Status Check -> Hits Entry to go ACTIVE
        if status == "PENDING":
            activated = False
            if setup_type == "LONG":
                # For Long, price touches or drops below entry to activate
                if spot <= entry:
                    activated = True
            else: # SHORT
                # For Short, price touches or rises above entry to activate
                if spot >= entry:
                    activated = True
                    
            if activated:
                update_setup_status(
                    setup_id, "ACTIVE", 
                    highest=spot, lowest=spot
                )
                send_telegram_notification(
                    f"🟢 *Setup Activated: {symbol} {setup_type}*\n"
                    f"▫️ Price has touched entry target of `${entry:.2f}` (Current spot: `${spot:.2f}`).\n"
                    f"▫️ Brain is now tracking setup resolution targets:\n"
                    f"🎯 *Take-Profit:* `${tp:.2f}` | 🛡️ *Stop-Loss:* `${sl:.2f}`"
                )
                continue
                
            # Expiry Check (24 hours in PENDING)
            setup_time = datetime.fromisoformat(setup["timestamp"])
            if (now_ts - setup_time).total_seconds() > 24 * 3600:
                update_setup_status(setup_id, "EXPIRED", resolution_time=now_ts.isoformat())
                logger.info(f"Setup #{setup_id} EXPIRED in PENDING status.")
                
        # 2. ACTIVE Status Check -> Hits TP or SL
        elif status == "ACTIVE":
            highest = max(highest if highest is not None else spot, spot)
            lowest = min(lowest if lowest is not None else spot, spot)
            
            resolved = False
            new_status = "ACTIVE"
            msg = ""
            
            if setup_type == "LONG":
                if spot >= tp:
                    new_status = "WIN"
                    resolved = True
                    msg = (
                        f"🎯 *BRAIN TARGET REACHED: {symbol} LONG WIN*\n\n"
                        f"▫️ *Entry Price:* `${entry:.2f}`\n"
                        f"▫️ *Target TP hit:* `${tp:.2f}` (Current: `${spot:.2f}`)\n"
                        f"▫️ *Setup Conviction:* `{setup['conviction_score']}%`\n"
                        f"▫️ *Highest Reached:* `${highest:.2f}`\n\n"
                        f"💰 Excellent resolution! Setup verified as Win."
                    )
                elif spot <= sl:
                    new_status = "LOSS"
                    resolved = True
                    msg = (
                        f"🛡️ *BRAIN ALERT: {symbol} LONG STOPPED OUT*\n\n"
                        f"▫️ *Entry Price:* `${entry:.2f}`\n"
                        f"▫️ *Invalidation SL hit:* `${sl:.2f}` (Current: `${spot:.2f}`)\n"
                        f"▫️ *Setup Conviction:* `{setup['conviction_score']}%`\n"
                        f"▫️ *Lowest Reached:* `${lowest:.2f}`\n\n"
                        f"📉 Setup resolved as Loss."
                    )
            else: # SHORT
                if spot <= tp:
                    new_status = "WIN"
                    resolved = True
                    msg = (
                        f"🎯 *BRAIN TARGET REACHED: {symbol} SHORT WIN*\n\n"
                        f"▫️ *Entry Price:* `${entry:.2f}`\n"
                        f"▫️ *Target TP hit:* `${tp:.2f}` (Current: `${spot:.2f}`)\n"
                        f"▫️ *Setup Conviction:* `{setup['conviction_score']}%`\n"
                        f"▫️ *Lowest Reached:* `${lowest:.2f}`\n\n"
                        f"💰 Excellent resolution! Setup verified as Win."
                    )
                elif spot >= sl:
                    new_status = "LOSS"
                    resolved = True
                    msg = (
                        f"🛡️ *BRAIN ALERT: {symbol} SHORT STOPPED OUT*\n\n"
                        f"▫️ *Entry Price:* `${entry:.2f}`\n"
                        f"▫️ *Invalidation SL hit:* `${sl:.2f}` (Current: `${spot:.2f}`)\n"
                        f"▫️ *Setup Conviction:* `{setup['conviction_score']}%`\n"
                        f"▫️ *Highest Reached:* `${highest:.2f}`\n\n"
                        f"📉 Setup resolved as Loss."
                    )
                    
            if resolved:
                update_setup_status(
                    setup_id, new_status, 
                    highest=highest, lowest=lowest, 
                    resolution_time=now_ts.isoformat()
                )
                send_telegram_notification(msg)
                
                # Check resolved metrics count and evaluate improvements
                check_resolved_setups_and_evaluate()
            else:
                # Update highest/lowest periodically in DB
                update_setup_status(setup_id, "ACTIVE", highest=highest, lowest=lowest)
                
                # Expiry Check (48 hours in ACTIVE)
                setup_time = datetime.fromisoformat(setup["timestamp"])
                if (now_ts - setup_time).total_seconds() > 48 * 3600:
                    update_setup_status(setup_id, "EXPIRED", highest=highest, lowest=lowest, resolution_time=now_ts.isoformat())
                    send_telegram_notification(
                        f"⏳ *Setup Expired: {symbol} {setup_type}*\n"
                        f"▫️ Trade was open for 48 hours without hitting TP or SL.\n"
                        f"▫️ Marked as expired. Entry: `${entry:.2f}` | Current: `${spot:.2f}`"
                    )

if __name__ == "__main__":
    init_db()
    track_active_setups_tick()
