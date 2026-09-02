import os
import sqlite3
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_NAME = "trading_memory.db"

def get_db_path():
    repo_path = os.getenv("TRADING_REPO_PATH", os.path.dirname(__file__))
    return os.path.join(repo_path, DB_NAME)

def get_connection():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    # Enforce SQLite Write-Ahead Logging (WAL) mode & normal synchronicity for server daemon robustness
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Create setups table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS setups (
        setup_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,            -- 'BTC/USDT' or 'SILVER/USDT'
        setup_type TEXT NOT NULL,        -- 'LONG' or 'SHORT'
        entry_order_type TEXT NOT NULL DEFAULT 'LIMIT', -- 'LIMIT' or 'BREAKOUT'
        spot_price REAL NOT NULL,
        entry_price REAL NOT NULL,
        sl_price REAL NOT NULL,
        tp_price REAL NOT NULL,          -- Full TP / TP2
        tp1_price REAL NOT NULL DEFAULT 0.0, -- 1.0 R:R / Session VWAP (50% exit)
        tp2_price REAL NOT NULL DEFAULT 0.0, -- 2.0+ R:R (Runner exit)
        be_price REAL,                   -- Break-even level after TP1
        tp1_hit INTEGER DEFAULT 0,       -- 1 if TP1 reached, 0 otherwise
        atr REAL NOT NULL,
        rsi REAL NOT NULL,
        vwap REAL NOT NULL,
        htf_regime TEXT DEFAULT 'NEUTRAL', -- 'BULLISH', 'BEARISH', 'NEUTRAL'
        funding_rate REAL,
        open_interest REAL,
        dxy_index REAL,
        status TEXT NOT NULL,            -- 'PENDING', 'ACTIVE', 'TP1_HIT', 'FULL_WIN', 'HALF_WIN_BE', 'LOSS', 'EXPIRED'
        conviction_score REAL NOT NULL,
        resolution_time TEXT,
        highest_reached REAL,
        lowest_reached REAL,
        trigger_msg_id INTEGER
    )
    """)
    
    # Check for existing table column migrations (table_xinfo includes virtual generated columns)
    existing_cols = {row["name"] for row in cursor.execute("PRAGMA table_xinfo(setups);").fetchall()}
    migrations = [
        ("entry_order_type", "TEXT DEFAULT 'LIMIT'"),
        ("tp1_price", "REAL DEFAULT 0.0"),
        ("tp2_price", "REAL DEFAULT 0.0"),
        ("be_price", "REAL"),
        ("tp1_hit", "INTEGER DEFAULT 0"),
        ("htf_regime", "TEXT DEFAULT 'NEUTRAL'"),
        ("id", "INTEGER GENERATED ALWAYS AS (setup_id)"),
        ("side", "TEXT GENERATED ALWAYS AS (setup_type)"),
        ("tp1_target", "REAL GENERATED ALWAYS AS (tp1_price)"),
        ("tp2_target", "REAL GENERATED ALWAYS AS (tp2_price)")
    ]
    for col_name, col_type in migrations:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE setups ADD COLUMN {col_name} {col_type}")
            logger.info(f"Applied migration: Added {col_name} to setups table.")

    # Purge legacy non-scoped assets from tracking and records
    cursor.execute("""
        DELETE FROM setups
        WHERE symbol NOT IN ('BTC/USDT', 'SILVER/USDT', 'BTCUSDT', 'SILVERUSDT', 'TESTBTC')
    """)
    
    # 2. Create settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS brain_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)
    
    conn.commit()
    
    # 3. Initialize default weights and FRED API key
    defaults = {
        "w_rsi": "25.0",
        "w_vwap": "25.0",
        "w_fvg": "15.0",
        "w_external": "35.0",
        "fred_api_key": "ba32dd734abc8235a0ceb07967ab4812"
    }
    
    for key, val in defaults.items():
        cursor.execute("SELECT 1 FROM brain_settings WHERE key = ?", (key,))
        if not cursor.fetchone():
            cursor.execute("INSERT INTO brain_settings (key, value) VALUES (?, ?)", (key, val))
            
    conn.commit()
    conn.close()
    logger.info(f"Database initialized successfully with WAL mode at {get_db_path()}")

def get_setting(key, default=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM brain_settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row["value"]
    return default

def set_setting(key, value):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO brain_settings (key, value)
    VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """, (key, str(value)))
    conn.commit()
    conn.close()

def save_setup(symbol, setup_type, spot, entry, sl, tp, atr, rsi, vwap, 
               tp1_price=None, tp2_price=None, entry_order_type="LIMIT", htf_regime="NEUTRAL",
               funding_rate=None, open_interest=None, dxy_index=None, conviction_score=0.0):
    conn = get_connection()
    cursor = conn.cursor()
    
    tp1 = tp1_price if tp1_price is not None else tp
    tp2 = tp2_price if tp2_price is not None else tp
    
    # Check if a pending or active setup with similar entry already exists to avoid duplication
    cursor.execute("""
    SELECT setup_id FROM setups 
    WHERE symbol = ? AND setup_type = ? AND entry_price = ? AND status IN ('PENDING', 'ACTIVE', 'TP1_HIT')
    """, (symbol, setup_type, entry))
    if cursor.fetchone():
        conn.close()
        return None
        
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
    INSERT INTO setups (
        timestamp, symbol, setup_type, entry_order_type, spot_price, entry_price, sl_price, tp_price,
        tp1_price, tp2_price, tp1_hit, atr, rsi, vwap, htf_regime, funding_rate, open_interest, dxy_index, status, conviction_score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
    """, (now_iso, symbol, setup_type, entry_order_type, spot, entry, sl, tp2, tp1, tp2, atr, rsi, vwap, htf_regime, funding_rate, open_interest, dxy_index, conviction_score))
    
    setup_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return setup_id

def update_setup_status(setup_id, status, highest=None, lowest=None, resolution_time=None, sl_price=None, be_price=None, tp1_hit=None):
    conn = get_connection()
    cursor = conn.cursor()
    
    updates = ["status = ?"]
    params = [status]
    
    if highest is not None:
        updates.append("highest_reached = ?")
        params.append(highest)
    if lowest is not None:
        updates.append("lowest_reached = ?")
        params.append(lowest)
    if resolution_time is not None:
        updates.append("resolution_time = ?")
        params.append(resolution_time)
    if sl_price is not None:
        updates.append("sl_price = ?")
        params.append(sl_price)
    if be_price is not None:
        updates.append("be_price = ?")
        params.append(be_price)
    if tp1_hit is not None:
        updates.append("tp1_hit = ?")
        params.append(1 if tp1_hit else 0)
        
    params.append(setup_id)
    query = f"UPDATE setups SET {', '.join(updates)} WHERE setup_id = ?"
    
    cursor.execute(query, tuple(params))
    conn.commit()
    conn.close()

def get_active_setups():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM setups WHERE status IN ('PENDING', 'ACTIVE', 'TP1_HIT') AND symbol IN ('BTC/USDT', 'SILVER/USDT', 'TESTBTC')")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_resolved_setups_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM setups WHERE status IN ('WIN', 'FULL_WIN', 'HALF_WIN_BE', 'LOSS', 'EXPIRED')")
    row = cursor.fetchone()
    conn.close()
    return row["count"]

def get_resolved_setups(limit=100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM setups WHERE status IN ('WIN', 'FULL_WIN', 'HALF_WIN_BE', 'LOSS', 'EXPIRED') ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
