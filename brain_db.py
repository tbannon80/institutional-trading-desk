import os
import sqlite3
from datetime import datetime, timezone

DB_NAME = "trading_memory.db"

def get_db_path():
    # Use repo directory if running locally or via mount
    repo_path = os.getenv("TRADING_REPO_PATH", os.path.dirname(__file__))
    return os.path.join(repo_path, DB_NAME)

def get_connection():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Create setups table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS setups (
        setup_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        setup_type TEXT NOT NULL,        -- 'LONG' or 'SHORT'
        spot_price REAL NOT NULL,
        entry_price REAL NOT NULL,
        sl_price REAL NOT NULL,
        tp_price REAL NOT NULL,
        atr REAL NOT NULL,
        rsi REAL NOT NULL,
        vwap REAL NOT NULL,
        funding_rate REAL,
        open_interest REAL,
        dxy_index REAL,
        status TEXT NOT NULL,            -- 'PENDING', 'ACTIVE', 'WIN', 'LOSS', 'EXPIRED'
        conviction_score REAL NOT NULL,
        resolution_time TEXT,
        highest_reached REAL,
        lowest_reached REAL,
        trigger_msg_id INTEGER
    )
    """)
    
    # 2. Create settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS brain_settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)
    
    conn.commit()
    
    # 3. Initialize default weights and FRED API key if they don't exist
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
    print(f"Database initialized successfully at {get_db_path()}")

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

def save_setup(symbol, setup_type, spot, entry, sl, tp, atr, rsi, vwap, funding_rate=None, open_interest=None, dxy_index=None, conviction_score=0.0):
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if a pending or active setup with similar entry already exists to avoid duplication
    cursor.execute("""
    SELECT setup_id FROM setups 
    WHERE symbol = ? AND setup_type = ? AND entry_price = ? AND status IN ('PENDING', 'ACTIVE')
    """, (symbol, setup_type, entry))
    if cursor.fetchone():
        conn.close()
        return None # Prevent duplicate logging
        
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
    INSERT INTO setups (
        timestamp, symbol, setup_type, spot_price, entry_price, sl_price, tp_price,
        atr, rsi, vwap, funding_rate, open_interest, dxy_index, status, conviction_score
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
    """, (now_iso, symbol, setup_type, spot, entry, sl, tp, atr, rsi, vwap, funding_rate, open_interest, dxy_index, conviction_score))
    
    setup_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return setup_id

def update_setup_status(setup_id, status, highest=None, lowest=None, resolution_time=None):
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
        
    params.append(setup_id)
    query = f"UPDATE setups SET {', '.join(updates)} WHERE setup_id = ?"
    
    cursor.execute(query, tuple(params))
    conn.commit()
    conn.close()

def get_active_setups():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM setups WHERE status IN ('PENDING', 'ACTIVE')")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_resolved_setups_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM setups WHERE status IN ('WIN', 'LOSS', 'EXPIRED')")
    row = cursor.fetchone()
    conn.close()
    return row["count"]

def get_resolved_setups(limit=100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM setups WHERE status IN ('WIN', 'LOSS', 'EXPIRED') ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
