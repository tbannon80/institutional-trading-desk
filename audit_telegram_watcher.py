import sqlite3
import datetime

def verify_watcher_state():
    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, symbol, side, entry_price, status FROM setups WHERE status = 'PENDING'")
    pending_rows = cursor.fetchall()
    print(f"[*] Verifying Telegram watcher proximity logic for {len(pending_rows)} pending setups...")
    for row in pending_rows:
        print(f"  - Setup ID {row[0]}: {row[1]} {row[2]} at Entry ${row[3]} -> Status: {row[4]}")
    conn.close()

if __name__ == "__main__":
    verify_watcher_state()
