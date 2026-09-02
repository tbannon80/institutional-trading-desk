import sqlite3

def audit_pending_setups():
    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, symbol, side, entry_price, sl_price, tp1_target, tp2_target, timestamp FROM setups WHERE status = 'PENDING'")
    rows = cursor.fetchall()
    print(f"[*] Current Active PENDING Setups: {len(rows)}")
    for row in rows:
        print(f"  - ID: {row[0]} | {row[1]} {row[2]} | Entry: {row[3]} | SL: {row[4]} | TP1: {row[5]} | TP2: {row[6]} | Created: {row[7]}")
    conn.close()

if __name__ == "__main__":
    audit_pending_setups()
