import sqlite3

def check_setup_states():
    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, symbol, side, entry_price, tp1_target, tp2_target, status FROM setups WHERE status != 'CLOSED'")
    rows = cursor.fetchall()
    print(f"[*] Active Setups Tracking Count: {len(rows)}")
    for row in rows:
        print(f"ID: {row[0]} | {row[1]} {row[2]} | Entry: {row[3]} | TP1: {row[4]} | TP2: {row[5]} | Status: {row[6]}")
    conn.close()

if __name__ == "__main__":
    check_setup_states()
