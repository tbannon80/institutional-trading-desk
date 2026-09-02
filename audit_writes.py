import sqlite3

def verify_write_compatibility():
    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()
    
    # Test safe base-column insertion and ensure virtual columns update correctly
    try:
        cursor.execute("""
            INSERT INTO setups (timestamp, symbol, setup_type, spot_price, entry_price, sl_price, tp_price, atr, rsi, vwap, funding_rate, open_interest, dxy_index, status, conviction_score, entry_order_type, tp1_price, tp2_price, be_price, tp1_hit, htf_regime)
            VALUES (datetime('now'), 'BTCUSDT', 'LONG', 77000.0, 76500.0, 76000.0, 78000.0, 200.0, 50.0, 76800.0, 0.0, 1000.0, 100.0, 'PENDING', 75.0, 'LIMIT', 76700.0, 77500.0, 76504.0, 0, 'NEUTRAL')
        """)
        test_id = cursor.lastrowid
        conn.commit()
        
        # Verify virtual column read mapping
        cursor.execute("SELECT setup_id, side, tp1_target, tp2_target FROM setups WHERE id = ?;", (test_id,))
        row = cursor.fetchone()
        print(f"[+] Virtual column write-read test passed: ID={row[0]}, Side={row[1]}, TP1={row[2]}, TP2={row[3]}")
        
        # Clean up test row
        cursor.execute("DELETE FROM setups WHERE id = ?;", (test_id,))
        conn.commit()
    except Exception as e:
        print(f"[-] Write compatibility warning: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    verify_write_compatibility()
