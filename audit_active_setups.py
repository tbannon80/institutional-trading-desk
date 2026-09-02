import sqlite3

def audit_database():
    conn = sqlite3.connect('data/trading.db')
    cursor = conn.cursor()
    
    # Check setup distributions and filter out test rows if necessary
    cursor.execute("SELECT status, count(*) FROM setups GROUP BY status;")
    print("[*] Setup Status Distribution:")
    for row in cursor.fetchall():
        print(f"  - Status: {row[0]} | Count: {row[1]}")
        
    # Inspect schema and virtual column mapping via PRAGMA table_xinfo
    cursor.execute("PRAGMA table_xinfo(setups);")
    print("\n[*] Table Schema & Virtual Column Mapping:")
    for col in cursor.fetchall():
        print(f"  - Col {col[0]}: {col[1]} ({col[2]}) [Hidden: {col[5]}]")
        
    conn.close()

if __name__ == "__main__":
    audit_database()
