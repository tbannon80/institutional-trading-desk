import sqlite3
import subprocess

def evaluate_local_health():
    print("[*] Evaluating local mini-HP environment health and feedback loops...")
    
    # 1. Check SQLite WAL integrity
    try:
        conn = sqlite3.connect('data/trading.db')
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()[0]
        print(f"[+] SQLite Database Integrity: {result}")
        conn.close()
    except Exception as e:
        print(f"[-] DATABASE ERROR: Integrity check failed: {e}")
        
    # 2. Check container health metrics
    try:
        status = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Status}}', 'trading_bot']).decode('ascii').strip()
        print(f"[+] Docker container 'trading_bot' status: {status}")
    except Exception as e:
        print(f"[-] CONTAINER ERROR: Unable to inspect trading_bot: {e}")

if __name__ == "__main__":
    evaluate_local_health()
