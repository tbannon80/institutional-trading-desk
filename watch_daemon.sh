#!/bin/bash
set -e

echo "[*] Initializing autonomous post-deployment daemon exception watcher..."

# Monitor container logs for 60 seconds post-sync for unhandled exceptions
COUNTER=0
while [ $COUNTER -lt 6 ]; do
    if docker logs trading_bot --tail 10 2>&1 | grep -E "ERROR|CRITICAL|Traceback|sqlite3.OperationalError"; then
        echo "[-] EXCEPTION DETECTED: Autonomous patch caused runtime exception. Initiating hard rollback..."
        git reset --hard HEAD~1
        docker restart trading_bot
        echo "[+] Rollback complete. Daemon reverted to previous stable SHA and restarted."
        exit 1
    fi
    sleep 10
    COUNTER=$((COUNTER+1))
done

echo "[+] Post-deployment verification complete: Zero exceptions detected. Autonomous sync locked."
