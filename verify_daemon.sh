#!/bin/bash
set -e

echo "[*] Checking Docker container trading_bot status..."
docker ps --filter "name=trading_bot" --format "{{.Status}}"

echo "[*] Inspecting recent background daemon logs for exceptions..."
docker logs trading_bot --tail 50 | grep -E "ERROR|CRITICAL|Traceback|sqlite3.OperationalError" || echo "Clean: No exceptions found in recent logs."

echo "[*] Verifying SQLite WAL mode and integrity..."
sqlite3 data/trading.db "PRAGMA journal_mode; PRAGMA integrity_check;"

echo "[*] Running smoke test on brain_scorer and telegram_watcher integration..."
python3 -m unittest discover -s tests/
