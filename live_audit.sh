#!/bin/bash
set -e

echo "[*] Verifying active trading_bot container status and resource health..."
docker stats trading_bot --no-stream

echo "[*] Checking last 20 log entries for daemon loop performance..."
docker logs trading_bot --tail 20

echo "[*] Verifying SQLite database journaling and concurrent access locks..."
sqlite3 data/trading.db "PRAGMA journal_mode; SELECT count(*) FROM setups;"
