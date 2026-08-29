import requests
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_candles(symbol: str, interval: str = "15m", limit: int = 100):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        res = requests.get(url, timeout=10).json()
        df = pd.DataFrame(res, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

def evaluate_market(symbol: str = "BTCUSDT", asset_type: str = "crypto"):
    df = fetch_candles(symbol, interval="15m", limit=100)
    if df.empty or len(df) < 30:
        return {"status": "error"}

    # Anchor calculations to closed candle to prevent repainting
    closed_df = df.iloc[:-1].copy()
    spot = float(df["close"].iloc[-1])

    # ATR (14) on closed bars
    high_low = closed_df["high"] - closed_df["low"]
    high_cp = (closed_df["high"] - closed_df["close"].shift()).abs()
    low_cp = (closed_df["low"] - closed_df["close"].shift()).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    # RSI (14)
    delta = closed_df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs)).iloc[-1]

    # Session VWAP
    cum_vol = closed_df["volume"].cumsum()
    cum_pv = (closed_df["close"] * closed_df["volume"]).cumsum()
    vwap = (cum_pv / (cum_vol + 1e-9)).iloc[-1]

    # Structural Swing Zones (Fixed 20-bar lookback)
    recent_highs = closed_df["high"].tail(20)
    recent_lows = closed_df["low"].tail(20)

    supply_entry = float(recent_highs.max())
    demand_entry = float(recent_lows.min())

    # Invalidation & Take-Profit Offsets
    supply_sl = supply_entry + (1.2 * atr)
    demand_sl = demand_entry - (1.2 * atr)
    tp1 = vwap

    return {
        "symbol": symbol,
        "spot": spot,
        "supply_entry": round(supply_entry, 1 if "BTC" in symbol else 2),
        "supply_sl": round(supply_sl, 1 if "BTC" in symbol else 2),
        "demand_entry": round(demand_entry, 1 if "BTC" in symbol else 2),
        "demand_sl": round(demand_sl, 1 if "BTC" in symbol else 2),
        "vwap": round(vwap, 1 if "BTC" in symbol else 2),
        "rsi": round(rsi, 1),
        "atr": round(atr, 2),
        "status": "active"
    }
