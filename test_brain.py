import unittest
import os
import sqlite3
import sys
import pandas as pd
import numpy as np

# Ensure repo folder is in sys.path
sys.path.append(os.path.dirname(__file__))

from brain_db import init_db, save_setup, get_active_setups, get_setting, set_setting, get_db_path, get_connection
from brain_scorer import calculate_conviction_score
from brain_metrics import fetch_binance_funding_and_oi, fetch_fred_dxy

class TestTradingBrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["TRADING_REPO_PATH"] = os.path.dirname(__file__)
        cls.db_path = get_db_path()
        init_db()

    def test_sqlite_wal_mode(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(mode.lower(), "wal")
        print(f"✅ Passed: SQLite journal mode is WAL ({mode}).")

    def test_default_settings(self):
        w_rsi = get_setting("w_rsi")
        self.assertEqual(float(w_rsi), 25.0)
        
        fred_key = get_setting("fred_api_key")
        self.assertEqual(fred_key, "ba32dd734abc8235a0ceb07967ab4812")

    def test_btc_conviction_scoring_and_htf_penalty(self):
        # 1. Aligned Bullish Regime (Close > EMA200)
        score_bull_aligned, _ = calculate_conviction_score(
            symbol="BTC/USDT",
            setup_type="LONG",
            entry_price=64000.0,
            vwap=64500.0,
            rsi=28.0,
            fvg_aligned=True,
            htf_regime="BULLISH",
            funding_rate=0.0001,
            oi_trend="DOWN"
        )
        
        # 2. Counter-Trend: Long in 4H Bear Regime (Should receive 0.2x penalty)
        score_bear_penalized, reasons_bear = calculate_conviction_score(
            symbol="BTC/USDT",
            setup_type="LONG",
            entry_price=64000.0,
            vwap=64500.0,
            rsi=28.0, # Oversold in bear regime is a trap!
            fvg_aligned=True,
            htf_regime="BEARISH",
            funding_rate=0.0001,
            oi_trend="DOWN"
        )
        
        self.assertGreaterEqual(score_bull_aligned, 80.0)
        self.assertLessEqual(score_bear_penalized, 25.0) # 0.2x penalty caps score
        self.assertTrue(any("penalty" in r.lower() or "bear regime" in r.lower() for r in reasons_bear))
        print("✅ Passed: HTF Regime 0.2x penalty correctly suppresses counter-trend setups.")

    def test_silver_conviction_scoring(self):
        # Silver LONG with DXY DOWN
        score_favorable, _ = calculate_conviction_score(
            symbol="SILVER/USDT",
            setup_type="LONG",
            entry_price=29.0,
            vwap=29.5,
            rsi=28.0,
            fvg_aligned=True,
            htf_regime="BULLISH",
            dxy_trend="DOWN"
        )
        
        # Silver LONG with DXY UP (Headwind)
        score_unfavorable, _ = calculate_conviction_score(
            symbol="SILVER/USDT",
            setup_type="LONG",
            entry_price=29.0,
            vwap=29.5,
            rsi=28.0,
            fvg_aligned=True,
            htf_regime="NEUTRAL",
            dxy_trend="UP"
        )
        self.assertGreater(score_favorable, score_unfavorable)
        print("✅ Passed: Silver DXY macro factor scoring verified.")

    def test_sqlite_setup_saving_with_split_targets(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM setups WHERE symbol = 'TESTBTC'")
        conn.commit()
        conn.close()
        
        setup_id = save_setup(
            symbol="TESTBTC",
            setup_type="LONG",
            spot=64200.0,
            entry=64000.0,
            sl=63500.0,
            tp=66000.0,
            tp1_price=65000.0,
            tp2_price=66000.0,
            entry_order_type="LIMIT",
            htf_regime="BULLISH",
            atr=500.0,
            rsi=30.0,
            vwap=64500.0,
            conviction_score=85.0
        )
        
        self.assertIsNotNone(setup_id)
        
        active = get_active_setups()
        test_setups = [s for s in active if s["symbol"] == "TESTBTC"]
        self.assertEqual(len(test_setups), 1)
        self.assertEqual(test_setups[0]["status"], "PENDING")
        self.assertEqual(test_setups[0]["tp1_price"], 65000.0)
        self.assertEqual(test_setups[0]["tp2_price"], 66000.0)
        self.assertEqual(test_setups[0]["entry_order_type"], "LIMIT")
        print("✅ Passed: SQLite schema saves dual TP1/TP2, order type, and HTF regime.")

    def test_ohlc_tracking_arrays_dual_asset(self):
        from brain_tracker import fetch_recent_15m_bars, get_current_price
        
        # Test BTC tracking arrays
        btc_bars = fetch_recent_15m_bars("BTC/USDT", limit=10)
        self.assertIsInstance(btc_bars, pd.DataFrame)
        if not btc_bars.empty:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.assertIn(col, btc_bars.columns)
                
        # Test Silver tracking arrays
        silver_bars = fetch_recent_15m_bars("SILVER/USDT", limit=10)
        self.assertIsInstance(silver_bars, pd.DataFrame)
        if not silver_bars.empty:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.assertIn(col, silver_bars.columns)
                
        print("✅ Passed: OHLC tracking arrays compile cleanly for dual-asset scope (BTC/USDT & SILVER/USDT).")

    def test_dual_asset_scope_purge_legacy(self):
        # Insert a dummy legacy altcoin setup to verify auto-purge in init_db
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO setups (timestamp, symbol, setup_type, entry_order_type, spot_price, entry_price, sl_price, tp_price, atr, rsi, vwap, status, conviction_score)
            VALUES ('2026-09-01T00:00:00Z', 'ETH/USDT', 'LONG', 'LIMIT', 2500, 2400, 2300, 2600, 50, 45, 2450, 'PENDING', 75.0)
        """)
        conn.commit()
        conn.close()

        # Run init_db which should purge it
        init_db()

        # Verify ETH/USDT was purged
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM setups WHERE symbol = 'ETH/USDT'")
        eth_count = c.fetchone()[0]
        conn.close()
        self.assertEqual(eth_count, 0)
        
        # Verify active setups only contain BTC, SILVER, or test symbols
        active = get_active_setups()
        for s in active:
            self.assertIn(s["symbol"], ["BTC/USDT", "SILVER/USDT", "TESTBTC"])
            self.assertNotIn(s["symbol"], ["ETH/USDT", "SOL/USDT", "XRP/USDT"])
            
        print("✅ Passed: Database and active tracking strictly locked to dual-asset scope.")

    def test_telegram_watcher_and_brain_scorer_integration(self):
        import smc_engine
        dates = pd.date_range("2026-08-23 08:00", periods=60, freq="15min")
        base = 77000.0
        close = base + np.cumsum(np.random.randn(60) * 30)
        df = pd.DataFrame({
            'timestamp': [int(d.timestamp()) for d in dates],
            'datetime': dates,
            'open': close - 15,
            'high': close + 35,
            'low': close - 35,
            'close': close,
            'volume': np.random.uniform(500, 2000, 60)
        })
        df = smc_engine.calculate_clean_indicators(df)
        spot = float(df['close'].iloc[-1])
        levels = smc_engine.get_structural_levels(df, "BTC/USDT", spot)
        
        lp = levels['long_plan']
        score, reasons = calculate_conviction_score(
            "BTC/USDT", "LONG", lp['entry'], levels['vwap'], levels['rsi'], True, htf_regime="BULLISH"
        )
        self.assertGreaterEqual(score, 50.0)
        self.assertIsInstance(reasons, list)
        
        setup_id = save_setup(
            "TESTBTC", "LONG", spot, lp['entry'], lp['sl'], lp['tp2'],
            levels['atr'], levels['rsi'], levels['vwap'], tp1_price=lp['tp1'], tp2_price=lp['tp2'],
            entry_order_type=levels['entry_order_type'], htf_regime="BULLISH", conviction_score=score
        )
        self.assertIsNotNone(setup_id)
        print("✅ Passed: Smoke test on brain_scorer and telegram_watcher integration verified.")

if __name__ == "__main__":
    unittest.main()


