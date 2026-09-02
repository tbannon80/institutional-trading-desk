import unittest
import os
import sqlite3
import sys

# Ensure repo folder is in sys.path
sys.path.append(os.path.dirname(__file__))

from brain_db import init_db, save_setup, get_active_setups, get_setting, set_setting, get_db_path
from brain_scorer import calculate_conviction_score
from brain_metrics import fetch_binance_funding_and_oi, fetch_fred_dxy

class TestTradingBrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Point to a test database instead of production
        os.environ["TRADING_REPO_PATH"] = os.path.dirname(__file__)
        cls.db_path = get_db_path()
        init_db()

    def test_default_settings(self):
        w_rsi = get_setting("w_rsi")
        self.assertEqual(float(w_rsi), 25.0)
        
        fred_key = get_setting("fred_api_key")
        self.assertEqual(fred_key, "ba32dd734abc8235a0ceb07967ab4812")

    def test_btc_conviction_scoring(self):
        # Test LONG BTC with healthy funding/OI
        score_healthy, reasons_healthy = calculate_conviction_score(
            symbol="BTC/USDT",
            setup_type="LONG",
            entry_price=64000.0,
            vwap=64500.0, # Below VWAP = favorable
            rsi=25.0,     # Oversold = favorable
            fvg_aligned=True,
            funding_rate=0.0001, # 0.01% = favorable
            oi_trend="DOWN"      # OI flush = favorable
        )
        
        # Test LONG BTC with overheated funding/rising OI (dangerous)
        score_dangerous, reasons_dangerous = calculate_conviction_score(
            symbol="BTC/USDT",
            setup_type="LONG",
            entry_price=64000.0,
            vwap=63500.0, # Above VWAP = unfavorable for LONG
            rsi=55.0,     # Neutral RSI = unfavorable for LONG
            fvg_aligned=False,
            funding_rate=0.0006, # 0.06% = dangerous
            oi_trend="UP"        # Rising OI = dangerous
        )
        
        self.assertGreater(score_healthy, score_dangerous)
        self.assertGreaterEqual(score_healthy, 80.0) # Favorable setup should be high conviction
        self.assertLessEqual(score_dangerous, 40.0) # Unfavorable setup should be low conviction

    def test_silver_conviction_scoring(self):
        # Test Silver LONG with DXY going DOWN (favorable)
        score_favorable, _ = calculate_conviction_score(
            symbol="SILVER/USDT",
            setup_type="LONG",
            entry_price=29.0,
            vwap=29.5,
            rsi=28.0,
            fvg_aligned=True,
            dxy_trend="DOWN"
        )
        
        # Test Silver LONG with DXY going UP (headwind)
        score_unfavorable, _ = calculate_conviction_score(
            symbol="SILVER/USDT",
            setup_type="LONG",
            entry_price=29.0,
            vwap=29.5,
            rsi=28.0,
            fvg_aligned=True,
            dxy_trend="UP"
        )
        self.assertGreater(score_favorable, score_unfavorable)

    def test_sqlite_setup_saving_and_tracking(self):
        # Clean existing test entries if any
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM setups WHERE symbol = 'TESTBTC'")
        conn.commit()
        conn.close()
        
        # Log a setup
        setup_id = save_setup(
            symbol="TESTBTC",
            setup_type="LONG",
            spot=64200.0,
            entry=64000.0,
            sl=63500.0,
            tp=65500.0,
            atr=500.0,
            rsi=30.0,
            vwap=64500.0,
            conviction_score=85.0
        )
        
        self.assertIsNotNone(setup_id)
        
        # Retrieve active setups
        active = get_active_setups()
        test_setups = [s for s in active if s["symbol"] == "TESTBTC"]
        self.assertEqual(len(test_setups), 1)
        self.assertEqual(test_setups[0]["status"], "PENDING")

if __name__ == "__main__":
    unittest.main()
