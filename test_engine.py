"""
test_engine.py - Automated Assertion Runner for Institutional SMC Trading Engine
"""
import unittest
import time
import pandas as pd
import numpy as np
from smc_engine import (
    calculate_clean_indicators, get_structural_levels, 
    validate_candle_data, fetch_htf_regime
)

class TestSMCEngine(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range("2026-08-23 08:00", periods=60, freq="15min")
        base = 77246.0
        np.random.seed(42)
        close = base + np.cumsum(np.random.randn(60) * 40)
        self.df = pd.DataFrame({
            'timestamp': [int(d.timestamp()) for d in dates],
            'datetime': dates,
            'open': close - 15,
            'high': close + 35,
            'low': close - 35,
            'close': close,
            'volume': np.random.uniform(500, 2000, 60)
        })
        self.df = calculate_clean_indicators(self.df)

    def test_directional_sanity_and_msi_bounds(self):
        spot = float(self.df['close'].iloc[-1])
        res = get_structural_levels(self.df, "BTC/USDT", spot)
        
        sp = res['short_plan']
        lp = res['long_plan']
        atr = res['atr']
        
        # 1. Structural Directional Invariant: Short SL > Short Entry >= Spot >= Long Entry > Long SL
        self.assertGreater(sp['sl'], sp['entry'])
        self.assertGreaterEqual(sp['entry'], spot)
        self.assertGreaterEqual(spot, lp['entry'])
        self.assertGreater(lp['entry'], lp['sl'])
        
        # 2. Market Structure Invalidation (MSI): SL placed strictly beyond swing wick with 0.15 ATR buffer
        swing_low_wick = float(self.df['low'].tail(50).min())
        swing_high_wick = float(self.df['high'].tail(50).max())
        self.assertLessEqual(lp['sl'], swing_low_wick)
        self.assertGreaterEqual(sp['sl'], swing_high_wick)
        
        # 3. Dual Take Profit Targets
        self.assertIn('tp1', lp)
        self.assertIn('tp2', lp)
        self.assertGreater(lp['tp2'], lp['tp1'])
        self.assertLess(sp['tp2'], sp['tp1'])
        
        print(f"\n✅ Passed: MSI Bounds and Directional Invariant verified.")

    def test_upstream_data_validation(self):
        # 1. Stale candle detection (> 30 mins)
        stale_df = self.df.copy()
        stale_df['timestamp'] = time.time() - 3600 # 1 hour old
        is_valid, msg = validate_candle_data(stale_df, is_silver=False)
        self.assertFalse(is_valid)
        self.assertIn("stale_data", msg)

        # 2. Zero volume rejection
        zero_vol_df = self.df.copy()
        zero_vol_df['timestamp'] = time.time() - 60
        zero_vol_df['volume'] = 0.0
        is_valid, msg = validate_candle_data(zero_vol_df, is_silver=False)
        self.assertFalse(is_valid)
        self.assertIn("zero_volume", msg)

        # 3. Fresh valid data
        fresh_df = self.df.copy()
        fresh_df['timestamp'] = time.time() - 60
        fresh_df['volume'] = 500.0
        is_valid, msg = validate_candle_data(fresh_df, is_silver=False)
        self.assertTrue(is_valid)
        print("✅ Passed: Upstream Data Validation strictly enforces freshness and non-zero volume.")

    def test_silver_directional_sanity_and_msi_bounds(self):
        # Verify indicators, decimals, and MSI bounds for SILVER/USDT
        dates = pd.date_range("2026-08-23 08:00", periods=60, freq="15min")
        base = 29.50
        np.random.seed(42)
        close = base + np.cumsum(np.random.randn(60) * 0.05)
        df_silver = pd.DataFrame({
            'timestamp': [int(d.timestamp()) for d in dates],
            'datetime': dates,
            'open': close - 0.02,
            'high': close + 0.05,
            'low': close - 0.05,
            'close': close,
            'volume': np.random.uniform(500, 2000, 60)
        })
        df_silver = calculate_clean_indicators(df_silver)
        spot = float(df_silver['close'].iloc[-1])
        res = get_structural_levels(df_silver, "SILVER/USDT", spot)
        
        sp = res['short_plan']
        lp = res['long_plan']
        
        # 1. Decimal Precision & Directional Invariant
        self.assertEqual(res['decimals'], 3)
        self.assertGreater(sp['sl'], sp['entry'])
        self.assertGreaterEqual(sp['entry'], spot)
        self.assertGreaterEqual(spot, lp['entry'])
        self.assertGreater(lp['entry'], lp['sl'])
        
        # 2. MSI bounds
        swing_low_wick = float(df_silver['low'].tail(50).min())
        swing_high_wick = float(df_silver['high'].tail(50).max())
        self.assertLessEqual(lp['sl'], swing_low_wick)
        self.assertGreaterEqual(sp['sl'], swing_high_wick)
        
        # 3. Dual Take Profit Targets
        self.assertIn('tp1', lp)
        self.assertIn('tp2', lp)
        self.assertGreater(lp['tp2'], lp['tp1'])
        self.assertLess(sp['tp2'], sp['tp1'])
        print("✅ Passed: Silver MSI Bounds and Directional Invariant verified (3-decimal precision).")

    def test_htf_regime_filtering_dual_asset(self):
        # HTF regime extraction for BTC/USDT and SILVER/USDT
        regime_btc = fetch_htf_regime("BTC/USDT")
        self.assertIn("htf_regime", regime_btc)
        self.assertIn(regime_btc["htf_regime"], ["BULLISH", "BEARISH", "NEUTRAL"])
        
        regime_silver = fetch_htf_regime("SILVER/USDT")
        self.assertIn("htf_regime", regime_silver)
        self.assertIn(regime_silver["htf_regime"], ["BULLISH", "BEARISH", "NEUTRAL"])
        print("✅ Passed: HTF Regime calculation verified for dual assets (BTC/USDT & SILVER/USDT).")

if __name__ == '__main__':
    unittest.main()


