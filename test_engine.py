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

    def test_outlier_wick_filtering_and_msi_bounds(self):
        # 1. BTC Outlier High Spike (+12 ATR above consolidation shelf)
        dates = pd.date_range("2026-08-23 08:00", periods=60, freq="15min")
        base = 77000.0
        np.random.seed(42)
        close = base + np.cumsum(np.random.randn(60) * 40)
        df_btc = pd.DataFrame({
            'timestamp': [int(d.timestamp()) for d in dates],
            'datetime': dates,
            'open': close - 15,
            'high': close + 35,
            'low': close - 35,
            'close': close,
            'volume': np.random.uniform(500, 2000, 60)
        })
        freak_high = float(df_btc['high'].max()) + 3000.0 # ~12 ATR freak spike
        df_btc.loc[25, 'high'] = freak_high
        df_btc = calculate_clean_indicators(df_btc)
        spot_btc = float(df_btc['close'].iloc[-1])
        res_btc = get_structural_levels(df_btc, "BTC/USDT", spot_btc)
        
        sp_btc = res_btc['short_plan']
        lp_btc = res_btc['long_plan']
        
        # Invalidation must NOT be hijacked by freak spike
        self.assertLess(sp_btc['sl'], freak_high)
        self.assertLess(sp_btc['msi_anchor'], freak_high)
        # Directional invariants must hold
        self.assertGreater(sp_btc['sl'], sp_btc['entry'])
        self.assertGreaterEqual(sp_btc['entry'], spot_btc)
        self.assertGreaterEqual(spot_btc, lp_btc['entry'])
        self.assertGreater(lp_btc['entry'], lp_btc['sl'])
        
        # 2. Silver Outlier Flash Crash (-15 ATR below consolidation shelf)
        dates_s = pd.date_range("2026-08-23 08:00", periods=60, freq="15min")
        base_s = 29.50
        close_s = base_s + np.cumsum(np.random.randn(60) * 0.05)
        df_silver = pd.DataFrame({
            'timestamp': [int(d.timestamp()) for d in dates_s],
            'datetime': dates_s,
            'open': close_s - 0.02,
            'high': close_s + 0.05,
            'low': close_s - 0.05,
            'close': close_s,
            'volume': np.random.uniform(500, 2000, 60)
        })
        crash_low = float(df_silver['low'].min()) - 3.0 # -15 ATR crash
        df_silver.loc[20, 'low'] = crash_low
        df_silver = calculate_clean_indicators(df_silver)
        spot_silver = float(df_silver['close'].iloc[-1])
        res_silver = get_structural_levels(df_silver, "SILVER/USDT", spot_silver)
        
        sp_silver = res_silver['short_plan']
        lp_silver = res_silver['long_plan']
        
        # Invalidation must NOT be hijacked by crash low
        self.assertGreater(lp_silver['sl'], crash_low)
        self.assertGreater(lp_silver['msi_anchor'], crash_low)
        # Directional invariants must hold with 3 decimals
        self.assertEqual(res_silver['decimals'], 3)
        self.assertGreater(sp_silver['sl'], sp_silver['entry'])
        self.assertGreaterEqual(sp_silver['entry'], spot_silver)
        self.assertGreaterEqual(spot_silver, lp_silver['entry'])
        self.assertGreater(lp_silver['entry'], lp_silver['sl'])
        print("✅ Passed: Statistical Outlier Wick Filtering verified on BTC and Silver (MSI protected).")

    def test_tp2_runner_dynamic_boundaries_and_anti_inflation(self):
        # 1. Verify that freak spike does NOT inflate TP2 to 1:14 R:R
        dates = pd.date_range("2026-08-23 08:00", periods=60, freq="15min")
        base = 77000.0
        np.random.seed(42)
        close = base + np.cumsum(np.random.randn(60) * 40)
        df = pd.DataFrame({
            'timestamp': [int(d.timestamp()) for d in dates],
            'datetime': dates,
            'open': close - 15,
            'high': close + 35,
            'low': close - 35,
            'close': close,
            'volume': np.random.uniform(500, 2000, 60)
        })
        df.loc[25, 'high'] = float(df['high'].max()) + 3500.0 # +14 ATR freak spike
        df = calculate_clean_indicators(df)
        spot = float(df['close'].iloc[-1])
        res = get_structural_levels(df, "BTC/USDT", spot)
        
        lp = res['long_plan']
        long_risk = lp['entry'] - lp['sl']
        long_rr2 = (lp['tp2'] - lp['entry']) / long_risk
        
        # TP2 must be dynamic and realistic (preventing 1:14 R:R distortion)
        self.assertGreaterEqual(long_rr2, 2.0)
        self.assertLess(long_rr2, 8.0)
        self.assertGreater(lp['tp2'], lp['tp1'])
        
        # 2. Macro trending breakout expands runner dynamically without rigid 3x cap
        close_trend = base + np.linspace(0, 2000, 60)
        df_trend = pd.DataFrame({
            'timestamp': [int(d.timestamp()) for d in dates],
            'datetime': dates,
            'open': close_trend - 20,
            'high': close_trend + 50,
            'low': close_trend - 50,
            'close': close_trend,
            'volume': np.random.uniform(500, 2000, 60)
        })
        df_trend = calculate_clean_indicators(df_trend)
        spot_trend = float(df_trend['close'].iloc[-1])
        res_trend = get_structural_levels(df_trend, "BTC/USDT", spot_trend)
        lp_trend = res_trend['long_plan']
        trend_risk = lp_trend['entry'] - lp_trend['sl']
        trend_rr2 = (lp_trend['tp2'] - lp_trend['entry']) / trend_risk
        
        self.assertGreater(trend_rr2, 3.0) # Confirms dynamic expansion beyond rigid 3x ceiling
        self.assertGreater(lp_trend['tp2'], lp_trend['tp1'])
        print("✅ Passed: Dynamic TP2 Runner boundaries prevent inflation & preserve trend expansion.")

if __name__ == '__main__':
    unittest.main()


