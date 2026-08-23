"""
test_engine.py - Automated Assertion Runner
"""
import unittest
import pandas as pd
import numpy as np
from smc_engine import calculate_clean_indicators, get_structural_levels

class TestSMCEngine(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range("2026-08-23 08:00", periods=60, freq="5min")
        base = 77246.0
        np.random.seed(42)
        close = base + np.cumsum(np.random.randn(60) * 40)
        self.df = pd.DataFrame({
            'datetime': dates,
            'open': close - 15,
            'high': close + 35,
            'low': close - 35,
            'close': close,
            'volume': np.random.uniform(500, 2000, 60)
        })
        self.df = calculate_clean_indicators(self.df)

    def test_directional_sanity_bounds(self):
        spot = float(self.df['close'].iloc[-1])
        res = get_structural_levels(self.df, "BTC/USDT", spot)
        
        sp = res['short_plan']
        lp = res['long_plan']
        
        self.assertGreater(sp['sl'], sp['entry'])
        self.assertGreaterEqual(sp['entry'], spot)
        self.assertGreaterEqual(spot, lp['entry'])
        self.assertGreater(lp['entry'], lp['sl'])
        print(f"\n✅ Passed: Short SL ({sp['sl']}) > Short Entry ({sp['entry']}) > Spot ({spot:.1f}) > Long Entry ({lp['entry']}) > Long SL ({lp['sl']})")

if __name__ == '__main__':
    unittest.main()
