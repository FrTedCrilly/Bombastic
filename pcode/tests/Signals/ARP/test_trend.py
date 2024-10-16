import unittest
import pandas as pd
from pcode.BT.Signals.ARP.trend import TrendSystem  # Adjust this import statement according to your project structure

class TestTrendSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a simple dataset to use in tests
        dates = pd.date_range('2020-01-01', periods=100)
        data = {
            'Open': range(1, 101) +  range(101, 95) ,
            'High': range(2, 102),
            'Low': range(1, 101),
            'Close': range(5, 105),
        }
        cls.test_data = pd.DataFrame(data, index=dates)

    def test_moving_average(self):
        trend_system = TrendSystem(self.test_data)
        moving_avg = trend_system.moving_average(10)
        # Verify the moving average of the last point
        self.assertAlmostEqual(moving_avg.iloc[-1], sum(range(95, 105)) / 10)

    def test_exponential_moving_average(self):
        trend_system = TrendSystem(self.test_data)
        ema = trend_system.exponential_moving_average(10)
        # Check some properties of EMA (e.g., it should not be NaN and should be a specific value for a known dataset)
        self.assertFalse(ema.isna().any())
        # Add more specific checks based on known values or properties of EMA

    def test_ema_crossover_signals(self):
        trend_system = TrendSystem(self.test_data)
        signals = trend_system.ema_crossover(5, 10)
        # Test some properties of the signals (e.g., signal series should not be empty)
        self.assertFalse(signals['signals'].empty)
        # Add more specific checks based on expected crossover signal behavior

    def test_breakout_signal(self):
        trend_system = TrendSystem(self.test_data)
        signals = trend_system.breakout_signal(20)
        # Test that the signal is correctly identifying breakouts
        # Since this is synthetic data, you would need to adjust your checks accordingly
        # For example, check if the breakout signal at a certain point matches expected value
        self.assertEqual(signals['signal'].iloc[-1], 1)  # Assuming the last point is a breakout

    def test_calculate_adx(self):
        trend_system = TrendSystem(self.test_data)
        adx, plus_di, minus_di = trend_system.calculate_adx(window=14)

        # Check if +DI and -DI are pandas Series
        self.assertTrue(isinstance(plus_di, pd.Series), "+DI should be a pandas Series")
        self.assertTrue(isinstance(minus_di, pd.Series), "-DI should be a pandas Series")

        # Check the length of the +DI and -DI series to ensure they match the input data length
        self.assertEqual(len(plus_di), len(self.test_data), "+DI length should match input data length")
        self.assertEqual(len(minus_di), len(self.test_data), "-DI length should match input data length")

        # Optionally, check for specific values if you have known outcomes or ranges
        # This can depend on the exact formula and expected results based on your input data
        # For example:
        # self.assertTrue((plus_di >= 0).all(), "+DI should be non-negative")
        # self.assertTrue((minus_di >= 0).all(), "-DI should be non-negative")

    def test_apply_hp_filter(self):
        trend_system = TrendSystem(self.test_data)
        trend = trend_system.apply_hp_filter(lamb=1600)
        self.assertTrue(isinstance(trend, pd.Series))
        self.assertEqual(len(trend), len(self.test_data))
        # You can also check for specific values if you have a known outcome

    def get_HP_ADX_signals(self):
        trend_system = TrendSystem(self.test_data)
        signals = trend_system.generate_signals(window=14, lamb=1600)
        self.assertTrue(isinstance(signals, pd.Series))
        self.assertEqual(len(signals), len(self.test_data))
        # Additional checks can be made based on expected signal values

    def test_bbands(self):
        # placeholder for unit test
        self.assertTrue(True)


# This allows the test script to be run directly
if __name__ == '__main__':
    unittest.main()
