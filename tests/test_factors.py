import unittest
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import factorforge from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factorforge import engineer_factors

class TestFactorEngineering(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        data = {
            'date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=30)),
            'open': np.random.uniform(98, 102, 30),
            'high': np.random.uniform(100, 105, 30),
            'low': np.random.uniform(95, 100, 30),
            'close': np.random.uniform(99, 104, 30),
            'volume': np.random.randint(100000, 500000, 30)
        }
        self.df = pd.DataFrame(data)
        self.df['high'] = self.df['close'] + np.random.uniform(0, 2, 30)
        self.df['low'] = self.df['close'] - np.random.uniform(0, 2, 30)


    def test_factor_columns_exist(self):
        """Test that all engineered factor columns are present in the DataFrame."""
        processed_df = engineer_factors(self.df.copy())
        expected_factors = [
            'momentum_10d', 'sma_gap_20d', 'atr_14d', 'earnings_yield_proxy'
        ]
        for factor in expected_factors:
            with self.subTest(factor=factor):
                self.assertIn(factor, processed_df.columns)

if __name__ == '__main__':
    unittest.main() 