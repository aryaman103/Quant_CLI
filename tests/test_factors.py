import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile

# Adjust path to import factorforge from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factorforge import engineer_factors, load_data, calculate_metrics, run_backtest

class TestFactorEngineering(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        np.random.seed(42)  # For reproducible tests
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

    def test_factor_data_types(self):
        """Test that engineered factors have correct data types."""
        processed_df = engineer_factors(self.df.copy())
        factors = ['momentum_10d', 'sma_gap_20d', 'atr_14d', 'earnings_yield_proxy']
        for factor in factors:
            with self.subTest(factor=factor):
                self.assertTrue(pd.api.types.is_numeric_dtype(processed_df[factor]))

    def test_load_data_file_not_found(self):
        """Test load_data raises FileNotFoundError for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')

    def test_load_data_missing_columns(self):
        """Test load_data raises ValueError for missing required columns."""
        # Create a temporary CSV file with missing columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('date,close\n2022-01-01,100\n')
            temp_file = f.name
        
        try:
            with self.assertRaises(ValueError):
                load_data(temp_file)
        finally:
            os.unlink(temp_file)

    def test_calculate_metrics(self):
        """Test performance metrics calculation."""
        # Create a simple equity curve
        dates = pd.date_range('2022-01-01', periods=10)
        equity_curve = pd.Series([1.0, 1.01, 1.02, 0.98, 1.05, 1.03, 1.08, 1.06, 1.10, 1.12], 
                                index=dates)
        
        cagr, sharpe, max_dd = calculate_metrics(equity_curve)
        
        self.assertIsInstance(cagr, float)
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0)  # Max drawdown should be negative or zero

    def test_run_backtest(self):
        """Test backtest function runs without errors."""
        # Create test data with proper date index
        test_df = self.df.copy()
        test_df.index = pd.to_datetime(test_df['date'])
        test_df['target'] = np.random.normal(0, 0.01, len(test_df))
        
        predictions = np.random.normal(0, 0.01, len(test_df))
        
        result = run_backtest(test_df, predictions)
        
        self.assertIn('equity_curve', result.columns)
        self.assertIn('strategy_return', result.columns)
        self.assertIn('signal', result.columns)

if __name__ == '__main__':
    unittest.main() 