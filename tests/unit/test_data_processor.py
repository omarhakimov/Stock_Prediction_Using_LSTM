"""
Unit tests for the data processor module.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.data_processor import StockDataProcessor


class TestStockDataProcessor(unittest.TestCase):
    """Test cases for StockDataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = StockDataProcessor(window_size=3)
        
        # Create sample data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Index': 'TEST'
        })
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = StockDataProcessor(window_size=5)
        self.assertEqual(processor.window_size, 5)
        self.assertIsNone(processor.scaler)
    
    def test_str_to_datetime(self):
        """Test string to datetime conversion."""
        date_str = '2020-01-15'
        result = self.processor.str_to_datetime(date_str)
        expected = pd.to_datetime(date_str).to_pydatetime()
        self.assertEqual(result, expected)
    
    def test_load_and_preprocess(self):
        """Test data loading and preprocessing."""
        df, scaler = self.processor.load_and_preprocess(self.temp_file.name, 'TEST')
        
        # Check dataframe structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Close', df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))
        
        # Check normalization
        self.assertTrue(df['Close'].min() >= 0)
        self.assertTrue(df['Close'].max() <= 1)
        
        # Check scaler
        self.assertIsNotNone(scaler)
    
    def test_preprocess_stock_data(self):
        """Test alternative preprocessing method."""
        df_copy = self.sample_data.copy()
        df, scaler = self.processor.preprocess_stock_data(df_copy)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Close', df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))
        self.assertIsNotNone(scaler)
    
    def test_windowed_df_simple(self):
        """Test simple windowing function."""
        df, scaler = self.processor.preprocess_stock_data(self.sample_data.copy())
        dates, X, y = self.processor.df_to_windowed_df_simple(df)
        
        # Check shapes
        self.assertEqual(len(dates), len(X))
        self.assertEqual(len(dates), len(y))
        self.assertEqual(X.shape[1], self.processor.window_size)
        self.assertEqual(X.shape[2], 1)
        
        # Check data types
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)
    
    def test_split_data(self):
        """Test data splitting functionality."""
        df, scaler = self.processor.preprocess_stock_data(self.sample_data.copy())
        dates, X, y = self.processor.df_to_windowed_df_simple(df)
        
        splits = self.processor.split_data(dates, X, y)
        
        # Check all splits exist
        required_keys = [
            'dates_train', 'X_train', 'y_train',
            'dates_val', 'X_val', 'y_val',
            'dates_test', 'X_test', 'y_test'
        ]
        
        for key in required_keys:
            self.assertIn(key, splits)
        
        # Check split sizes
        total_samples = len(dates)
        train_samples = splits['X_train'].shape[0]
        val_samples = splits['X_val'].shape[0]
        test_samples = splits['X_test'].shape[0]
        
        self.assertEqual(train_samples + val_samples + test_samples, total_samples)
        self.assertGreater(train_samples, val_samples)
        self.assertGreater(train_samples, test_samples)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame(columns=['Date', 'Close', 'Index'])
        
        # Create temporary file with empty data
        temp_empty = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        empty_data.to_csv(temp_empty.name, index=False)
        temp_empty.close()
        
        try:
            with self.assertRaises(Exception):
                self.processor.load_and_preprocess(temp_empty.name, 'TEST')
        finally:
            os.unlink(temp_empty.name)
    
    def test_invalid_stock_symbol(self):
        """Test handling of invalid stock symbol."""
        with self.assertRaises(ValueError):
            self.processor.load_and_preprocess(self.temp_file.name, 'INVALID')


if __name__ == '__main__':
    unittest.main()
