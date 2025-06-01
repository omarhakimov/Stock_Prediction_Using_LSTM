"""
Integration tests for the complete stock prediction pipeline.
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
from src.model import LSTMModel
from src.predictor import StockPredictor, RiskClassifier
from src.visualizer import StockVisualizer


class TestStockPredictionPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
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
        
        # Set environment variable to force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_complete_pipeline(self):
        """Test the complete prediction pipeline."""
        # 1. Data processing
        processor = StockDataProcessor(window_size=3)
        df, scaler = processor.load_and_preprocess(self.temp_file.name, 'TEST')
        dates, X, y = processor.df_to_windowed_df_simple(df)
        data_splits = processor.split_data(dates, X, y)
        
        # 2. Model training (with reduced parameters for speed)
        model = LSTMModel(window_size=3, lstm_units=8, dense_units=4)
        trained_model = model.train_simple(
            data_splits['X_train'], data_splits['y_train'],
            data_splits['X_val'], data_splits['y_val'],
            epochs=2  # Minimal epochs for testing
        )
        
        # 3. Predictions
        predictor = StockPredictor(trained_model, scaler)
        test_predictions = trained_model.predict(data_splits['X_test'], verbose=0)
        
        # 4. Risk classification
        classifier = RiskClassifier()
        test_predictions_denorm = scaler.inverse_transform(test_predictions).flatten()
        
        daily_classification = classifier.classify_daily(
            data_splits['dates_test'], test_predictions_denorm, 'TEST'
        )
        
        # 5. Future predictions
        last_window = data_splits['X_test'][-1].flatten()
        future_predictions_df = predictor.predict_future(last_window, days=5)
        
        # Assertions
        self.assertIsNotNone(trained_model)
        self.assertEqual(test_predictions.shape[0], len(data_splits['y_test']))
        self.assertIsInstance(daily_classification, pd.DataFrame)
        self.assertEqual(len(future_predictions_df), 5)
        self.assertIn('Classification', daily_classification.columns)
        
        print("✅ Complete pipeline test passed")
    
    def test_pipeline_with_minimal_data(self):
        """Test pipeline with minimal data requirements."""
        # Create minimal dataset
        dates = pd.date_range('2021-01-01', '2021-01-31', freq='D')
        prices = 100 + np.random.randn(len(dates)) * 0.1
        
        minimal_data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Index': 'MINIMAL'
        })
        
        temp_minimal = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        minimal_data.to_csv(temp_minimal.name, index=False)
        temp_minimal.close()
        
        try:
            processor = StockDataProcessor(window_size=3)
            df, scaler = processor.load_and_preprocess(temp_minimal.name, 'MINIMAL')
            
            if len(df) >= 10:  # Minimum required for meaningful splits
                dates, X, y = processor.df_to_windowed_df_simple(df)
                self.assertGreater(len(dates), 0)
                print("✅ Minimal data test passed")
            else:
                print("⚠️ Minimal data test skipped - insufficient data")
        
        finally:
            os.unlink(temp_minimal.name)
    
    def test_error_handling(self):
        """Test error handling in the pipeline."""
        processor = StockDataProcessor()
        
        # Test with non-existent file
        with self.assertRaises(Exception):
            processor.load_and_preprocess('non_existent_file.csv')
        
        # Test with invalid stock symbol
        with self.assertRaises(ValueError):
            processor.load_and_preprocess(self.temp_file.name, 'INVALID')
        
        print("✅ Error handling test passed")


if __name__ == '__main__':
    unittest.main()
