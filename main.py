#!/usr/bin/env python3
"""
Main execution script for stock price prediction.

This script demonstrates how to use the stock prediction package to:
1. Load and preprocess stock data
2. Train LSTM models
3. Make predictions
4. Classify risk levels
5. Generate visualizations

Usage:
    python main.py [--stock SYMBOL] [--mode single|batch] [--future-days DAYS]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import StockDataProcessor
from src.model import LSTMModel
from src.predictor import StockPredictor, RiskClassifier, BatchProcessor
from src.visualizer import StockVisualizer
from src.utils import create_output_directories, log_processing_step, set_random_seeds
import config


def analyze_single_stock(file_path: str, stock_symbol: str = None, future_days: int = 60):
    """
    Analyze a single stock.
    
    Args:
        file_path: Path to the data file
        stock_symbol: Stock symbol to analyze (if None, analyzes first stock)
        future_days: Number of days to predict into the future
    """
    log_processing_step("Starting single stock analysis", stock_symbol or "Default")
    
    # Initialize components
    processor = StockDataProcessor()
    visualizer = StockVisualizer()
    classifier = RiskClassifier()
    
    # Load and preprocess data
    log_processing_step("Loading and preprocessing data")
    df, scaler = processor.load_and_preprocess(file_path, stock_symbol)
    
    # Create windowed dataset
    log_processing_step("Creating windowed dataset")
    dates, X, y = processor.df_to_windowed_df_simple(df)
    
    # Split data
    log_processing_step("Splitting data")
    data_splits = processor.split_data(dates, X, y)
    
    # Visualize data splits
    visualizer.plot_data_splits(
        data_splits['dates_train'], data_splits['y_train'],
        data_splits['dates_val'], data_splits['y_val'],
        data_splits['dates_test'], data_splits['y_test']
    )
    
    # Train model
    log_processing_step("Training LSTM model")
    model = LSTMModel()
    trained_model = model.train_simple(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val'],
        epochs=config.EPOCHS
    )
    
    # Save model
    model_path = os.path.join(config.MODEL_DIR, f"{stock_symbol or 'stock'}_model.keras")
    model.save_model(model_path)
    log_processing_step(f"Model saved to {model_path}")
    
    # Make predictions
    log_processing_step("Making predictions")
    predictor = StockPredictor(trained_model, scaler)
    
    train_predictions = trained_model.predict(data_splits['X_train'], verbose=0).flatten()
    val_predictions = trained_model.predict(data_splits['X_val'], verbose=0).flatten()
    test_predictions = trained_model.predict(data_splits['X_test'], verbose=0).flatten()
    
    # Visualize training results
    visualizer.plot_training_results(
        data_splits['dates_train'], data_splits['y_train'], train_predictions,
        data_splits['dates_val'], data_splits['y_val'], val_predictions,
        data_splits['dates_test'], data_splits['y_test'], test_predictions
    )
    
    # Denormalize test predictions for classification
    test_predictions_denorm = scaler.inverse_transform(
        test_predictions.reshape(-1, 1)
    ).flatten()
    
    # Generate classifications
    log_processing_step("Generating risk classifications")
    daily_classification = classifier.classify_daily(
        data_splits['dates_test'], test_predictions_denorm, stock_symbol or "Stock"
    )
    monthly_classification = classifier.classify_monthly(
        data_splits['dates_test'], test_predictions_denorm, stock_symbol or "Stock"
    )
    
    # Save classifications
    daily_path = os.path.join(config.OUTPUT_DIR, 'classifications', config.DAILY_CLASSIFICATION_FILE)
    monthly_path = os.path.join(config.OUTPUT_DIR, 'classifications', config.MONTHLY_CLASSIFICATION_FILE)
    
    daily_classification.to_csv(daily_path, index=False)
    monthly_classification.to_csv(monthly_path, index=False)
    
    log_processing_step(f"Classifications saved to {daily_path} and {monthly_path}")
    
    # Visualize classifications
    visualizer.plot_daily_classifications(daily_classification)
    visualizer.plot_monthly_classifications(monthly_classification)
    
    # Predict future prices
    log_processing_step(f"Predicting {future_days} days into the future")
    last_window = data_splits['X_test'][-1].flatten()
    future_predictions_df = predictor.predict_future(last_window, days=future_days)
    
    # Classify future predictions
    classified_future_df = classifier.classify_future_predictions(future_predictions_df)
    
    # Save future predictions
    future_path = os.path.join(
        config.OUTPUT_DIR, 'predictions', 
        config.FUTURE_PREDICTIONS_FILE.format(stock=stock_symbol or 'stock')
    )
    future_predictions_df.to_csv(future_path, index=False)
    log_processing_step(f"Future predictions saved to {future_path}")
    
    # Visualize future predictions
    visualizer.plot_future_with_classification(classified_future_df)
    
    print("\n=== Analysis Complete ===")
    print(f"Daily classifications: {len(daily_classification)} records")
    print(f"Monthly classifications: {len(monthly_classification)} records")
    print(f"Future predictions: {len(future_predictions_df)} days")


def analyze_batch_stocks(file_path: str):
    """
    Analyze multiple stocks in batch mode.
    
    Args:
        file_path: Path to the CSV file containing multiple stocks
    """
    log_processing_step("Starting batch stock analysis")
    
    # Initialize batch processor
    processor = BatchProcessor()
    
    # Process all stocks
    log_processing_step("Processing all stocks")
    results = processor.process_all_stocks(file_path)
    
    # Save results
    output_path = os.path.join(config.OUTPUT_DIR, 'predictions', config.BATCH_RESULTS_FILE)
    processor.save_results(results, output_path)
    
    log_processing_step(f"Batch results saved to {output_path}")
    print(f"\n=== Batch Analysis Complete ===")
    print(f"Processed {len(set([r['Stock'] for r in results]))} stocks")
    print(f"Generated {len(results)} predictions")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Stock Price Prediction with LSTM')
    parser.add_argument('--stock', type=str, help='Stock symbol to analyze (for single mode)')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single',
                       help='Analysis mode: single stock or batch processing')
    parser.add_argument('--future-days', type=int, default=60,
                       help='Number of days to predict into the future')
    parser.add_argument('--data-path', type=str, default=config.DATA_PATH,
                       help='Path to the data file')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Create output directories
    create_output_directories()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please place your stock data CSV file in the data/raw/ directory")
        return
    
    try:
        if args.mode == 'single':
            analyze_single_stock(args.data_path, args.stock, args.future_days)
        else:
            analyze_batch_stocks(args.data_path)
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
