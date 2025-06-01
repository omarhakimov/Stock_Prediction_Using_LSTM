"""
Prediction and Classification Module

Handles stock price predictions and risk classification.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Optional
import config


class StockPredictor:
    """Handles stock price predictions and future forecasting."""
    
    def __init__(self, model, scaler: MinMaxScaler):
        """
        Initialize the predictor.
        
        Args:
            model: Trained LSTM model
            scaler: Fitted MinMaxScaler for denormalization
        """
        self.model = model
        self.scaler = scaler
    
    def predict_future(self, last_window: np.ndarray, days: int = config.FUTURE_DAYS) -> pd.DataFrame:
        """
        Predict future stock prices recursively.
        
        Args:
            last_window: Last sequence of normalized data
            days: Number of days to predict
            
        Returns:
            DataFrame with future dates and predicted prices
        """
        predictions = []
        current_window = last_window.copy()
        
        # Generate business days for predictions
        last_date = pd.Timestamp.now().date()
        future_dates = pd.date_range(start=last_date, periods=days + 1, freq="B")[1:]
        
        for _ in range(days):
            # Predict next value
            if current_window.ndim == 1:
                current_window = current_window.reshape(1, -1, 1)
            
            next_prediction = self.model.predict(current_window, verbose=0)[0, 0]
            predictions.append(next_prediction)
            
            # Update window
            current_window = current_window.flatten()
            current_window[:-1] = current_window[1:]
            current_window[-1] = next_prediction
            current_window = current_window.reshape(1, -1, 1)
        
        # Denormalize predictions
        predictions_denorm = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return pd.DataFrame({
            'Date': future_dates[:len(predictions)],
            'Predicted Prices': predictions_denorm
        })
    
    def predict_future_steps(self, initial_window: np.ndarray, steps: int = 1370) -> List[float]:
        """
        Predict future steps (notebook implementation).
        
        Args:
            initial_window: Initial window for prediction
            steps: Number of steps to predict
            
        Returns:
            List of predictions
        """
        predictions = []
        current_window = initial_window.copy()
        
        for _ in range(steps):
            next_prediction = self.model.predict(
                current_window[np.newaxis, :, :], verbose=0
            )[0, 0]
            predictions.append(next_prediction)
            current_window = np.roll(current_window, -1)
            current_window[-1] = next_prediction
            
        return predictions


class RiskClassifier:
    """Classifies stocks based on volatility and risk metrics."""
    
    def __init__(self, conservative_threshold: float = config.CONSERVATIVE_THRESHOLD,
                 risky_threshold: float = config.RISKY_THRESHOLD):
        """
        Initialize the risk classifier.
        
        Args:
            conservative_threshold: Volatility threshold for conservative classification
            risky_threshold: Volatility threshold for risky classification
        """
        self.conservative_threshold = conservative_threshold
        self.risky_threshold = risky_threshold
    
    def classify_stock(self, std_dev: float, mean: float) -> str:
        """
        Classify a stock based on volatility.
        
        Args:
            std_dev: Standard deviation of prices
            mean: Mean of prices
            
        Returns:
            Classification string
        """
        if mean == 0 or np.isnan(std_dev) or np.isnan(mean):
            return "Unknown"
        
        volatility = (std_dev / mean) * 100
        
        if volatility > self.risky_threshold:
            return "Risky"
        elif self.conservative_threshold <= volatility <= self.risky_threshold:
            return "Moderate"
        else:
            return "Conservative"
    
    def classify_daily(self, dates: np.ndarray, predictions: np.ndarray, 
                      stock_name: str = "Stock", window_size: int = 7) -> pd.DataFrame:
        """
        Perform daily classification of stock predictions.
        
        Args:
            dates: Array of dates
            predictions: Array of predicted prices
            stock_name: Name of the stock
            window_size: Rolling window size for statistics
            
        Returns:
            DataFrame with daily classifications
        """
        df_daily = pd.DataFrame({
            "Stock Name": stock_name,
            "Dates": dates,
            "Predicted Prices": predictions
        })
        
        df_daily["Mean"] = df_daily["Predicted Prices"].rolling(window=window_size).mean()
        df_daily["Std Deviation"] = df_daily["Predicted Prices"].rolling(window=window_size).std()
        
        df_daily["Classification"] = df_daily.apply(
            lambda row: self.classify_stock(row["Std Deviation"], row["Mean"]), 
            axis=1
        )
        
        # Remove rows with insufficient rolling data
        df_daily.dropna(inplace=True)
        
        return df_daily
    
    def classify_monthly(self, dates: np.ndarray, predictions: np.ndarray, 
                        stock_name: str = "Stock") -> pd.DataFrame:
        """
        Perform monthly classification of stock predictions.
        
        Args:
            dates: Array of dates
            predictions: Array of predicted prices
            stock_name: Name of the stock
            
        Returns:
            DataFrame with monthly classifications
        """
        df_monthly = pd.DataFrame({
            "Dates": dates,
            "Predicted Prices": predictions
        })
        
        df_monthly["Dates"] = pd.to_datetime(df_monthly["Dates"])
        df_monthly.set_index("Dates", inplace=True)
        
        # Resample to monthly aggregations
        df_monthly = df_monthly.resample("M").agg(["mean", "std"])["Predicted Prices"]
        df_monthly.reset_index(inplace=True)
        df_monthly.rename(columns={"mean": "Mean", "std": "Std Deviation"}, inplace=True)
        
        # Add stock name as first column
        df_monthly.insert(0, "Stock Name", stock_name)
        
        df_monthly["Classification"] = df_monthly.apply(
            lambda row: self.classify_stock(row["Std Deviation"], row["Mean"]), 
            axis=1
        )
        
        return df_monthly
    
    def classify_future_predictions(self, future_df: pd.DataFrame, 
                                  window_size: int = 3) -> pd.DataFrame:
        """
        Classify future predicted stock prices.
        
        Args:
            future_df: DataFrame with future predictions
            window_size: Rolling window size
            
        Returns:
            DataFrame with classifications
        """
        future_df = future_df.copy()
        
        # Calculate rolling statistics
        future_df["Mean"] = future_df["Predicted Prices"].rolling(window=window_size).mean()
        future_df["Std Deviation"] = future_df["Predicted Prices"].rolling(window=window_size).std()
        
        # Classify based on volatility
        future_df["Classification"] = future_df.apply(
            lambda row: self.classify_stock(row["Std Deviation"], row["Mean"]), 
            axis=1
        )
        
        return future_df


class BatchProcessor:
    """Processes multiple stocks in batch mode."""
    
    def __init__(self):
        """Initialize the batch processor."""
        self.results = []
    
    def process_all_stocks(self, file_path: str) -> List[dict]:
        """
        Process all stocks from a CSV file.
        
        Args:
            file_path: Path to CSV file containing multiple stocks
            
        Returns:
            List of dictionaries with results for each stock
        """
        from .data_processor import StockDataProcessor
        from .model import LSTMModel
        
        # Load data
        all_stocks_df = pd.read_csv(file_path)
        results = []
        
        for stock, stock_df in all_stocks_df.groupby('Index'):
            print(f"Processing stock: {stock}")
            
            try:
                # Process data
                processor = StockDataProcessor()
                processed_df, scaler = processor.preprocess_stock_data(stock_df)
                dates, X, Y = processor.df_to_windowed_df_simple(processed_df)
                
                # Split data
                data_splits = processor.split_data(dates, X, Y)
                
                # Train model
                model = LSTMModel()
                trained_model = model.train_simple(
                    data_splits['X_train'], data_splits['y_train'],
                    data_splits['X_val'], data_splits['y_val'],
                    epochs=50
                )
                
                # Predict future
                predictor = StockPredictor(trained_model, scaler)
                future_predictions = predictor.predict_future_steps(
                    data_splits['X_test'][-1], 
                    steps=config.BATCH_PREDICTION_DAYS
                )
                
                # Generate future dates
                future_dates = pd.date_range(
                    start=dates[-1], 
                    periods=config.BATCH_PREDICTION_DAYS, 
                    freq='D'
                )
                
                # Store results
                for date, pred in zip(future_dates, future_predictions):
                    results.append({
                        'Stock': stock,
                        'Date': date,
                        'Actual': None,
                        'Prediction': pred
                    })
                    
            except Exception as e:
                print(f"Error processing {stock}: {str(e)}")
                continue
        
        return results
    
    def save_results(self, results: List[dict], output_path: str) -> None:
        """
        Save batch processing results to CSV.
        
        Args:
            results: List of result dictionaries
            output_path: Output file path
        """
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
