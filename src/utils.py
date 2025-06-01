"""
Utility Functions

Common utility functions used across the stock prediction project.
"""

import pandas as pd
import numpy as np
import os
import datetime
from typing import List, Dict, Any, Optional
import config


def create_output_directories() -> None:
    """Create all necessary output directories."""
    directories = [
        config.DATA_DIR,
        config.OUTPUT_DIR,
        config.MODEL_DIR,
        os.path.join(config.OUTPUT_DIR, 'predictions'),
        os.path.join(config.OUTPUT_DIR, 'classifications'),
        os.path.join(config.OUTPUT_DIR, 'plots')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def validate_data_format(df: pd.DataFrame) -> bool:
    """
    Validate that the input dataframe has the required format.
    
    Args:
        df: Input dataframe to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['Date', 'Close']
    
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns. Expected: {required_columns}")
        return False
    
    # Check if Date can be converted to datetime
    try:
        pd.to_datetime(df['Date'])
    except:
        print("Date column cannot be converted to datetime")
        return False
    
    # Check if Close is numeric
    if not pd.api.types.is_numeric_dtype(df['Close']):
        print("Close column is not numeric")
        return False
    
    return True


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate various evaluation metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with metric names and values
    """
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }


def save_predictions_to_csv(predictions: Dict[str, Any], filepath: str) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Dictionary containing prediction data
        filepath: Output file path
    """
    df = pd.DataFrame(predictions)
    df.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")


def load_stock_data(filepath: str, stock_symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Load stock data from CSV file.
    
    Args:
        filepath: Path to CSV file
        stock_symbol: Optional stock symbol to filter
        
    Returns:
        Loaded dataframe
    """
    try:
        df = pd.read_csv(filepath)
        
        if stock_symbol and 'Index' in df.columns:
            df = df[df['Index'] == stock_symbol].copy()
        
        if not validate_data_format(df):
            raise ValueError("Invalid data format")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def get_stock_symbols(filepath: str) -> List[str]:
    """
    Get list of unique stock symbols from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        List of unique stock symbols
    """
    try:
        df = pd.read_csv(filepath)
        if 'Index' in df.columns:
            return df['Index'].unique().tolist()
        else:
            return ['Unknown']
    except Exception as e:
        print(f"Error getting stock symbols: {str(e)}")
        return []


def format_date(date_obj: datetime.datetime) -> str:
    """
    Format datetime object to string.
    
    Args:
        date_obj: Datetime object
        
    Returns:
        Formatted date string
    """
    return date_obj.strftime('%Y-%m-%d')


def parse_date(date_str: str) -> datetime.datetime:
    """
    Parse date string to datetime object.
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
        
    Returns:
        Datetime object
    """
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')


def get_date_range(start_date: str, end_date: str, freq: str = 'D') -> pd.DatetimeIndex:
    """
    Generate date range between two dates.
    
    Args:
        start_date: Start date string
        end_date: End date string
        freq: Frequency (D=daily, B=business days, M=monthly)
        
    Returns:
        DatetimeIndex with date range
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def print_model_summary(model_metrics: Dict[str, float], stock_symbol: str = "Stock") -> None:
    """
    Print a summary of model performance.
    
    Args:
        model_metrics: Dictionary with model metrics
        stock_symbol: Stock symbol or name
    """
    print(f"\n=== Model Performance Summary for {stock_symbol} ===")
    print("-" * 50)
    for metric, value in model_metrics.items():
        print(f"{metric:10s}: {value:.6f}")
    print("-" * 50)


def log_processing_step(step: str, stock_symbol: str = "Stock") -> None:
    """
    Log processing steps for debugging.
    
    Args:
        step: Description of the processing step
        stock_symbol: Stock symbol being processed
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {stock_symbol}: {step}")


def check_gpu_availability() -> bool:
    """
    Check if GPU is available for TensorFlow.
    
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except:
        return False


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except:
        pass


class ProgressTracker:
    """Simple progress tracker for batch operations."""
    
    def __init__(self, total_items: int):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
        """
        self.total_items = total_items
        self.current_item = 0
        self.start_time = datetime.datetime.now()
    
    def update(self, item_name: str = "") -> None:
        """
        Update progress.
        
        Args:
            item_name: Name of current item being processed
        """
        self.current_item += 1
        percentage = (self.current_item / self.total_items) * 100
        elapsed = datetime.datetime.now() - self.start_time
        
        print(f"Progress: {self.current_item}/{self.total_items} ({percentage:.1f}%) - "
              f"Current: {item_name} - Elapsed: {elapsed}")
    
    def finish(self) -> None:
        """Print completion message."""
        total_time = datetime.datetime.now() - self.start_time
        print(f"Processing completed in {total_time}")


def memory_usage_check() -> Dict[str, float]:
    """
    Check current memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent
        }
    except ImportError:
        return {'status': 'psutil not available'}
