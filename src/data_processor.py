"""
Data Processing Module

Handles data loading, preprocessing, normalization, and windowing for stock price prediction.
"""

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import config


class StockDataProcessor:
    """Handles all data preprocessing operations for stock price prediction."""
    
    def __init__(self, window_size: int = config.WINDOW_SIZE):
        """
        Initialize the data processor.
        
        Args:
            window_size: Number of previous days to use for prediction
        """
        self.window_size = window_size
        self.scaler = None
        
    def str_to_datetime(self, date_str: str) -> datetime.datetime:
        """
        Convert string date to datetime object.
        
        Args:
            date_str: Date string in format 'YYYY-MM-DD'
            
        Returns:
            datetime object
        """
        split = date_str.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)
    
    def load_and_preprocess(self, file_path: str, stock_symbol: Optional[str] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Load and preprocess stock data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            stock_symbol: Optional stock symbol to filter data
            
        Returns:
            Tuple of (processed_dataframe, scaler)
        """
        # Load data
        df = pd.read_csv(file_path)
        
        # Filter by stock symbol if provided
        if stock_symbol and 'Index' in df.columns:
            df = df[df['Index'] == stock_symbol].copy()
        
        # Select relevant columns
        df = df[['Date', 'Close']].copy()
        
        # Filter by date range
        df = df[(df['Date'] >= config.DATE_RANGE_START) & 
                (df['Date'] <= config.DATE_RANGE_END)]
        
        # Check if we have any data after filtering
        if len(df) == 0:
            raise ValueError(f"No data found for stock symbol '{stock_symbol}' in the specified date range")
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Normalize the Close prices
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df['Close'] = self.scaler.fit_transform(df[['Close']])
        
        # Set date as index
        df.set_index('Date', inplace=True)
        
        return df, self.scaler
    
    def preprocess_stock_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
        """
        Preprocess stock data (alternative method from notebook).
        
        Args:
            df: Raw dataframe with Date and Close columns
            
        Returns:
            Tuple of (processed_dataframe, scaler)
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Close']]
        
        # Check if we have any data
        if len(df) == 0:
            raise ValueError("No data available for preprocessing")
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df['Close'] = self.scaler.fit_transform(df[['Close']])
        df.index = df.pop('Date')
        
        return df, self.scaler
    
    def df_to_windowed_df(self, dataframe: pd.DataFrame, 
                         first_date_str: str, last_date_str: str) -> pd.DataFrame:
        """
        Convert dataframe to windowed format for time series prediction.
        
        Args:
            dataframe: Input dataframe with Close prices
            first_date_str: Start date for windowing
            last_date_str: End date for windowing
            
        Returns:
            Windowed dataframe ready for model training
        """
        first_date = self.str_to_datetime(first_date_str)
        last_date = self.str_to_datetime(last_date_str)
        
        target_date = first_date
        dates, X, Y = [], [], []
        
        last_time = False
        while True:
            df_subset = dataframe.loc[:target_date].tail(self.window_size + 1)
            
            if len(df_subset) != self.window_size + 1:
                print(f'Error: Window of size {self.window_size} is too large for date {target_date}')
                break
                
            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]
            
            dates.append(target_date)
            X.append(x)
            Y.append(y)
            
            # Move to next week
            next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
            if len(next_week) < 2:
                break
                
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
            
            if last_time:
                break
                
            target_date = next_date
            
            if target_date >= last_date:
                last_time = True
        
        # Create return dataframe
        ret_df = pd.DataFrame({'Target Date': dates})
        
        X = np.array(X)
        for i in range(self.window_size):
            ret_df[f'Target-{self.window_size-i}'] = X[:, i]
            
        ret_df['Target'] = Y
        
        return ret_df
    
    def df_to_windowed_df_simple(self, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple windowing function (from notebook implementation).
        
        Args:
            dataframe: Input dataframe with Close prices
            
        Returns:
            Tuple of (dates, X, Y) arrays
        """
        dates, X, Y = [], [], []
        
        for i in range(len(dataframe) - self.window_size):
            x = dataframe['Close'].iloc[i:i + self.window_size].values
            y = dataframe['Close'].iloc[i + self.window_size]
            dates.append(dataframe.index[i + self.window_size])
            X.append(x)
            Y.append(y)
            
        return np.array(dates), np.array(X, dtype=np.float32).reshape(-1, self.window_size, 1), np.array(Y, dtype=np.float32)
    
    def windowed_df_to_date_X_y(self, windowed_dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert windowed dataframe to arrays for model training.
        
        Args:
            windowed_dataframe: Windowed dataframe from df_to_windowed_df
            
        Returns:
            Tuple of (dates, X, y) arrays
        """
        df_as_np = windowed_dataframe.to_numpy()
        
        dates = df_as_np[:, 0]
        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
        Y = df_as_np[:, -1]
        
        return dates, X.astype(np.float32), Y.astype(np.float32)
    
    def split_data(self, dates: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Split data into train, validation, and test sets.
        
        Args:
            dates: Array of dates
            X: Feature array
            y: Target array
            
        Returns:
            Dictionary containing split data
        """
        n_samples = len(dates)
        train_end = int(n_samples * config.TRAIN_SPLIT)
        val_end = int(n_samples * (config.TRAIN_SPLIT + config.VALIDATION_SPLIT))
        
        return {
            'dates_train': dates[:train_end],
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'dates_val': dates[train_end:val_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'dates_test': dates[val_end:],
            'X_test': X[val_end:],
            'y_test': y[val_end:]
        }
