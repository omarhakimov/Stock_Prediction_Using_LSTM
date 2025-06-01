"""
Visualization Module

Handles all plotting and visualization for stock price predictions and classifications.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Optional, List, Tuple
import config


class StockVisualizer:
    """Handles visualization of stock data, predictions, and classifications."""
    
    def __init__(self, style: str = config.PLOT_STYLE, 
                 figsize: Tuple[int, int] = config.FIGURE_SIZE):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use('default')  # Use default style as seaborn-v0_8 might not be available
        self.figsize = figsize
        self.colors = config.CLASSIFICATION_COLORS
    
    def plot_price_history(self, df: pd.DataFrame, title: str = "Stock Price History") -> None:
        """
        Plot historical stock prices.
        
        Args:
            df: DataFrame with Date index and Close column
            title: Plot title
        """
        plt.figure(figsize=self.figsize)
        plt.plot(df.index, df['Close'])
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_training_results(self, dates_train: np.ndarray, y_train: np.ndarray,
                            train_predictions: np.ndarray,
                            dates_val: np.ndarray, y_val: np.ndarray,
                            val_predictions: np.ndarray,
                            dates_test: np.ndarray, y_test: np.ndarray,
                            test_predictions: np.ndarray) -> None:
        """
        Plot training, validation, and test results.
        
        Args:
            dates_train: Training dates
            y_train: Training actual values
            train_predictions: Training predictions
            dates_val: Validation dates
            y_val: Validation actual values
            val_predictions: Validation predictions
            dates_test: Test dates
            y_test: Test actual values
            test_predictions: Test predictions
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(dates_train, train_predictions, label='Training Predictions', alpha=0.7)
        plt.plot(dates_train, y_train, label='Training Observations', alpha=0.7)
        plt.plot(dates_val, val_predictions, label='Validation Predictions', alpha=0.7)
        plt.plot(dates_val, y_val, label='Validation Observations', alpha=0.7)
        plt.plot(dates_test, test_predictions, label='Testing Predictions', alpha=0.7)
        plt.plot(dates_test, y_test, label='Testing Observations', alpha=0.7)
        
        plt.title('Model Performance: Predictions vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_data_splits(self, dates_train: np.ndarray, y_train: np.ndarray,
                        dates_val: np.ndarray, y_val: np.ndarray,
                        dates_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Plot data splits (train/validation/test).
        
        Args:
            dates_train: Training dates
            y_train: Training values
            dates_val: Validation dates
            y_val: Validation values
            dates_test: Test dates
            y_test: Test values
        """
        plt.figure(figsize=self.figsize)
        
        plt.plot(dates_train, y_train, label='Train', alpha=0.8)
        plt.plot(dates_val, y_val, label='Validation', alpha=0.8)
        plt.plot(dates_test, y_test, label='Test', alpha=0.8)
        
        plt.title('Data Splits')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_future_predictions(self, future_df: pd.DataFrame, 
                               title: str = "Future Stock Price Predictions") -> None:
        """
        Plot future price predictions.
        
        Args:
            future_df: DataFrame with Date and Predicted Prices columns
            title: Plot title
        """
        plt.figure(figsize=self.figsize)
        plt.plot(future_df['Date'], future_df['Predicted Prices'], 
                label='Future Predictions', color='orange', linewidth=2)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_daily_classifications(self, daily_df: pd.DataFrame, 
                                  title: str = "Daily Stock Classification Over Time") -> None:
        """
        Plot daily classifications with color coding.
        
        Args:
            daily_df: DataFrame with daily classifications
            title: Plot title
        """
        # Map classifications to numeric values for coloring
        classification_map = {"Conservative": 1, "Moderate": 2, "Risky": 3, "Unknown": 0}
        daily_df = daily_df.copy()
        daily_df["Classification Numeric"] = daily_df["Classification"].map(classification_map)
        
        # Map to colors
        color_map = {1: "green", 2: "orange", 3: "red", 0: "gray"}
        daily_colors = daily_df["Classification Numeric"].map(color_map)
        
        plt.figure(figsize=self.figsize)
        plt.plot(daily_df["Dates"], daily_df["Predicted Prices"], 
                label="Daily Predicted Prices", color="blue", alpha=0.6)
        plt.scatter(daily_df["Dates"], daily_df["Predicted Prices"], 
                   c=daily_colors, label="Daily Classification", alpha=0.8, s=10)
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Predicted Prices")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_classifications(self, monthly_df: pd.DataFrame, 
                                   title: str = "Monthly Stock Classification Over Time") -> None:
        """
        Plot monthly classifications with color coding.
        
        Args:
            monthly_df: DataFrame with monthly classifications
            title: Plot title
        """
        # Map classifications to numeric values for coloring
        classification_map = {"Conservative": 1, "Moderate": 2, "Risky": 3, "Unknown": 0}
        monthly_df = monthly_df.copy()
        monthly_df["Classification Numeric"] = monthly_df["Classification"].map(classification_map)
        
        # Map to colors
        color_map = {1: "lightgreen", 2: "yellow", 3: "darkred", 0: "gray"}
        monthly_colors = monthly_df["Classification Numeric"].map(color_map)
        
        plt.figure(figsize=self.figsize)
        plt.plot(monthly_df["Dates"], monthly_df["Mean"], 
                label="Monthly Mean Prices", color="blue", alpha=0.6)
        plt.scatter(monthly_df["Dates"], monthly_df["Mean"], 
                   c=monthly_colors, label="Monthly Classification", 
                   alpha=0.9, s=100, edgecolors="black")
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Mean Prices")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_future_with_classification(self, classified_future_df: pd.DataFrame,
                                       title: str = "Future Stock Price Predictions with Classification") -> None:
        """
        Plot future predictions with risk classifications.
        
        Args:
            classified_future_df: DataFrame with future predictions and classifications
            title: Plot title
        """
        # Remove rows with NaN classifications
        classified_future_df = classified_future_df.dropna(subset=["Classification"])
        
        # Map classification to colors
        colors = classified_future_df["Classification"].map(self.colors).fillna("gray")
        
        plt.figure(figsize=self.figsize)
        plt.plot(classified_future_df["Date"], classified_future_df["Predicted Prices"], 
                label="Future Predictions", color="orange", linewidth=2)
        plt.scatter(classified_future_df["Date"], classified_future_df["Predicted Prices"],
                   c=colors, label="Classification", s=30, alpha=0.8)
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Predicted Prices")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_classification_distribution(self, classifications: pd.Series, 
                                       title: str = "Risk Classification Distribution") -> None:
        """
        Plot distribution of risk classifications.
        
        Args:
            classifications: Series with classification values
            title: Plot title
        """
        plt.figure(figsize=(8, 6))
        
        # Count classifications
        class_counts = classifications.value_counts()
        colors = [self.colors.get(cls, 'gray') for cls in class_counts.index]
        
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title(title)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_volatility_over_time(self, df: pd.DataFrame, 
                                 title: str = "Volatility Over Time") -> None:
        """
        Plot volatility metrics over time.
        
        Args:
            df: DataFrame with Date, Mean, and Std Deviation columns
            title: Plot title
        """
        df = df.copy()
        df['Volatility'] = (df['Std Deviation'] / df['Mean']) * 100
        
        plt.figure(figsize=self.figsize)
        plt.plot(df['Dates'], df['Volatility'], linewidth=2)
        plt.axhline(y=config.CONSERVATIVE_THRESHOLD, color='green', linestyle='--', 
                   label=f'Conservative Threshold ({config.CONSERVATIVE_THRESHOLD}%)')
        plt.axhline(y=config.RISKY_THRESHOLD, color='red', linestyle='--', 
                   label=f'Risky Threshold ({config.RISKY_THRESHOLD}%)')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Volatility (%)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def save_all_plots(self, output_dir: str = config.OUTPUT_DIR) -> None:
        """
        Save all current plots to files.
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # This would need to be called after each plot creation
        # plt.savefig(os.path.join(plots_dir, 'plot_name.png'), dpi=config.DPI, bbox_inches='tight')
        pass
