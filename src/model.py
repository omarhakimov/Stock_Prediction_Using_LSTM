"""
LSTM Model Module

Contains the LSTM neural network model for stock price prediction.
"""

import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional
import config


class LSTMModel:
    """LSTM neural network model for stock price prediction."""
    
    def __init__(self, window_size: int = config.WINDOW_SIZE, 
                 lstm_units: int = config.LSTM_UNITS,
                 dense_units: int = config.DENSE_UNITS):
        """
        Initialize the LSTM model.
        
        Args:
            window_size: Input sequence length
            lstm_units: Number of LSTM units
            dense_units: Number of dense layer units
        """
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.model = None
        
    def build_model(self) -> Sequential:
        """
        Build the LSTM model architecture.
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            Input((self.window_size, 1)),
            LSTM(self.lstm_units),
            Dense(self.dense_units, activation='relu'),
            Dense(self.dense_units, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = config.EPOCHS,
              model_name: str = "stock_model") -> Sequential:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            model_name: Name for saving the model
            
        Returns:
            Trained model
        """
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(config.MODEL_DIR, f"{model_name}.keras"),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model
    
    def train_simple(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = config.EPOCHS) -> Sequential:
        """
        Simple training method (matches notebook implementation).
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            
        Returns:
            Trained model
        """
        self.model = Sequential([
            Input((X_train.shape[1], 1)),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mean_absolute_error']
        )
        
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            verbose=1
        )
        
        return self.model
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
        else:
            raise ValueError("Model has not been trained yet.")
    
    def load_model(self, filepath: str) -> Sequential:
        """
        Load a pre-trained model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        self.model = load_model(filepath)
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': loss,
            'mean_absolute_error': mae
        }
