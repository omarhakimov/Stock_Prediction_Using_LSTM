"""
Configuration settings for the stock prediction project.
"""

import os

# Base directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# Model parameters
WINDOW_SIZE = 60
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
LSTM_UNITS = 50
DENSE_UNITS = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Data processing parameters
WINDOW_SIZE = 60
FUTURE_DAYS = 60
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Classification thresholds (in percentage)
CONSERVATIVE_THRESHOLD = 2.0
RISKY_THRESHOLD = 5.0

# Visualization settings
PLOT_STYLE = 'default'
FIGURE_SIZE = (12, 8)
DPI = 300

# Classification colors
CLASSIFICATION_COLORS = {
    'Conservative': 'green',
    'Moderate': 'orange',
    'Risky': 'red',
    'Unknown': 'gray'
}

# File paths
DEFAULT_DATA_FILE = os.path.join(PROJECT_ROOT, 'sample_stock_data.csv')

# Output file patterns
FUTURE_PREDICTIONS_FILE = 'future_predictions_{symbol}_{days}days.csv'
DAILY_CLASSIFICATIONS_FILE = 'daily_classifications_{symbol}.csv'
MONTHLY_CLASSIFICATIONS_FILE = 'monthly_classifications_{symbol}.csv'
MODEL_FILE = '{symbol}_model.keras'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s'
