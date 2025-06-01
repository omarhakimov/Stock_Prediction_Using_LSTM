# Configuration file for stock prediction project

import os

# Data settings
DATA_PATH = 'data/raw/all_stocks_less_years.csv'
DATE_RANGE_START = '2010-01-01'
DATE_RANGE_END = '2021-12-31'

# Model parameters
WINDOW_SIZE = 3
LSTM_UNITS = 64
DENSE_UNITS = 32
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Data splits
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Prediction settings
FUTURE_DAYS = 60
BATCH_PREDICTION_DAYS = 1370

# Risk classification thresholds
CONSERVATIVE_THRESHOLD = 5.0  # volatility < 5%
RISKY_THRESHOLD = 10.0        # volatility > 10%

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# Output file names
DAILY_CLASSIFICATION_FILE = 'daily_classification.csv'
MONTHLY_CLASSIFICATION_FILE = 'monthly_classification.csv'
FUTURE_PREDICTIONS_FILE = 'future_predictions_{stock}.csv'
BATCH_RESULTS_FILE = 'stock_predictions_combined.csv'

# Visualization settings
FIGURE_SIZE = (14, 7)
DPI = 300
PLOT_STYLE = 'seaborn-v0_8'

# Classification colors
CLASSIFICATION_COLORS = {
    'Conservative': 'green',
    'Moderate': 'orange', 
    'Risky': 'red',
    'Unknown': 'gray'
}
