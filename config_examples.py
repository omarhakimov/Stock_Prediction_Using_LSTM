# Example configuration file for different environments

# Development settings
DEV_CONFIG = {
    'DATA_PATH': 'data/raw/sample_stock_data.csv',
    'EPOCHS': 10,  # Faster training for development
    'FUTURE_DAYS': 30,
    'BATCH_PREDICTION_DAYS': 100
}

# Production settings
PROD_CONFIG = {
    'DATA_PATH': 'data/raw/all_stocks_less_years.csv',
    'EPOCHS': 100,
    'FUTURE_DAYS': 60,
    'BATCH_PREDICTION_DAYS': 1370
}

# Testing settings
TEST_CONFIG = {
    'DATA_PATH': 'tests/data/test_stock_data.csv',
    'EPOCHS': 5,
    'FUTURE_DAYS': 10,
    'BATCH_PREDICTION_DAYS': 50
}
