#!/bin/bash

# Installation and setup script for Stock Price Prediction project

echo "🚀 Setting up Stock Price Prediction project..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
required_version="3.8"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Python 3.8 or higher is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,models}
mkdir -p outputs/{predictions,classifications,plots}
mkdir -p tests/{unit,integration,data}

# Download sample data (if URL provided)
echo "📊 Setting up sample data..."
if [ ! -f "data/raw/sample_stock_data.csv" ]; then
    python3 -c "
import pandas as pd
import numpy as np
import os

# Generate sample stock data
np.random.seed(42)
dates = pd.date_range('2010-01-01', '2021-12-31', freq='D')
prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)

sample_data = pd.DataFrame({
    'Date': dates,
    'Close': prices,
    'Index': 'SAMPLE'
})

os.makedirs('data/raw', exist_ok=True)
sample_data.to_csv('data/raw/sample_stock_data.csv', index=False)
print('✅ Sample data created at data/raw/sample_stock_data.csv')
"
fi

# Run a quick test
echo "🧪 Running quick test..."
python3 -c "
import sys
sys.path.append('src')
from src.data_processor import StockDataProcessor
from src.model import LSTMModel
print('✅ All modules imported successfully!')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the example: python main.py --mode single --future-days 30"
echo "3. Or use the Jupyter notebook: jupyter notebook notebooks/stock_analysis_example.ipynb"
echo ""
echo "📚 For more information, see README.md"
