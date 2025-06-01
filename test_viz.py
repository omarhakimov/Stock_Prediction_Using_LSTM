#!/usr/bin/env python3
"""
Simple test visualization to debug issues.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src import config
    print("✓ Config loaded successfully")
except Exception as e:
    print(f"✗ Config error: {e}")

# Test data loading
print("Testing data loading...")

try:
    # Test historical data
    historical_path = os.path.join('data', 'raw', 'sample_stock_data.csv')
    print(f"Looking for historical data at: {historical_path}")
    if os.path.exists(historical_path):
        historical_data = pd.read_csv(historical_path)
        print(f"✓ Historical data loaded: {len(historical_data)} rows")
        print(historical_data.head())
    else:
        print(f"✗ Historical data not found at: {historical_path}")
except Exception as e:
    print(f"✗ Error loading historical data: {e}")

try:
    # Test future predictions
    future_path = os.path.join('outputs', 'predictions', 'future_predictions_AAPL.csv')
    print(f"\\nLooking for future predictions at: {future_path}")
    if os.path.exists(future_path):
        future_data = pd.read_csv(future_path)
        print(f"✓ Future predictions loaded: {len(future_data)} rows")
        print(future_data.head())
    else:
        print(f"✗ Future predictions not found at: {future_path}")
except Exception as e:
    print(f"✗ Error loading future predictions: {e}")

# Test directory creation
try:
    plots_dir = os.path.join('outputs', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"✓ Plots directory created/verified: {plots_dir}")
except Exception as e:
    print(f"✗ Error creating plots directory: {e}")

# Test simple plot
try:
    print("\\nTesting simple plot generation...")
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    plt.figure(figsize=(10, 6))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
    plt.title('Test Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    test_plot_path = os.path.join('outputs', 'plots', 'test_plot.png')
    plt.savefig(test_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if os.path.exists(test_plot_path):
        print(f"✓ Test plot saved successfully: {test_plot_path}")
    else:
        print("✗ Test plot was not saved")
        
except Exception as e:
    print(f"✗ Error creating test plot: {e}")

print("\\nTest completed!")
