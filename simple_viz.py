#!/usr/bin/env python3
"""
Simple visualization generator for stock predictions.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting visualization generation...")

# Create plots directory
plots_dir = os.path.join('outputs', 'plots')
os.makedirs(plots_dir, exist_ok=True)
print(f"Created plots directory: {plots_dir}")

# Load historical data
try:
    historical_data = pd.read_csv('data/raw/sample_stock_data.csv')
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    print(f"Loaded historical data: {len(historical_data)} rows")
    
    # Create historical price plot
    plt.figure(figsize=(12, 8))
    plt.plot(historical_data['Date'], historical_data['Close'], linewidth=2, color='blue')
    plt.title('AAPL Historical Stock Prices', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    hist_plot_path = os.path.join(plots_dir, 'historical_prices.png')
    plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved historical plot: {hist_plot_path}")
    
except Exception as e:
    print(f"Error with historical data: {e}")

# Load future predictions
try:
    future_data = pd.read_csv('outputs/predictions/future_predictions_AAPL.csv')
    future_data['Date'] = pd.to_datetime(future_data['Date'])
    print(f"Loaded future predictions: {len(future_data)} rows")
    
    # Create future predictions plot
    plt.figure(figsize=(12, 8))
    plt.plot(future_data['Date'], future_data['Predicted Prices'], 
             marker='o', linewidth=2, color='orange', markersize=6)
    plt.title('AAPL Future Price Predictions', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    future_plot_path = os.path.join(plots_dir, 'future_predictions.png')
    plt.savefig(future_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved future predictions plot: {future_plot_path}")
    
except Exception as e:
    print(f"Error with future predictions: {e}")

# Load monthly classifications
try:
    monthly_data = pd.read_csv('outputs/classifications/monthly_classification.csv')
    if not monthly_data.empty:
        monthly_data['Dates'] = pd.to_datetime(monthly_data['Dates'])
        print(f"Loaded monthly classifications: {len(monthly_data)} rows")
        
        # Create risk classification plot
        plt.figure(figsize=(10, 6))
        
        # Classification distribution pie chart
        class_counts = monthly_data['Classification'].value_counts()
        colors = {'Conservative': 'green', 'Moderate': 'orange', 'Risky': 'red'}
        plot_colors = [colors.get(cls, 'gray') for cls in class_counts.index]
        
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                colors=plot_colors, startangle=90)
        plt.title('Risk Classification Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        risk_plot_path = os.path.join(plots_dir, 'risk_classification.png')
        plt.savefig(risk_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved risk classification plot: {risk_plot_path}")
    else:
        print("Monthly classification data is empty")
        
except Exception as e:
    print(f"Error with monthly classifications: {e}")

# Create combined view
try:
    if 'historical_data' in locals() and 'future_data' in locals():
        plt.figure(figsize=(16, 8))
        
        # Plot historical data
        plt.plot(historical_data['Date'], historical_data['Close'], 
                label='Historical Prices', linewidth=2, color='blue')
        
        # Plot future predictions
        plt.plot(future_data['Date'], future_data['Predicted Prices'], 
                label='Future Predictions', linewidth=2, color='red', 
                linestyle='--', marker='o', markersize=4)
        
        plt.title('AAPL Complete Timeline: Historical + Predictions', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        combined_plot_path = os.path.join(plots_dir, 'combined_timeline.png')
        plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined timeline plot: {combined_plot_path}")
        
except Exception as e:
    print(f"Error creating combined plot: {e}")

print("\\nVisualization generation completed!")
print(f"Check the plots directory: {plots_dir}")

# List generated files
try:
    files = os.listdir(plots_dir)
    if files:
        print("\\nGenerated files:")
        for file in files:
            print(f"  - {file}")
    else:
        print("\\nNo files were generated")
except:
    print("\\nCould not list generated files")
