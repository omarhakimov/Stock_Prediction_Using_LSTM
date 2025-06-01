import pandas as pd
import os
from datetime import datetime

# Generate summary report
plots_dir = 'outputs/plots'
report_path = os.path.join(plots_dir, 'analysis_summary.txt')

# Load data for summary
historical_data = pd.read_csv('data/raw/sample_stock_data.csv')
future_data = pd.read_csv('outputs/predictions/future_predictions_AAPL.csv')
monthly_data = pd.read_csv('outputs/classifications/monthly_classification.csv')

with open(report_path, 'w') as f:
    f.write('STOCK PREDICTION ANALYSIS SUMMARY\n')
    f.write('=' * 50 + '\n\n')
    f.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    
    f.write('HISTORICAL DATA ANALYSIS:\n')
    f.write('-' * 30 + '\n')
    f.write(f'Data points: {len(historical_data)}\n')
    f.write(f'Date range: {historical_data["Date"].iloc[0]} to {historical_data["Date"].iloc[-1]}\n')
    f.write(f'Price range: ${historical_data["Close"].min():.2f} - ${historical_data["Close"].max():.2f}\n')
    f.write(f'Average price: ${historical_data["Close"].mean():.2f}\n')
    f.write(f'Price volatility (std): ${historical_data["Close"].std():.2f}\n\n')
    
    total_return = ((future_data["Predicted Prices"].iloc[-1] / future_data["Predicted Prices"].iloc[0]) - 1) * 100
    f.write('FUTURE PREDICTIONS:\n')
    f.write('-' * 20 + '\n')
    f.write(f'Prediction period: {len(future_data)} days\n')
    f.write(f'Starting price: ${future_data["Predicted Prices"].iloc[0]:.2f}\n')
    f.write(f'Ending price: ${future_data["Predicted Prices"].iloc[-1]:.2f}\n')
    f.write(f'Total predicted return: {total_return:.2f}%\n\n')
    
    if not monthly_data.empty:
        class_counts = monthly_data["Classification"].value_counts()
        f.write('RISK CLASSIFICATION SUMMARY:\n')
        f.write('-' * 30 + '\n')
        for classification, count in class_counts.items():
            percentage = (count / len(monthly_data)) * 100
            f.write(f'{classification}: {count} periods ({percentage:.1f}%)\n')
    
    f.write('\nFILES GENERATED:\n')
    f.write('-' * 15 + '\n')
    f.write('- historical_prices.png: Historical price analysis\n')
    f.write('- future_predictions.png: Future predictions analysis\n')
    f.write('- combined_timeline.png: Historical + predicted timeline\n')
    f.write('- risk_classification.png: Risk classification analysis\n')
    f.write('- performance_dashboard.png: Model performance metrics\n')
    f.write('- advanced_analytics.png: Advanced statistical analysis\n')
    f.write('- analysis_summary.txt: This summary report\n')

print(f'Summary report saved: {report_path}')
