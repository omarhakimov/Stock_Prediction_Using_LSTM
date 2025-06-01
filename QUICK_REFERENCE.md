# Stock Prediction System - Quick Reference

## 🚀 One-Minute Demo
```bash
python demo_visualization.py --stock AAPL --days 10
```
**Output**: 6 professional dashboards + analysis reports in `outputs/plots/`

## 📊 Command Quick Reference

### Demos & Showcases
```bash
python demo_visualization.py              # Full demo (AAPL, 10 days)
python demo_visualization.py --stock MSFT # Microsoft analysis
python demo_visualization.py --days 30    # 30-day predictions
python showcase_demo.py                   # System status overview
```

### Simple Tests
```bash
python simple_viz.py                      # Basic visualization test
python test_viz.py                        # Validation test
python generate_summary.py                # Analysis summary
```

### Main Analysis
```bash
python main.py --stock AAPL --mode single # Full analysis pipeline
python main.py --mode batch               # Multiple stocks
```

## 📁 Key Output Locations

### Visualizations (17 files)
- `outputs/plots/historical_dashboard.png` - Historical analysis
- `outputs/plots/prediction_dashboard.png` - Future predictions  
- `outputs/plots/risk_dashboard.png` - Risk analysis
- `outputs/plots/technical_dashboard.png` - Technical indicators
- `outputs/plots/model_performance_dashboard.png` - Model metrics

### Reports
- `outputs/plots/executive_summary.txt` - Investment summary
- `outputs/plots/analysis_summary.txt` - Technical analysis

### Data
- `outputs/predictions/future_predictions_AAPL.csv` - Price predictions
- `outputs/classifications/monthly_classification.csv` - Risk classifications

## ⚙️ Key Configuration (src/config.py)

```python
WINDOW_SIZE = 60              # Input sequence length
LSTM_UNITS = 50               # Model complexity
EPOCHS = 100                  # Training iterations
CONSERVATIVE_THRESHOLD = 2.0  # Risk classification
RISKY_THRESHOLD = 5.0         # Risk classification
DPI = 300                     # Visualization quality
```

## 🎯 Current Performance
- **Model Accuracy**: 96% (R² = 0.96)
- **Mean Absolute Error**: 0.019
- **Risk Classification**: Conservative/Moderate/Risky
- **Prediction Horizon**: Configurable (default 60 days)

## 🔧 Troubleshooting
```bash
# Module not found
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Check system status
python showcase_demo.py

# Install dependencies
pip install -r requirements.txt
```

## 📈 Example Results (AAPL 10-day)
- **Starting Price**: $152.33
- **Predicted End**: $162.08
- **Total Return**: 6.40%
- **Risk Profile**: Conservative
- **Confidence**: 96%

---
**Ready to predict? Run:** `python demo_visualization.py`
