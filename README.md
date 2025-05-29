# 📈 LSTM Stock Price Prediction

This project uses an LSTM (Long Short-Term Memory) neural network to predict future stock prices based on historical stock data. It's built using Python and TensorFlow/Keras.

## 📁 Project Structure

LSTM-Stock-Prediction/
│
├── data/ # Contains historical stock price data (CSV format)
├── models/ # Saved trained LSTM models
├── plots/ # Generated plots from training and testing
├── lstm_stock_predictor.py # Main LSTM training and prediction script
├── utils.py # Helper functions (data loading, scaling, etc.)
├── README.md # Project documentation
└── requirements.txt # Python dependencies


## ⚙️ Requirements

- Python 3.7+
- TensorFlow / Keras
- NumPy
- pandas
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧠 Model Overview
Input: Historical stock price (e.g., Open, Close, Volume)

Output: Predicted stock price for the next day

Architecture: Stacked LSTM layers with dropout

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

## 🚀 How to Use
Prepare the Data:

Place your stock CSV file (e.g., AAPL.csv) into the data/ folder. Make sure it includes Date, Open, High, Low, Close, Volume.

Train the Model:
python lstm_stock_predictor.py

## View the Results:

Loss and accuracy will be printed during training.

Prediction results and plots are saved in the plots/ folder.

## 📊 Example Plot
You’ll see a graph like this showing the actual vs predicted stock price:


## 📌 Notes
You can customize the lookback window and epochs in the script.

Feature scaling is done using MinMaxScaler.

Performance can vary based on the stock and data length.

## 🧠 Future Work
Add support for multi-stock prediction

Deploy as a web app

Include technical indicators as features