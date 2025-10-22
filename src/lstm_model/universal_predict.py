"""
Universal LSTM Prediction Script
Make predictions for any stock in the trained model
"""

import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
LOOKBACK = 30  # Number of days to use for prediction
FORECAST_DAYS = 5  # Predicting 5 days ahead

print(f"{'='*60}")
print(f"Universal LSTM Stock Price Predictor")
print(f"{'='*60}\n")

# Load model and encoders
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "universal_lstm_model.h5")
encoder_path = os.path.join(script_dir, "stock_encoder.pkl")
scaler_x_path = os.path.join(script_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(script_dir, "scaler_y.pkl")

print("Loading model and encoders...")
model = keras.models.load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
with open(scaler_x_path, 'rb') as f:
    scaler_X = pickle.load(f)
with open(scaler_y_path, 'rb') as f:
    scaler_y = pickle.load(f)

print(f"✓ Model loaded successfully")
print(f"✓ Available stocks: {', '.join(label_encoder.classes_)}\n")

def prepare_features(df):
    """Prepare technical indicators for a stock dataframe"""
    df = df.copy()
    
    # Returns and volume changes
    df['return'] = df['Close'].pct_change()
    df['volume_change'] = df['Volume'].pct_change()
    
    # Moving averages
    for n in [5, 10, 20]:
        df[f'sma_{n}'] = df['Close'].rolling(n).mean()
        df[f'std_{n}'] = df['Close'].rolling(n).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df = df.dropna()
    return df

def predict_stock(stock_symbol, data_path=None):
    """
    Make a 5-day ahead prediction for a stock
    
    Args:
        stock_symbol: Stock symbol (e.g., 'NABIL', 'SCB')
        data_path: Path to stock CSV file (optional, auto-detected if None)
    
    Returns:
        Predicted price for 5 days ahead
    """
    
    print(f"{'-'*60}")
    print(f"Predicting {stock_symbol}")
    print(f"{'-'*60}\n")
    
    # Verify stock is in trained model
    if stock_symbol not in label_encoder.classes_:
        print(f"❌ Error: {stock_symbol} not in trained model")
        print(f"Available stocks: {', '.join(label_encoder.classes_)}")
        return None
    
    # Auto-detect data path if not provided
    if data_path is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")
        
        # Mapping of stock symbols to file names
        stock_files = {
            'NABIL': 'stock_daily.csv',
            'SCB': 'stock_daily_scb.csv'
        }
        
        if stock_symbol in stock_files:
            data_path = os.path.join(data_dir, stock_files[stock_symbol])
        else:
            data_path = os.path.join(data_dir, f"{stock_symbol.lower()}.csv")
    
    # Load data
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return None
    
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['Date'], keep='first')
    df = df[df['Close'] > 0]
    
    print(f"✓ Loaded data: {len(df)} days ({df['Date'].min()} to {df['Date'].max()})")
    
    # Prepare features
    df = prepare_features(df)
    
    # Get last 30 days
    if len(df) < LOOKBACK:
        print(f"❌ Error: Need at least {LOOKBACK} days of data, got {len(df)}")
        return None
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 
                   'volume_change', 'sma_5', 'sma_10', 'sma_20', 
                   'std_5', 'std_10', 'std_20', 'rsi']
    
    last_sequence = df[feature_cols].iloc[-LOOKBACK:].values
    last_date = df['Date'].iloc[-1]
    last_close = df['Close'].iloc[-1]
    
    # Scale features
    last_sequence_scaled = scaler_X.transform(last_sequence)
    last_sequence_scaled = last_sequence_scaled.reshape(1, LOOKBACK, len(feature_cols))
    
    # Get stock ID
    stock_id = label_encoder.transform([stock_symbol])[0]
    stock_id_input = np.array([[stock_id]])
    
    # Make prediction
    prediction_scaled = model.predict([last_sequence_scaled, stock_id_input], verbose=0)
    prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
    
    # Calculate metrics
    price_change = prediction - last_close
    pct_change = (price_change / last_close) * 100
    
    print(f"\n{'Results':^60}")
    print(f"{'-'*60}")
    print(f"  Last Date: {last_date.strftime('%Y-%m-%d')}")
    print(f"  Current Price: {last_close:.2f} NPR")
    print(f"  Predicted Price ({FORECAST_DAYS} days ahead): {prediction:.2f} NPR")
    print(f"  Expected Change: {price_change:+.2f} NPR ({pct_change:+.2f}%)")
    
    if pct_change > 0:
        print(f"  Signal: 🟢 BUY (Expected gain: {pct_change:.2f}%)")
    else:
        print(f"  Signal: 🔴 SELL (Expected loss: {abs(pct_change):.2f}%)")
    
    print(f"{'-'*60}\n")
    
    return {
        'stock': stock_symbol,
        'last_date': last_date,
        'current_price': last_close,
        'predicted_price': prediction,
        'price_change': price_change,
        'pct_change': pct_change,
        'forecast_days': FORECAST_DAYS
    }

# Example usage
if __name__ == "__main__":
    print(f"Making predictions for all available stocks...\n")
    
    results = []
    for stock in label_encoder.classes_:
        result = predict_stock(stock)
        if result:
            results.append(result)
    
    # Summary table
    print(f"\n{'='*60}")
    print(f"Summary of Predictions ({FORECAST_DAYS} days ahead)")
    print(f"{'='*60}\n")
    
    if results:
        print(f"{'Stock':<10} {'Current':<12} {'Predicted':<12} {'Change':<15} {'Signal':<10}")
        print(f"{'-'*60}")
        
        for r in results:
            signal = "🟢 BUY" if r['pct_change'] > 0 else "🔴 SELL"
            print(f"{r['stock']:<10} {r['current_price']:<12.2f} {r['predicted_price']:<12.2f} "
                  f"{r['pct_change']:+6.2f}% {signal:<10}")
        
        print(f"{'-'*60}\n")
