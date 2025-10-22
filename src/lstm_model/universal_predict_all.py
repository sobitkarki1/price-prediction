"""
Universal LSTM Prediction Script - All 287 NEPSE Stocks
Predicts 5-day ahead stock price for any supported stock symbol
"""

import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
import os
from datetime import datetime, timedelta

def load_supported_stocks():
    """Load list of supported stock symbols"""
    with open('src/lstm_model/supported_stocks.txt', 'r') as f:
        stocks = [line.strip() for line in f if line.strip()]
    return stocks

def prepare_prediction_data(stock_symbol, lookback=30):
    """
    Prepare data for prediction
    
    Args:
        stock_symbol: Stock ticker symbol (e.g., 'NABIL', 'SCB')
        lookback: Number of days to look back (default: 30)
    
    Returns:
        X_seq: Sequence data (30 days x 14 features)
        X_stock: Stock ID
        last_date: Last date in the data
        last_close: Last closing price
    """
    # Load stock data
    file_path = f'data/processed/stock_daily_{stock_symbol}.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Feature engineering
    df['return'] = df['Close'].pct_change()
    df['volume_change'] = df['Volume'].pct_change()
    
    # Moving averages
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    
    # Volatility
    df['std_5'] = df['Close'].rolling(window=5).std()
    df['std_10'] = df['Close'].rolling(window=10).std()
    df['std_20'] = df['Close'].rolling(window=20).std()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop NaN
    df = df.dropna()
    
    if len(df) < lookback:
        raise ValueError(f"Insufficient data for {stock_symbol}. Need at least {lookback} days, got {len(df)}")
    
    # Get last 30 days
    last_sequence = df.tail(lookback)
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'volume_change',
                    'sma_5', 'sma_10', 'sma_20', 'std_5', 'std_10', 'std_20', 'rsi']
    
    X_seq = last_sequence[feature_cols].values
    last_date = df['Date'].iloc[-1]
    last_close = df['Close'].iloc[-1]
    
    return X_seq, last_date, last_close

def predict_stock_price(stock_symbol, model, stock_encoder, scaler_X, scaler_y):
    """
    Predict stock price 5 days ahead
    
    Args:
        stock_symbol: Stock ticker symbol
        model: Trained Keras model
        stock_encoder: LabelEncoder for stocks
        scaler_X: Feature scaler
        scaler_y: Target scaler
    
    Returns:
        dict with prediction details
    """
    # Check if stock is supported
    supported_stocks = load_supported_stocks()
    if stock_symbol not in supported_stocks:
        raise ValueError(f"Stock {stock_symbol} not supported. Run with --list to see supported stocks.")
    
    # Prepare data
    X_seq, last_date, last_close = prepare_prediction_data(stock_symbol)
    
    # Encode stock
    stock_id = stock_encoder.transform([stock_symbol])[0]
    
    # Scale features
    X_seq_scaled = scaler_X.transform(X_seq)
    X_seq_scaled = X_seq_scaled.reshape(1, 30, 14)
    X_stock = np.array([stock_id])
    
    # Predict
    y_pred_scaled = model.predict([X_seq_scaled, X_stock], verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]
    
    # Calculate metrics
    price_change = y_pred - last_close
    pct_change = (price_change / last_close) * 100
    
    # Prediction date (5 trading days ahead ≈ 7 calendar days)
    prediction_date = last_date + timedelta(days=7)
    
    return {
        'symbol': stock_symbol,
        'last_date': last_date,
        'last_close': last_close,
        'predicted_price': y_pred,
        'price_change': price_change,
        'pct_change': pct_change,
        'prediction_date': prediction_date
    }

def display_prediction(result):
    """Display prediction results in a formatted way"""
    symbol = result['symbol']
    last_close = result['last_close']
    predicted_price = result['predicted_price']
    pct_change = result['pct_change']
    
    # Determine signal
    if pct_change > 5:
        signal = "🟢 STRONG BUY"
    elif pct_change > 2:
        signal = "🟢 BUY"
    elif pct_change > -2:
        signal = "🟡 HOLD"
    elif pct_change > -5:
        signal = "🔴 SELL"
    else:
        signal = "🔴 STRONG SELL"
    
    print("\n" + "="*80)
    print(f"PREDICTION FOR {symbol}")
    print("="*80)
    print(f"📅 Last Date:              {result['last_date'].strftime('%Y-%m-%d')}")
    print(f"💰 Last Close Price:       NPR {last_close:.2f}")
    print(f"🔮 Predicted Price (5d):   NPR {predicted_price:.2f}")
    print(f"📊 Expected Change:        NPR {result['price_change']:+.2f} ({pct_change:+.2f}%)")
    print(f"📅 Prediction Date:        ~{result['prediction_date'].strftime('%Y-%m-%d')}")
    print(f"🎯 Signal:                 {signal}")
    print("="*80)

def batch_predict(stock_list, model, stock_encoder, scaler_X, scaler_y, top_n=10):
    """
    Predict for multiple stocks and show top movers
    
    Args:
        stock_list: List of stock symbols
        model: Trained model
        stock_encoder: Stock encoder
        scaler_X: Feature scaler
        scaler_y: Target scaler
        top_n: Number of top stocks to show
    """
    results = []
    
    print(f"\n🔮 Predicting for {len(stock_list)} stocks...")
    
    for i, symbol in enumerate(stock_list, 1):
        try:
            result = predict_stock_price(symbol, model, stock_encoder, scaler_X, scaler_y)
            results.append(result)
            print(f"✅ [{i}/{len(stock_list)}] {symbol}: {result['pct_change']:+.2f}%")
        except Exception as e:
            print(f"❌ [{i}/{len(stock_list)}] {symbol}: {str(e)}")
    
    if not results:
        print("No successful predictions!")
        return
    
    # Sort by percentage change
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pct_change', ascending=False)
    
    print("\n" + "="*80)
    print(f"TOP {top_n} POTENTIAL GAINERS (Next 5 Days)")
    print("="*80)
    
    for i, row in results_df.head(top_n).iterrows():
        signal = "🟢" if row['pct_change'] > 0 else "🔴"
        print(f"{signal} {row['symbol']:8s} | Last: NPR {row['last_close']:8.2f} | "
              f"Predicted: NPR {row['predicted_price']:8.2f} | Change: {row['pct_change']:+6.2f}%")
    
    print("\n" + "="*80)
    print(f"TOP {top_n} POTENTIAL LOSERS (Next 5 Days)")
    print("="*80)
    
    for i, row in results_df.tail(top_n).iterrows():
        signal = "🟢" if row['pct_change'] > 0 else "🔴"
        print(f"{signal} {row['symbol']:8s} | Last: NPR {row['last_close']:8.2f} | "
              f"Predicted: NPR {row['predicted_price']:8.2f} | Change: {row['pct_change']:+6.2f}%")
    
    # Save to CSV
    results_df.to_csv('batch_predictions.csv', index=False)
    print(f"\n💾 All predictions saved to batch_predictions.csv")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Universal LSTM Stock Price Prediction')
    parser.add_argument('--stock', type=str, help='Stock symbol to predict')
    parser.add_argument('--list', action='store_true', help='List all supported stocks')
    parser.add_argument('--batch', nargs='+', help='Predict for multiple stocks')
    parser.add_argument('--top20', action='store_true', help='Predict for top 20 stocks by data size')
    parser.add_argument('--all', action='store_true', help='Predict for all 287 stocks')
    
    args = parser.parse_args()
    
    # List supported stocks
    if args.list:
        supported_stocks = load_supported_stocks()
        print(f"\n✅ {len(supported_stocks)} Supported Stocks:")
        print("="*80)
        for i, stock in enumerate(supported_stocks, 1):
            print(f"{stock:12s}", end='  ')
            if i % 6 == 0:
                print()
        print("\n" + "="*80)
        return
    
    # Load model and encoders
    print("📂 Loading model and encoders...")
    model = load_model('src/lstm_model/universal_lstm_all_stocks.h5')
    
    with open('src/lstm_model/stock_encoder_all.pkl', 'rb') as f:
        stock_encoder = pickle.load(f)
    with open('src/lstm_model/scaler_X_all.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('src/lstm_model/scaler_y_all.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    
    # Single stock prediction
    if args.stock:
        result = predict_stock_price(args.stock, model, stock_encoder, scaler_X, scaler_y)
        display_prediction(result)
    
    # Batch prediction
    elif args.batch:
        batch_predict(args.batch, model, stock_encoder, scaler_X, scaler_y)
    
    # Top 20 stocks
    elif args.top20:
        # Get top 20 from conversion summary
        summary = pd.read_csv('data/processed/conversion_summary.csv')
        top20 = summary.nlargest(20, 'days')['symbol'].tolist()
        batch_predict(top20, model, stock_encoder, scaler_X, scaler_y, top_n=10)
    
    # All stocks
    elif args.all:
        supported_stocks = load_supported_stocks()
        batch_predict(supported_stocks, model, stock_encoder, scaler_X, scaler_y, top_n=20)
    
    else:
        print("❌ Please specify --stock, --batch, --top20, --all, or --list")
        print("\nExamples:")
        print("  python src/lstm_model/universal_predict_all.py --stock NABIL")
        print("  python src/lstm_model/universal_predict_all.py --batch NABIL SCB EBL")
        print("  python src/lstm_model/universal_predict_all.py --top20")
        print("  python src/lstm_model/universal_predict_all.py --all")
        print("  python src/lstm_model/universal_predict_all.py --list")

if __name__ == "__main__":
    main()
