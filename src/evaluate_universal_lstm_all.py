"""
Evaluate Universal LSTM Model on All 287 NEPSE Stocks
"""

import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model():
    """Evaluate the Universal LSTM model on test set"""
    
    print("="*80)
    print("UNIVERSAL LSTM MODEL EVALUATION - 287 NEPSE STOCKS")
    print("="*80)
    
    # Load model
    print("\n📂 Loading model...")
    model_path = 'src/lstm_model/universal_lstm_all_stocks.h5'
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    model = load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    # Load encoders and scalers
    print("\n📂 Loading encoders and scalers...")
    try:
        with open('src/lstm_model/stock_encoder_all.pkl', 'rb') as f:
            stock_encoder = pickle.load(f)
        with open('src/lstm_model/scaler_X_all.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('src/lstm_model/scaler_y_all.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        print("✅ All encoders and scalers loaded")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Note: Encoders may not have been saved yet. They should be saved during training.")
        return
    
    # Load test data
    print("\n📂 Loading test data...")
    processed_dir = 'data/processed'
    
    # Get all stock files
    stock_files = [f for f in os.listdir(processed_dir) 
                   if f.startswith('stock_daily_') and f.endswith('.csv')]
    
    print(f"Found {len(stock_files)} stock files")
    
    # Load and combine all data
    all_data = []
    for file in stock_files:
        try:
            df = pd.read_csv(os.path.join(processed_dir, file))
            symbol = file.replace('stock_daily_', '').replace('.csv', '')
            df['stock'] = symbol
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.sort_values(['stock', 'Date'])
    
    print(f"📊 Total records: {len(combined_df):,}")
    print(f"📊 Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    # Feature engineering (same as training)
    print("\n⚙️ Engineering features...")
    
    def add_features(df):
        df = df.copy()
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
        
        return df.dropna()
    
    processed_stocks = combined_df.groupby('stock', group_keys=False).apply(add_features)
    print(f"✅ Features engineered: {len(processed_stocks):,} records after cleaning")
    
    # Create sequences
    print("\n⚙️ Creating sequences...")
    lookback = 30
    forecast_horizon = 5
    
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'volume_change',
                    'sma_5', 'sma_10', 'sma_20', 'std_5', 'std_10', 'std_20', 'rsi']
    
    X_seq, X_stock, y = [], [], []
    
    for stock in processed_stocks['stock'].unique():
        stock_data = processed_stocks[processed_stocks['stock'] == stock].copy()
        
        if len(stock_data) < lookback + forecast_horizon:
            continue
            
        stock_id = stock_encoder.transform([stock])[0]
        
        for i in range(len(stock_data) - lookback - forecast_horizon + 1):
            X_seq.append(stock_data[feature_cols].iloc[i:i+lookback].values)
            X_stock.append(stock_id)
            y.append(stock_data['Close'].iloc[i+lookback+forecast_horizon-1])
    
    X_seq = np.array(X_seq)
    X_stock = np.array(X_stock)
    y = np.array(y)
    
    print(f"✅ Created {len(X_seq):,} sequences")
    
    # Split test set (last 20%)
    train_size = int(len(X_seq) * 0.8)
    X_seq_test = X_seq[train_size:]
    X_stock_test = X_stock[train_size:]
    y_test = y[train_size:]
    
    print(f"📊 Test set: {len(X_seq_test):,} sequences")
    
    # Normalize
    print("\n⚙️ Normalizing data...")
    X_seq_test_scaled = scaler_X.transform(X_seq_test.reshape(-1, X_seq_test.shape[-1])).reshape(X_seq_test.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred_scaled = model.predict([X_seq_test_scaled, X_stock_test], verbose=1)
    
    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    print("\n📊 OVERALL PERFORMANCE METRICS")
    print("="*80)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    
    print(f"R² Score:      {r2:.4f}")
    print(f"RMSE:          {rmse:.2f} NPR")
    print(f"MAE:           {mae:.2f} NPR")
    print(f"MAPE:          {mape:.2f}%")
    
    # Per-stock performance
    print("\n📊 PER-STOCK PERFORMANCE (Top 20 by data size)")
    print("="*80)
    
    # Get top 20 stocks by data
    top_stocks = processed_stocks.groupby('stock').size().sort_values(ascending=False).head(20).index
    
    results = []
    for stock in top_stocks:
        stock_id = stock_encoder.transform([stock])[0]
        mask = X_stock_test == stock_id
        
        if mask.sum() == 0:
            continue
            
        y_test_stock = y_test[mask]
        y_pred_stock = y_pred[mask]
        
        r2_stock = r2_score(y_test_stock, y_pred_stock)
        rmse_stock = np.sqrt(mean_squared_error(y_test_stock, y_pred_stock))
        mape_stock = calculate_mape(y_test_stock, y_pred_stock)
        
        results.append({
            'Stock': stock,
            'Test Samples': mask.sum(),
            'R²': r2_stock,
            'RMSE': rmse_stock,
            'MAPE%': mape_stock
        })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results
    print("\n💾 Saving results...")
    results_df.to_csv('evaluation_results_top20.csv', index=False)
    print("✅ Results saved to evaluation_results_top20.csv")
    
    # Summary statistics
    print("\n📊 SUMMARY STATISTICS (Top 20 Stocks)")
    print("="*80)
    print(f"Average R²:    {results_df['R²'].mean():.4f}")
    print(f"Average RMSE:  {results_df['RMSE'].mean():.2f} NPR")
    print(f"Average MAPE:  {results_df['MAPE%'].mean():.2f}%")
    print(f"Best Stock:    {results_df.loc[results_df['R²'].idxmax(), 'Stock']} (R²={results_df['R²'].max():.4f})")
    print(f"Worst Stock:   {results_df.loc[results_df['R²'].idxmin(), 'Stock']} (R²={results_df['R²'].min():.4f})")
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    evaluate_model()
