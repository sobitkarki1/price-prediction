"""
Random Forest with 10x Parameters
Testing if significantly more trees and depth improves performance
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import time

# Get the correct path to stock_daily.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")
data_path = os.path.join(data_dir, "stock_daily.csv")

df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates(subset=['Date'], keep='first')
df = df[df['Close'] > 0]

print(f"{'='*60}")
print(f"Random Forest - 10x Parameters Test")
print(f"{'='*60}\n")

# ========== FEATURE ENGINEERING ==========
df['return'] = df['Close'].pct_change()
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['price_range'] = (df['High'] - df['Low']) / df['Close']
df['volume_change'] = df['Volume'].pct_change()

for n in [5, 10, 20, 30]:
    df[f'sma_{n}'] = df['Close'].rolling(n).mean()
    df[f'ema_{n}'] = df['Close'].ewm(span=n, adjust=False).mean()
    df[f'price_sma_{n}_ratio'] = df['Close'] / df[f'sma_{n}']
    df[f'std_{n}'] = df['Close'].rolling(n).std()
    df[f'cv_{n}'] = df[f'std_{n}'] / df['Close'].rolling(n).mean()

df['vol_sma_10'] = df['Volume'].rolling(10).mean()
df['vol_sma_20'] = df['Volume'].rolling(20).mean()
df['vol_ratio_10'] = df['Volume'] / df['vol_sma_10']

# ATR
df['high_low'] = df['High'] - df['Low']
df['high_close'] = abs(df['High'] - df['Close'].shift(1))
df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr_14'] = df['true_range'].rolling(14).mean()

# RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi_14'] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_diff'] = df['macd'] - df['macd_signal']

# Bollinger Bands
sma_20 = df['Close'].rolling(20).mean()
std_20 = df['Close'].rolling(20).std()
df['bb_upper'] = sma_20 + (2 * std_20)
df['bb_lower'] = sma_20 - (2 * std_20)
df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# Lag features
for lag in [1, 2, 3, 5, 7]:
    df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    df[f'return_lag_{lag}'] = df['return'].shift(lag)

# Time features
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter

df['target'] = df['Close'].shift(-5)
df = df.dropna()

feature_cols = [col for col in df.columns if col not in ['Date', 'target']]
X = df[feature_cols]
y = df['target']

print(f"Dataset: {len(X):,} samples, {len(feature_cols)} features\n")

# ========== MODEL COMPARISON ==========
print(f"{'='*60}")
print("Comparing Normal vs 10x Parameters")
print(f"{'='*60}\n")

configs = [
    {
        'name': 'Normal Parameters',
        'params': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    {
        'name': '10x Parameters',
        'params': {
            'n_estimators': 2000,      # 10x trees
            'max_depth': 150,          # 10x depth
            'min_samples_split': 2,    # More splitting (was 10)
            'min_samples_leaf': 1,     # Allow smaller leaves (was 4)
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
    }
]

tscv = TimeSeriesSplit(n_splits=5)

for config in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"{'='*60}")
    print(f"Parameters: {config['params']}")
    print()
    
    fold_metrics = []
    total_train_time = 0
    total_predict_time = 0
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = RandomForestRegressor(**config['params'])
        
        # Time training
        start_train = time.time()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_time = time.time() - start_train
        total_train_time += train_time
        
        # Time prediction
        start_pred = time.time()
        preds = model.predict(X.iloc[test_idx])
        pred_time = time.time() - start_pred
        total_predict_time += pred_time
        
        y_test = y.iloc[test_idx]
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
        fold_metrics.append({
            'fold': fold,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'train_time': train_time,
            'pred_time': pred_time
        })
        
        print(f"Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}% (Train: {train_time:.2f}s)")
    
    # Calculate averages
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    avg_mape = np.mean([m['mape'] for m in fold_metrics])
    
    print(f"\n{'-'*60}")
    print(f"Average Performance ({config['name']}):")
    print(f"  Avg RMSE: {avg_rmse:.2f} NPR")
    print(f"  Avg MAE: {avg_mae:.2f} NPR")
    print(f"  Avg MAPE: {avg_mape:.2f}%")
    print(f"  Avg R²: {avg_r2:.4f}")
    print(f"  Total Training Time: {total_train_time:.2f}s")
    print(f"  Total Prediction Time: {total_predict_time:.2f}s")
    print(f"{'-'*60}")

print(f"\n{'='*60}")
print("Comparison Complete!")
print(f"{'='*60}")
