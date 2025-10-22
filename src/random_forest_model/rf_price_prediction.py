import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# Get the correct path to stock_daily.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")
data_path = os.path.join(data_dir, "stock_daily.csv")

df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Remove duplicates and invalid data
df = df.drop_duplicates(subset=['Date'], keep='first')
df = df[df['Close'] > 0]

print(f"{'='*60}")
print(f"Random Forest - NABIL Stock Price Prediction")
print(f"{'='*60}\n")

# ========== FEATURE ENGINEERING ==========

# Basic price features
df['return'] = df['Close'].pct_change()
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['price_range'] = (df['High'] - df['Low']) / df['Close']

# Volume features
df['volume_change'] = df['Volume'].pct_change()
df['vol_sma_10'] = df['Volume'].rolling(10).mean()
df['vol_sma_20'] = df['Volume'].rolling(20).mean()
df['vol_ratio_10'] = df['Volume'] / df['vol_sma_10']

# Moving averages - multiple timeframes
for n in [5, 10, 20, 30]:
    df[f'sma_{n}'] = df['Close'].rolling(n).mean()
    df[f'ema_{n}'] = df['Close'].ewm(span=n, adjust=False).mean()
    df[f'price_sma_{n}_ratio'] = df['Close'] / df[f'sma_{n}']

# Volatility
for n in [10, 20, 30]:
    df[f'std_{n}'] = df['Close'].rolling(n).std()
    df[f'cv_{n}'] = df[f'std_{n}'] / df['Close'].rolling(n).mean()

# ATR - Average True Range
df['high_low'] = df['High'] - df['Low']
df['high_close'] = abs(df['High'] - df['Close'].shift(1))
df['low_close'] = abs(df['Low'] - df['Close'].shift(1))
df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr_14'] = df['true_range'].rolling(14).mean()

# RSI - Relative Strength Index
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_14'] = calculate_rsi(df['Close'], 14)

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

# Target - predict 5 days ahead
df['target'] = df['Close'].shift(-5)

# Drop NaN
df = df.dropna()

# Prepare features
feature_cols = [col for col in df.columns if col not in ['Date', 'target']]
X = df[feature_cols]
y = df['target']

print(f"Dataset Info:")
print(f"  Total samples: {len(X):,}")
print(f"  Total features: {len(feature_cols)}")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  Prediction horizon: 5 days ahead\n")

# ========== RANDOM FOREST MODEL ==========
print(f"{'='*60}")
print(f"Training Random Forest Model with Time Series CV")
print(f"{'='*60}\n")

tscv = TimeSeriesSplit(n_splits=5)
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    # Random Forest with optimized parameters
    model = RandomForestRegressor(
        n_estimators=200,          # Number of trees
        max_depth=15,              # Maximum depth of trees
        min_samples_split=10,      # Minimum samples to split node
        min_samples_leaf=4,        # Minimum samples in leaf
        max_features='sqrt',       # Number of features for best split
        bootstrap=True,            # Use bootstrap samples
        random_state=42,
        n_jobs=-1,                 # Use all CPU cores
        verbose=0
    )
    
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    
    # Calculate metrics
    y_test = y.iloc[test_idx]
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    fold_metrics.append({
        'fold': fold,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'test_size': len(test_idx)
    })
    
    print(f"Fold {fold}:")
    print(f"  Train Size: {len(train_idx):,} samples")
    print(f"  Test Size: {len(test_idx):,} samples")
    print(f"  RMSE: {rmse:.2f} NPR")
    print(f"  MAE: {mae:.2f} NPR")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R² Score: {r2:.4f}")
    print()

# Calculate average metrics
avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
avg_mae = np.mean([m['mae'] for m in fold_metrics])
avg_r2 = np.mean([m['r2'] for m in fold_metrics])
avg_mape = np.mean([m['mape'] for m in fold_metrics])

print(f"{'-'*60}")
print(f"Average Performance (Random Forest):")
print(f"  Avg RMSE: {avg_rmse:.2f} NPR")
print(f"  Avg MAE: {avg_mae:.2f} NPR")
print(f"  Avg MAPE: {avg_mape:.2f}%")
print(f"  Avg R²: {avg_r2:.4f}")
print(f"{'-'*60}")

# ========== FEATURE IMPORTANCE ==========
print(f"\n{'='*60}")
print("Feature Importance Analysis")
print(f"{'='*60}\n")

# Train final model on all data
final_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
final_model.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 20 Most Important Features:")
for idx, row in feature_importance.head(20).iterrows():
    print(f"  {row['feature']:30s} : {row['importance']:.4f}")

print(f"\n{'='*60}")
print("Random Forest Model Complete!")
print(f"{'='*60}")
