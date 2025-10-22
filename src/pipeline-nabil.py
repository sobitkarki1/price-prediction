import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# Get the correct path to stock_daily.csv
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")
data_path = os.path.join(data_dir, "stock_daily.csv")

df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Remove duplicates and invalid data
df = df.drop_duplicates(subset=['Date'], keep='first')
df = df[df['Close'] > 0]  # Remove zero prices

# ========== BASIC PRICE FEATURES ==========
df['return'] = df['Close'].pct_change()
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['price_range'] = (df['High'] - df['Low']) / df['Close']  # Daily volatility
df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']

# ========== VOLUME FEATURES ==========
df['volume_change'] = df['Volume'].pct_change()
df['price_volume'] = df['Close'] * df['Volume']  # Money flow
df['volume_price_trend'] = df['Volume'] * df['return']  # Momentum indicator

# ========== MOVING AVERAGES ==========
for n in [3, 5, 10, 20, 30]:
    # Price moving averages
    df[f'sma_{n}'] = df['Close'].rolling(n).mean()
    df[f'ema_{n}'] = df['Close'].ewm(span=n, adjust=False).mean()
    
    # Price distance from moving average
    df[f'price_sma_{n}_ratio'] = df['Close'] / df[f'sma_{n}']
    
    # Volatility
    df[f'std_{n}'] = df['Close'].rolling(n).std()
    df[f'cv_{n}'] = df[f'std_{n}'] / df[f'sma_{n}']  # Coefficient of variation
    
    # Volume moving averages
    df[f'vol_sma_{n}'] = df['Volume'].rolling(n).mean()
    df[f'vol_ratio_{n}'] = df['Volume'] / df[f'vol_sma_{n}']

# ========== MOMENTUM INDICATORS ==========
# RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_14'] = calculate_rsi(df['Close'], 14)
df['rsi_7'] = calculate_rsi(df['Close'], 7)

# MACD (Moving Average Convergence Divergence)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_diff'] = df['macd'] - df['macd_signal']

# Stochastic Oscillator
low_14 = df['Low'].rolling(14).min()
high_14 = df['High'].rolling(14).max()
df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
df['stoch_d'] = df['stoch_k'].rolling(3).mean()

# Rate of Change
for n in [5, 10, 20]:
    df[f'roc_{n}'] = ((df['Close'] - df['Close'].shift(n)) / df['Close'].shift(n)) * 100

# ========== BOLLINGER BANDS ==========
for n in [20]:
    sma = df['Close'].rolling(n).mean()
    std = df['Close'].rolling(n).std()
    df[f'bb_upper_{n}'] = sma + (2 * std)
    df[f'bb_lower_{n}'] = sma - (2 * std)
    df[f'bb_width_{n}'] = (df[f'bb_upper_{n}'] - df[f'bb_lower_{n}']) / sma
    df[f'bb_position_{n}'] = (df['Close'] - df[f'bb_lower_{n}']) / (df[f'bb_upper_{n}'] - df[f'bb_lower_{n}'])

# ========== TREND INDICATORS ==========
# Average Directional Index (ADX) - simplified
df['high_low_diff'] = df['High'] - df['Low']
df['high_close_diff'] = abs(df['High'] - df['Close'].shift(1))
df['low_close_diff'] = abs(df['Low'] - df['Close'].shift(1))
df['true_range'] = df[['high_low_diff', 'high_close_diff', 'low_close_diff']].max(axis=1)
df['atr_14'] = df['true_range'].rolling(14).mean()

# ========== LAG FEATURES ==========
for lag in [1, 2, 3, 5, 7]:
    df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
    df[f'return_lag_{lag}'] = df['return'].shift(lag)

# ========== TIME-BASED FEATURES ==========
df['day_of_week'] = df['Date'].dt.dayofweek
df['day_of_month'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter

# ========== TARGET ==========
df['target'] = df['Close'].shift(-5)  # 5 days ahead

# Drop rows with NaN values
df = df.dropna()

X = df.drop(['Date','target'], axis=1)
y = df['target']

print(f"{'='*60}")
print(f"NABIL Bank Stock Price Prediction (5 days ahead)")
print(f"{'='*60}")
print(f"Total samples: {len(X):,}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Features: {X.columns.tolist()}")
print(f"\n")

tscv = TimeSeriesSplit(n_splits=5)
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    # Enhanced model with better hyperparameters
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])
    
    # Calculate metrics
    y_test = y.iloc[test_idx]
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    fold_metrics.append({
        'fold': fold,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'test_size': len(test_idx)
    })
    
    print(f"Fold {fold}:")
    print(f"  Train Size: {len(train_idx):,} samples")
    print(f"  Test Size: {len(test_idx):,} samples")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")
    print()

# Calculate average metrics across folds
avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
avg_mae = np.mean([m['mae'] for m in fold_metrics])
avg_r2 = np.mean([m['r2'] for m in fold_metrics])

print(f"{'-'*60}")
print(f"Average Performance Across All Folds:")
print(f"  Avg RMSE: {avg_rmse:.2f}")
print(f"  Avg MAE: {avg_mae:.2f}")
print(f"  Avg R²: {avg_r2:.4f}")
print(f"{'-'*60}")
