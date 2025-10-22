import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
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

print(f"{'='*60}")
print(f"NABIL Bank - Optimized Feature Selection Model")
print(f"{'='*60}\n")

# ========== CAREFULLY SELECTED FEATURES ==========

# 1. Core Price Features
df['return'] = df['Close'].pct_change()
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['price_range'] = (df['High'] - df['Low']) / df['Close']

# 2. Volume Features (most important from analysis)
df['volume_change'] = df['Volume'].pct_change()
df['vol_sma_20'] = df['Volume'].rolling(20).mean()
df['vol_sma_30'] = df['Volume'].rolling(30).mean()
df['vol_ratio_20'] = df['Volume'] / df['vol_sma_20']
df['vol_ratio_30'] = df['Volume'] / df['vol_sma_30']

# 3. Key Moving Averages
df['sma_20'] = df['Close'].rolling(20).mean()
df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['price_sma_20_ratio'] = df['Close'] / df['sma_20']

# 4. Volatility Measures
df['std_20'] = df['Close'].rolling(20).std()
df['std_30'] = df['Close'].rolling(30).std()
df['cv_20'] = df['std_20'] / df['sma_20']  # Coefficient of variation
df['cv_30'] = df['std_30'] / df['Close'].rolling(30).mean()

# ATR - Average True Range
df['high_low_diff'] = df['High'] - df['Low']
df['high_close_diff'] = abs(df['High'] - df['Close'].shift(1))
df['low_close_diff'] = abs(df['Low'] - df['Close'].shift(1))
df['true_range'] = df[['high_low_diff', 'high_close_diff', 'low_close_diff']].max(axis=1)
df['atr_14'] = df['true_range'].rolling(14).mean()

# 5. Momentum Indicators
# RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_7'] = calculate_rsi(df['Close'], 7)
df['rsi_14'] = calculate_rsi(df['Close'], 14)

# MACD
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
df['roc_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
df['roc_20'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100

# 6. Bollinger Bands
sma = df['Close'].rolling(20).mean()
std = df['Close'].rolling(20).std()
df['bb_upper_20'] = sma + (2 * std)
df['bb_lower_20'] = sma - (2 * std)
df['bb_position_20'] = (df['Close'] - df['bb_lower_20']) / (df['bb_upper_20'] - df['bb_lower_20'])

# 7. Strategic Lag Features
df['close_lag_1'] = df['Close'].shift(1)
df['close_lag_2'] = df['Close'].shift(2)
df['close_lag_5'] = df['Close'].shift(5)
df['return_lag_1'] = df['return'].shift(1)
df['return_lag_2'] = df['return'].shift(2)

# 8. Time Features (important from analysis)
df['month'] = df['Date'].dt.month
df['day_of_month'] = df['Date'].dt.day
df['quarter'] = df['Date'].dt.quarter

# Target
df['target'] = df['Close'].shift(-5)  # 5 days ahead

# Drop rows with NaN values
df = df.dropna()

# Prepare features
feature_cols = [col for col in df.columns if col not in ['Date', 'target']]
X = df[feature_cols]
y = df['target']

print(f"Initial Features: {len(feature_cols)}")
print(f"Total samples: {len(X):,}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")

# ========== FEATURE SELECTION ==========
print(f"{'='*60}")
print("Performing Feature Selection...")
print(f"{'='*60}\n")

# Use SelectKBest to find top features
k_features = 25  # Select top 25 features
selector = SelectKBest(score_func=f_regression, k=k_features)
selector.fit(X, y)

# Get selected features
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

selected_features = feature_scores.head(k_features)['feature'].tolist()

print(f"Top {k_features} Selected Features:")
for idx, row in feature_scores.head(k_features).iterrows():
    print(f"  {row['feature']:30s} : {row['score']:10.2f}")

X_selected = X[selected_features]

print(f"\n{'='*60}")
print(f"Training Model with {len(selected_features)} Features")
print(f"{'='*60}\n")

# ========== CROSS-VALIDATION ==========
tscv = TimeSeriesSplit(n_splits=5)
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_selected), 1):
    # Optimized model parameters
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=25,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_selected.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X_selected.iloc[test_idx])
    
    # Calculate metrics
    y_test = y.iloc[test_idx]
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Calculate percentage errors
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
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R² Score: {r2:.4f}")
    print()

# Calculate average metrics
avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
avg_mae = np.mean([m['mae'] for m in fold_metrics])
avg_r2 = np.mean([m['r2'] for m in fold_metrics])
avg_mape = np.mean([m['mape'] for m in fold_metrics])

print(f"{'-'*60}")
print(f"Average Performance (Feature-Selected Model):")
print(f"  Avg RMSE: {avg_rmse:.2f} NPR")
print(f"  Avg MAE: {avg_mae:.2f} NPR")
print(f"  Avg MAPE: {avg_mape:.2f}%")
print(f"  Avg R²: {avg_r2:.4f}")
print(f"{'-'*60}")

# Feature importance from final model
print(f"\n{'='*60}")
print("Feature Importance (Selected Features)")
print(f"{'='*60}\n")

final_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=25,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)
final_model.fit(X_selected, y)

feature_importance = pd.DataFrame({
    'feature': X_selected.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Most Important Features:")
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:30s} : {row['importance']:8.1f}")

print(f"\n{'='*60}")
print("Model Optimization Complete!")
print(f"{'='*60}")
