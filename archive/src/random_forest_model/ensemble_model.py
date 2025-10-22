"""
Ensemble Model: Random Forest + LightGBM
Combines predictions from both models for potentially better performance
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# Get the correct path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")
data_path = os.path.join(data_dir, "stock_daily.csv")

df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates(subset=['Date'], keep='first')
df = df[df['Close'] > 0]

print(f"{'='*60}")
print(f"Ensemble Model: Random Forest + LightGBM")
print(f"{'='*60}\n")

# ========== FEATURE ENGINEERING ==========
df['return'] = df['Close'].pct_change()
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
df['price_range'] = (df['High'] - df['Low']) / df['Close']
df['volume_change'] = df['Volume'].pct_change()

for n in [5, 10, 20, 30]:
    df[f'sma_{n}'] = df['Close'].rolling(n).mean()
    df[f'ema_{n}'] = df['Close'].ewm(span=n, adjust=False).mean()
    df[f'std_{n}'] = df['Close'].rolling(n).std()

df['vol_sma_20'] = df['Volume'].rolling(20).mean()
df['atr_14'] = (df['High'] - df['Low']).rolling(14).mean()

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

# Lags
for lag in [1, 2, 3, 5]:
    df[f'close_lag_{lag}'] = df['Close'].shift(lag)

# Target
df['target'] = df['Close'].shift(-5)
df = df.dropna()

feature_cols = [col for col in df.columns if col not in ['Date', 'target']]
X = df[feature_cols]
y = df['target']

print(f"Dataset: {len(X):,} samples, {len(feature_cols)} features")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")

# ========== ENSEMBLE TRAINING ==========
print(f"{'='*60}")
print("Training Ensemble Model (RF + LightGBM)")
print(f"{'='*60}\n")

tscv = TimeSeriesSplit(n_splits=5)
fold_metrics = {
    'rf': [],
    'lgbm': [],
    'ensemble': []
}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    # Random Forest Model
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    # LightGBM Model
    lgbm_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    
    # Train both models
    rf_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    lgbm_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    # Get predictions
    rf_preds = rf_model.predict(X.iloc[test_idx])
    lgbm_preds = lgbm_model.predict(X.iloc[test_idx])
    
    # Ensemble: Average predictions (you can also try weighted average)
    ensemble_preds = 0.5 * rf_preds + 0.5 * lgbm_preds
    
    y_test = y.iloc[test_idx]
    
    # Calculate metrics for each model
    for name, preds in [('rf', rf_preds), ('lgbm', lgbm_preds), ('ensemble', ensemble_preds)]:
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        
        fold_metrics[name].append({
            'fold': fold,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        })
    
    print(f"Fold {fold} (Test Size: {len(test_idx):,}):")
    print(f"  Random Forest  - RMSE: {fold_metrics['rf'][-1]['rmse']:.2f}, R²: {fold_metrics['rf'][-1]['r2']:.4f}")
    print(f"  LightGBM       - RMSE: {fold_metrics['lgbm'][-1]['rmse']:.2f}, R²: {fold_metrics['lgbm'][-1]['r2']:.4f}")
    print(f"  Ensemble (50/50) - RMSE: {fold_metrics['ensemble'][-1]['rmse']:.2f}, R²: {fold_metrics['ensemble'][-1]['r2']:.4f}")
    print()

# ========== RESULTS COMPARISON ==========
print(f"{'='*60}")
print("Average Performance Comparison")
print(f"{'='*60}\n")

for name, label in [('rf', 'Random Forest'), ('lgbm', 'LightGBM'), ('ensemble', 'Ensemble')]:
    metrics = fold_metrics[name]
    avg_rmse = np.mean([m['rmse'] for m in metrics])
    avg_mae = np.mean([m['mae'] for m in metrics])
    avg_r2 = np.mean([m['r2'] for m in metrics])
    avg_mape = np.mean([m['mape'] for m in metrics])
    
    print(f"{label}:")
    print(f"  Avg RMSE: {avg_rmse:.2f} NPR")
    print(f"  Avg MAE: {avg_mae:.2f} NPR")
    print(f"  Avg MAPE: {avg_mape:.2f}%")
    print(f"  Avg R²: {avg_r2:.4f}")
    print()

# Determine winner
rf_r2 = np.mean([m['r2'] for m in fold_metrics['rf']])
lgbm_r2 = np.mean([m['r2'] for m in fold_metrics['lgbm']])
ensemble_r2 = np.mean([m['r2'] for m in fold_metrics['ensemble']])

best_r2 = max(rf_r2, lgbm_r2, ensemble_r2)
if ensemble_r2 == best_r2:
    winner = "🏆 Ensemble (Best Performance!)"
elif rf_r2 == best_r2:
    winner = "🏆 Random Forest (Best Performance!)"
else:
    winner = "🏆 LightGBM (Best Performance!)"

print(f"{'-'*60}")
print(f"Winner: {winner}")
print(f"{'-'*60}")

print(f"\n{'='*60}")
print("Ensemble Model Complete!")
print(f"{'='*60}")
