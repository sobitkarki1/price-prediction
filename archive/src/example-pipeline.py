import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

df = pd.read_csv("data/transactions.csv")

# Store metrics for each stock
all_metrics = []

for stock, group in df.groupby("Stock Symbol"):
    print(f"\n{'='*60}")
    print(f"Processing Stock: {stock}")
    print(f"{'='*60}")
    
    group = group.sort_values("Transaction ID")
    group["price_change"] = group["Price"].pct_change()
    group["vol_avg"] = group["Quantity"].rolling(20).mean()
    group["price_mean"] = group["Price"].rolling(20).mean()
    group = group.dropna()

    X = group[["Quantity", "price_mean", "vol_avg", "price_change"]]
    y = group["Price"].shift(-1).dropna()
    X = X.iloc[:-1]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = lgb.LGBMRegressor(verbose=-1)
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
        
        print(f"\nFold {fold}:")
        print(f"  Test Size: {len(test_idx):,} samples")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R² Score: {r2:.4f}")
    
    # Calculate average metrics across folds
    avg_rmse = np.mean([m['rmse'] for m in fold_metrics])
    avg_mae = np.mean([m['mae'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    
    print(f"\n{'-'*60}")
    print(f"Average Performance for {stock}:")
    print(f"  Avg RMSE: {avg_rmse:.2f}")
    print(f"  Avg MAE: {avg_mae:.2f}")
    print(f"  Avg R²: {avg_r2:.4f}")
    print(f"{'-'*60}")
    
    all_metrics.append({
        'stock': stock,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'avg_r2': avg_r2,
        'total_samples': len(X)
    })

# Summary
print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
for metric in all_metrics:
    print(f"\nStock: {metric['stock']}")
    print(f"  Total Samples: {metric['total_samples']:,}")
    print(f"  Avg RMSE: {metric['avg_rmse']:.2f}")
    print(f"  Avg MAE: {metric['avg_mae']:.2f}")
    print(f"  Avg R²: {metric['avg_r2']:.4f}")
