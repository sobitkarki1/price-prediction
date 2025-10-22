# Random Forest Price Prediction Model

## Overview

This folder contains a Random Forest-based stock price prediction model for NABIL Bank, designed as an alternative to the LightGBM approach.

## Model Architecture

**Algorithm**: Random Forest Regressor
- **Trees**: 200 estimators
- **Max Depth**: 15
- **Features**: 55 technical indicators
- **Prediction Horizon**: 5 days ahead
- **Validation**: 5-fold Time Series Cross-Validation

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Average R² Score** | 0.6402 |
| **Average RMSE** | 132.18 NPR |
| **Average MAE** | 101.14 NPR |
| **Average MAPE** | 6.82% |

### Fold-by-Fold Results

| Fold | Train Size | Test Size | RMSE | MAE | MAPE | R² Score |
|------|-----------|-----------|------|-----|------|----------|
| 1 | 436 | 435 | 238.95 | 200.91 | 11.03% | 0.4509 |
| 2 | 871 | 435 | 140.83 | 102.97 | 4.75% | 0.7202 |
| 3 | 1,306 | 435 | 125.04 | 82.61 | 5.18% | 0.9150 |
| 4 | 1,741 | 435 | 84.35 | 69.11 | 8.64% | 0.1774 |
| 5 | 2,176 | 435 | 71.74 | 50.10 | 4.52% | 0.9374 |

## Features Used

### Price Features (Core)
- Open, High, Low, Close
- Daily returns (linear and log)
- Price range (volatility)

### Moving Averages
- Simple Moving Averages (SMA): 5, 10, 20, 30 days
- Exponential Moving Averages (EMA): 5, 10, 20, 30 days
- Price-to-SMA ratios

### Volatility Indicators
- Standard deviation (10, 20, 30 days)
- Coefficient of variation
- Average True Range (ATR-14)
- Bollinger Bands (upper, lower, position)

### Momentum Indicators
- RSI (14-period)
- MACD (signal, histogram)

### Volume Features
- Volume change
- Volume moving averages (10, 20 days)
- Volume ratio

### Lag Features
- Price lags: 1, 2, 3, 5, 7 days
- Return lags: 1, 2, 3, 5, 7 days

### Time Features
- Day of week
- Month
- Quarter

## Top 10 Most Important Features

1. **Close** (0.1202) - Current closing price
2. **High** (0.0994) - Daily high price
3. **Low** (0.0914) - Daily low price
4. **sma_5** (0.0839) - 5-day simple moving average
5. **Open** (0.0817) - Opening price
6. **ema_5** (0.0814) - 5-day exponential moving average
7. **ema_10** (0.0793) - 10-day exponential moving average
8. **close_lag_1** (0.0729) - Previous day's close
9. **close_lag_2** (0.0439) - Close from 2 days ago
10. **close_lag_3** (0.0327) - Close from 3 days ago

## Key Insights

### Strengths
✅ **Better average performance** than LightGBM (R² = 0.64 vs 0.57)
✅ **Lower error metrics** (RMSE 132 vs 143-152)
✅ **Clearer feature importance** - focuses on recent price action
✅ **More stable** - less prone to overfitting
✅ **Interpretable** - tree-based decisions are easier to understand

### Weaknesses
⚠️ **Slower training** - Takes longer than gradient boosting
⚠️ **Fold 1 & 4 still struggle** - Limited training data is challenging
⚠️ **Memory intensive** - 200 trees store all training data
⚠️ **Still can't predict shocks** - No model can predict unexpected news

## Usage

```bash
# Run the Random Forest model
python src/random_forest_model/rf_price_prediction.py
```

## Comparison with Other Models

| Model | Features | R² Score | RMSE | MAE | MAPE |
|-------|----------|----------|------|-----|------|
| **Random Forest** | 55 | **0.6402** ✅ | **132.18** ✅ | **101.14** ✅ | **6.82%** ✅ |
| LightGBM (Basic) | 15 | 0.6578 | 143.25 | 104.98 | N/A |
| LightGBM (Full) | 83 | 0.5670 | 152.20 | 116.06 | N/A |
| LightGBM (Optimized) | 25 | 0.5602 | 147.75 | 115.11 | 7.72% |

**Winner**: Random Forest performs competitively with better average metrics!

## Recommendations

### When to Use Random Forest:
- You need **interpretable** feature importance
- You want **robust** predictions with less tuning
- You have **sufficient memory** for tree storage
- You prefer **stable** performance over peak performance

### When to Use LightGBM:
- You need **faster** training and prediction
- You have **very large** datasets
- You want to **fine-tune** hyperparameters extensively
- Memory is limited

### Best Practice:
**Use an ensemble of both!** Combine Random Forest + LightGBM predictions for potentially better results.

## Future Enhancements

1. **Hyperparameter tuning** with GridSearchCV
2. **Ensemble with LightGBM** using stacking
3. **Add more external features** (market indices, economic data)
4. **Implement feature selection** to reduce to top 30 features
5. **Add prediction intervals** for uncertainty quantification

## License

This model is for educational purposes only. Not financial advice.
