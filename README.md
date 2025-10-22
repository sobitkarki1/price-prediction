# NEPSE Stock Price Prediction

## Dataset Overview

This project contains trading data from the Nepal Stock Exchange (NEPSE), including:
- **NBB** stock transaction data: 177,209 individual trade records
- **NABIL Bank** daily stock data: 2,649 trading days (2010-2021)

## Data Files

### Transaction-Level Data (NBB)
- `NEPSE136.csv` - Complete dataset (177,209 records)
- `NEPSE136_train.csv` - Training set (122,464 records, 80%)
- `NEPSE136_test.csv` - Test set (30,617 records, 20%)
- `transactions.csv` - Processed format for pipeline

### Daily Stock Data (NABIL)
- `nabil.csv` - Original NABIL Bank daily data
- `stock_daily.csv` - Processed OHLCV format for prediction

## Models

### 1. Transaction-Level Model (`example-pipeline.py`)
- **Target**: Predict next transaction price
- **Features**: 15 basic features (rolling averages, volume, price changes)
- **Performance**: R² = 0.97 (Excellent)
- **Use Case**: Very short-term price movement prediction

### 2. Daily Price Model (`pipeline-nabil.py`)
- **Target**: Predict stock price 5 days ahead
- **Features**: 83+ advanced technical indicators including:
  - Moving averages (SMA, EMA) for multiple periods
  - Momentum indicators (RSI, MACD, Stochastic, ROC)
  - Volatility measures (Bollinger Bands, ATR)
  - Volume analysis
  - Lag features
  - Time-based features
- **Performance**: R² = 0.57 (Moderate)
- **Use Case**: Medium-term trend prediction

## Model Performance Summary

| Model | Prediction | RMSE | MAE | R² Score | Status |
|-------|-----------|------|-----|----------|--------|
| NBB Transaction | Next trade | 17.95 | 11.12 | 0.9698 | ✅ Excellent |
| NABIL 5-day (Basic) | 5 days ahead | 143.25 | 104.98 | 0.6578 | ✅ Good |
| NABIL 5-day (Enhanced) | 5 days ahead | 152.20 | 116.06 | 0.5670 | ⚠️ Moderate |

## Key Findings

### Most Important Features for NABIL Prediction:
1. Volume moving averages (vol_sma_30, vol_sma_20)
2. Time features (month, day_of_month)
3. Volatility indicators (ATR, coefficient of variation)
4. Momentum indicators (MACD, RSI, Stochastic)
5. Rate of change indicators

### Limitations
- ⚠️ Model performance varies significantly across time periods
- ⚠️ Cannot predict market shocks or news-driven events
- ⚠️ Data quality issues (duplicates, missing values)
- ⚠️ **NOT recommended for real trading without additional safeguards**

## Scripts

- `prepare_transactions.py` - Converts train/test data to pipeline format
- `prepare_transactions_for_nabil.py` - Converts NABIL data to OHLCV format
- `example-pipeline.py` - Transaction-level prediction model
- `pipeline-nabil.py` - Daily price prediction model with 83+ features

## Usage

```bash
# Prepare data
python src/prepare_transactions.py
python src/prepare_transactions_for_nabil.py

# Run models
python src/example-pipeline.py
python src/pipeline-nabil.py
```
