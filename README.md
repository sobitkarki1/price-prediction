# NEPSE Stock Price Prediction

## 🎯 Quick Start - Universal LSTM Model (RECOMMENDED)

The **Universal LSTM** model is a unified deep learning solution that can predict multiple stocks:

```bash
# Make predictions for all stocks
python src/lstm_model/universal_predict.py

# Add new stocks and retrain
# 1. Add CSV to data/stock_daily_<SYMBOL>.csv
# 2. Update universal_lstm.py to load it
# 3. Retrain: python src/lstm_model/universal_lstm.py
```

**Performance**: R² = 0.942 | RMSE = 125.92 NPR | Currently supports: NABIL, SCB

📖 See `src/lstm_model/ADDING_STOCKS.md` for guide on adding new stocks

---

## Dataset Overview

This project contains trading data from the Nepal Stock Exchange (NEPSE), including:
- **NBB** stock transaction data: 177,209 individual trade records
- **NABIL Bank** daily stock data: 2,649 trading days (2010-2021)
- **SCB Bank** daily stock data: 2,660 trading days (2010-2021)

## Data Files

### Transaction-Level Data (NBB)
- `NEPSE136.csv` - Complete dataset (177,209 records)
- `NEPSE136_train.csv` - Training set (122,464 records, 80%)
- `NEPSE136_test.csv` - Test set (30,617 records, 20%)
- `transactions.csv` - Processed format for pipeline

### Daily Stock Data
- `nabil.csv` / `stock_daily.csv` - NABIL Bank OHLCV data
- `scb.csv` / `stock_daily_scb.csv` - SCB Bank OHLCV data

## Models

### 🏆 Universal LSTM (Recommended)
- **Location**: `src/lstm_model/universal_lstm.py`
- **Type**: Multi-stock deep learning model
- **Features**: Stock embeddings + 14 technical indicators
- **Architecture**: 3 LSTM layers (128→64→32) + stock embedding fusion
- **Performance**: R² = 0.942, RMSE = 125.92 NPR, MAPE = 12.71%
- **Stocks**: NABIL, SCB (easily add more!)
- **Use Case**: Production-ready for 5-day ahead predictions

### 1. Transaction-Level Model (`example-pipeline.py`)
- **Target**: Predict next transaction price
- **Features**: 15 basic features (rolling averages, volume, price changes)
- **Performance**: R² = 0.97 (Excellent)
- **Use Case**: Very short-term price movement prediction

### 2. Daily Price Models
Various implementations for 5-day ahead prediction:

**LightGBM Models**:
- `pipeline-nabil.py` - Enhanced with 83 features (R² = 0.567)
- Basic version - 15 features (R² = 0.658)

**Random Forest Models** (`src/random_forest_model/`):
- Standard RF - 55 features (R² = 0.640, RMSE = 132.18)
- 10x parameters - Tested and rejected (worse performance, 12x slower)

**Deep Learning Models** (`src/lstm_model/`):
- **Universal LSTM** - Multi-stock (R² = 0.942) ⭐ BEST
- Single-stock NABIL LSTM - (R² = 0.856) ⭐ EXCELLENT
- Single-stock SCB LSTM - (R² = -18.24) ❌ FAILED

## Model Performance Summary

### Complete Rankings (5-Day Ahead Prediction)

| Rank | Model | Type | R² Score | RMSE | MAE | MAPE | Status |
|------|-------|------|----------|------|-----|------|--------|
| 🥇 | Universal LSTM | Deep Learning | **0.9420** | 125.92 | 94.63 | 12.71% | ✅ BEST |
| 🥈 | NABIL LSTM | Deep Learning | **0.8560** | 112.51 | 81.56 | 7.22% | ✅ Excellent |
| 🥉 | LightGBM Basic | Gradient Boost | 0.6578 | 143.25 | 104.98 | 7.84% | ✅ Good |
| 4 | Random Forest | Tree Ensemble | 0.6402 | 132.18 | 101.14 | 6.82% | ✅ Good |
| 5 | RF Normal | Tree Ensemble | 0.6319 | 135.25 | 103.59 | 6.93% | ✅ Good |
| 6 | Ensemble RF+LGB | Hybrid | 0.5809 | 140.25 | 107.43 | 7.15% | ⚠️ Moderate |
| 7 | NABIL Enhanced | Gradient Boost | 0.5670 | 152.20 | 116.06 | N/A | ⚠️ Moderate |
| 8 | NABIL Optimized | Feature Select | 0.5602 | 148.91 | 113.84 | N/A | ⚠️ Moderate |
| ❌ | RF 10x Parameters | Tree Ensemble | 0.6121 | 139.15 | 106.78 | 7.08% | ❌ Worse + Slow |
| ❌ | SCB Models (All) | Any | < 0 | 250+ | 250+ | 40%+ | ❌ Unusable |

### Transaction-Level (NBB)
| Model | Target | RMSE | MAE | R² Score | Status |
|-------|--------|------|-----|----------|--------|
| LightGBM | Next trade | 17.95 | 11.12 | 0.9698 | ✅ Excellent |

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
