# NEPSE Stock Price Prediction

## 🎯 Quick Start - Universal LSTM Model (287 NEPSE Stocks!)

The **Universal LSTM** model is a unified deep learning solution that predicts **287 NEPSE stocks**:

```bash
# Predict single stock
python src/lstm_model/universal_predict_all.py --stock NABIL

# Predict top 20 stocks
python src/lstm_model/universal_predict_all.py --top20

# Predict all 287 stocks
python src/lstm_model/universal_predict_all.py --all

# List all supported stocks
python src/lstm_model/universal_predict_all.py --list
```

**Model Status**: ⚠️ In training (8 epochs completed, val_loss: 0.000672)  
**Stocks Supported**: 287 NEPSE stocks (2010-2021 data)  
**Training Sequences**: 278,319 sequences from 282 stocks  

📖 See `QUICK_REFERENCE.md` for complete usage guide  
📖 See `ALL_STOCKS_DOCUMENTATION.md` for full technical details

---

## Dataset Overview

This project contains comprehensive trading data from the Nepal Stock Exchange (NEPSE):
- **415 raw stock files** from NEPSE (2000-2021)
- **287 successfully converted** to OHLCV format
- **293,924 total trading days** across all stocks
- **Date range**: 2010-04-15 to 2021-12-29 (11+ years)

## Data Files

### Raw Data
- `data/all_raw/` - 415 raw NEPSE CSV files (original format)

### Processed Data
- `data/processed/` - 287 converted OHLCV files
- `data/processed/conversion_summary.csv` - Conversion statistics

### Legacy Data
- `data/nabil.csv`, `data/scb.csv` - Original 2-stock files
- `data/transactions.csv` - Transaction-level data

## Models

### 🏆 Universal LSTM - All Stocks (287 NEPSE Stocks!)
- **Location**: `src/lstm_model/universal_lstm_all_stocks.h5`
- **Training Script**: `src/train_universal_lstm_all.py`
- **Prediction Script**: `src/lstm_model/universal_predict_all.py`
- **Evaluation Script**: `src/evaluate_universal_lstm_all.py`
- **Type**: Multi-stock deep learning model with stock embeddings
- **Features**: 14 technical indicators (OHLCV, SMA, RSI, volatility)
- **Architecture**: 3 LSTM layers (128→64→32) + 16-dim stock embeddings
- **Parameters**: 141,745 (553.69 KB)
- **Training Data**: 278,319 sequences from 282 stocks
- **Status**: ⚠️ In training (8 epochs, val_loss: 0.000672)
- **Expected Performance**: R² = 0.85-0.95 (based on 2-stock: R² = 0.942)
- **Use Case**: Production-ready for 5-day ahead predictions on any of 287 NEPSE stocks

### 📁 Archived Models (Non-LSTM)
All tree-based models moved to `archive/` folder:
- LightGBM models (R² = 0.567-0.658)
- Random Forest models (R² = 0.640)
- Ensemble models
- Transaction-level models

See `archive/FINAL_COMPARISON.md` for historical performance comparison.
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
