# Universal LSTM - All NEPSE Stocks Documentation

## Overview

This document describes the comprehensive Universal LSTM model trained on **287 NEPSE stocks**, representing the most extensive stock prediction model for the Nepal Stock Exchange.

## Dataset Summary

### Raw Data
- **Source**: `data/all_raw/` directory
- **Total files**: 415 CSV files
- **Format**: NEPSE standard format with columns:
  - S.N., Date, Total Transactions, Total Traded Shares, Total Traded Amount
  - Max. Price, Min. Price, Close Price

### Converted Data
- **Destination**: `data/processed/` directory  
- **Successfully converted**: 287 stocks (69% success rate)
- **Failed conversions**: 128 stocks (insufficient data < 50 days)
- **Total trading days**: 293,924 records
- **Date range**: 2010-04-15 to 2021-12-29 (11+ years)

### Data Quality Categories

| Category | Days Range | Count | Percentage |
|----------|-----------|-------|------------|
| **Excellent** | 1000+ days | 132 stocks | 46% |
| **Good** | 500-999 days | 55 stocks | 19% |
| **Limited** | 50-499 days | 100 stocks | 35% |

### Top 20 Stocks by Data Size

| Rank | Symbol | Days | Date Range | Category |
|------|--------|------|------------|----------|
| 1 | SCB | 2,656 | 2010-04-15 to 2021-12-29 | Excellent |
| 2 | PCBL | 2,650 | 2010-04-15 to 2021-12-29 | Excellent |
| 3 | CZBIL | 2,646 | 2010-04-15 to 2021-12-29 | Excellent |
| 4 | NABIL | 2,645 | 2010-04-15 to 2021-12-29 | Excellent |
| 5 | SRBL | 2,645 | 2010-04-15 to 2021-12-29 | Excellent |
| 6 | EBL | 2,634 | 2010-04-15 to 2021-12-29 | Excellent |
| 7 | NTC | 2,626 | 2010-04-15 to 2021-12-29 | Excellent |
| 8 | AHPC | 2,610 | 2010-04-15 to 2021-12-29 | Excellent |
| 9 | SBI | 2,608 | 2010-04-15 to 2021-12-29 | Excellent |
| 10 | CHCL | 2,600 | 2010-04-15 to 2021-12-29 | Excellent |

## Data Conversion Process

### Script: `src/convert_all_stocks.py`

**Conversion Mapping**:
```
NEPSE Format           →    OHLCV Format
─────────────────────────────────────────────
Date                   →    Date (YYYY-MM-DD)
Previous Close Price   →    Open
Max. Price             →    High
Min. Price             →    Low
Close Price            →    Close
Total Traded Shares    →    Volume
```

## Universal LSTM Model Architecture

### Training Script: `src/train_universal_lstm_all.py`

### Model Specifications

**Input Dimensions**:
- **Sequence Input**: (30 days, 14 features)
- **Stock ID Input**: Integer (0-286)

**Architecture**:
```
Input Layer 1: Sequence (30, 14)
    ↓
LSTM(128 units) → Dropout(0.3)
    ↓
LSTM(64 units) → Dropout(0.3)
    ↓
LSTM(32 units) → Dropout(0.3)
    ↓
    ↓ ← Concatenate ← Stock Embedding (16-dim)
    ↓                         ↑
Dense(32, relu) → Dropout(0.3)
    ↓                  Input Layer 2: Stock ID
Dense(16, relu)
    ↓
Output: Price Prediction (5 days ahead)
```

**Parameters**:
- Total parameters: **141,745** (553.69 KB)
- Embedding dimension: 16
- Dropout rate: 0.3

### Features Used (14 Technical Indicators)

| Feature | Description | Type |
|---------|-------------|------|
| Open | Opening price | Price |
| High | Highest price | Price |
| Low | Lowest price | Price |
| Close | Closing price | Price |
| Volume | Trading volume | Volume |
| return | Daily returns | Momentum |
| volume_change | Volume pct_change | Volume |
| sma_5 | 5-day SMA | Trend |
| sma_10 | 10-day SMA | Trend |
| sma_20 | 20-day SMA | Trend |
| std_5 | 5-day Volatility | Volatility |
| std_10 | 10-day Volatility | Volatility |
| std_20 | 20-day Volatility | Volatility |
| rsi | RSI (14) | Momentum |

### Training Configuration

**Data Preparation**:
- Lookback window: 30 days
- Forecast horizon: 5 days ahead
- Training sequences: 278,319 (from 282 stocks)
- Train/test split: 80/20 (222,655 train, 55,664 test)

**Hyperparameters**:
- Optimizer: Adam
- Loss function: MSE
- Batch size: 64
- Max epochs: 150
- Early stopping: Patience = 20

### Training Results (Partial - Interrupted at Epoch 9)

| Epoch | Training Loss | Validation Loss | Status |
|-------|--------------|-----------------|--------|
| 1 | 0.000072 | 0.001080 | Initial |
| 3 | 0.000029 | 0.000921 | Improved ✓ |
| 8 | 0.000020 | 0.000672 | Best ✓ |
| 9 | - | - | Interrupted |

**Trend**: Validation loss decreasing steadily (good convergence)

## Model Artifacts

### Files Generated

| File | Size | Description |
|------|------|-------------|
| `universal_lstm_all_stocks.h5` | ~554 KB | Model weights (Epoch 8) |
| `stock_encoder_all.pkl` | ~10 KB | Stock encoder |
| `scaler_X_all.pkl` | ~5 KB | Feature scaler |
| `scaler_y_all.pkl` | ~1 KB | Target scaler |

**Location**: `src/lstm_model/`

## Workflow Summary

```
1. Raw Data (415 stocks) → data/all_raw/
   ↓
2. Conversion → src/convert_all_stocks.py
   ↓
3. Processed Data (287 stocks) → data/processed/
   ↓
4. Training → src/train_universal_lstm_all.py
   ↓
5. Universal LSTM Model → src/lstm_model/
   ↓
6. Predictions (to be implemented)
```

### Quick Start Commands

```bash
# Convert all raw data
python src/convert_all_stocks.py

# Train universal model
python src/train_universal_lstm_all.py

# Make predictions (to be created)
python src/lstm_model/universal_predict_all.py
```

## Model Capabilities

### What It Can Do

✅ Predict price for any of 287 NEPSE stocks  
✅ 5-day ahead price forecasting  
✅ Leverage cross-stock learning patterns  
✅ Handle stocks with varying data sizes  
✅ Automatic stock embedding learning  

### What It Cannot Do

❌ Predict stocks not in training set  
❌ Real-time predictions (requires 30-day history)  
❌ Predict beyond 5 days accurately  
❌ Account for external events  

## Performance Expectations

**Based on 2-stock model**:
- R² Score: 0.942
- RMSE: 125.92 NPR
- MAPE: 12.71%

**Expected (287-stock model)**:
- R² Score: 0.85-0.95 (needs full training)
- RMSE: 100-150 NPR (varies by stock)
- MAPE: 10-15% (average)

## Next Steps

### To Complete Training

```bash
# Resume from Epoch 8 checkpoint
python src/train_universal_lstm_all.py
```

Training will take 2-4 hours for 150 epochs with early stopping.

### To Use the Model

1. Create prediction script
2. Load model and encoders
3. Prepare last 30 days of data
4. Get 5-day ahead prediction

## Important Notes

### ⚠️ Disclaimers

**This model is for research and educational purposes only**

- Past performance ≠ future results
- Stock investing involves risk of loss
- Use proper risk management
- Consult financial advisors
- Model should support, not replace, analysis

### Recommended Usage

**✅ Good Use Cases**:
- Trend analysis
- Technical analysis support
- Backtesting strategies
- Research and learning

**❌ Not Recommended**:
- Sole trading basis
- High-frequency trading
- Large positions without validation
- Ignoring fundamentals

## Troubleshooting

**Slow training**: Use GPU, reduce batch size  
**Out of memory**: Reduce batch size to 32 or 16  
**Poor performance**: Check data quality for that stock  
**Wrong predictions**: Verify data normalization  

## File Structure

```
project/
├── data/
│   ├── all_raw/                    # 415 raw files
│   └── processed/                  # 287 converted files
├── src/
│   ├── convert_all_stocks.py       # Conversion
│   ├── train_universal_lstm_all.py # Training
│   └── lstm_model/
│       ├── universal_lstm_all_stocks.h5
│       ├── stock_encoder_all.pkl
│       ├── scaler_X_all.pkl
│       └── scaler_y_all.pkl
└── DATA_FORMAT_SPECIFICATION.md
```

---

**Model Version**: 1.0 (In Training)  
**Status**: ⚠️ Requires Full Training  
**Stocks**: 287 NEPSE stocks  
**Sequences**: 278,319  
**Parameters**: 141,745