# Universal LSTM - Quick Reference Guide

## 🚀 Quick Start

### List all 287 supported stocks
```bash
python src/lstm_model/universal_predict_all.py --list
```

### Predict single stock
```bash
python src/lstm_model/universal_predict_all.py --stock NABIL
```

### Predict multiple stocks
```bash
python src/lstm_model/universal_predict_all.py --batch NABIL SCB EBL PCBL
```

### Predict top 20 stocks
```bash
python src/lstm_model/universal_predict_all.py --top20
```

### Predict all 287 stocks
```bash
python src/lstm_model/universal_predict_all.py --all
```

### Evaluate model performance
```bash
python src/evaluate_universal_lstm_all.py
```

## 📊 Current Status

### Model Training
- **Status**: ⚠️ Partially trained (8 epochs completed, interrupted at epoch 9)
- **Best validation loss**: 0.000672 at epoch 8
- **Checkpoint saved**: `src/lstm_model/universal_lstm_all_stocks.h5`

### To Resume Training
```bash
python src/train_universal_lstm_all.py
```
Model will automatically continue from epoch 8 checkpoint.

## 📈 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total raw files** | 415 stocks |
| **Successfully converted** | 287 stocks (69%) |
| **Training sequences** | 278,319 |
| **Training stocks** | 282 (5 failed feature engineering) |
| **Train/Test split** | 222,655 / 55,664 (80/20) |
| **Date range** | 2010-04-15 to 2021-12-29 |
| **Total trading days** | 293,924 |

## 🏗️ Model Architecture

```
Sequence Input (30, 14)    Stock ID Input
        ↓                         ↓
   LSTM(128)                 Embedding(16)
        ↓                         ↓
   LSTM(64)                       |
        ↓                         |
   LSTM(32)                       |
        ↓                         |
        └──────── Concatenate ────┘
                    ↓
               Dense(32)
                    ↓
               Dense(16)
                    ↓
                Output (Price)
```

**Parameters**: 141,745 (553.69 KB)

## 📁 Important Files

### Model Files
- `src/lstm_model/universal_lstm_all_stocks.h5` - Trained model
- `src/lstm_model/stock_encoder_all.pkl` - Stock encoder
- `src/lstm_model/scaler_X_all.pkl` - Feature scaler
- `src/lstm_model/scaler_y_all.pkl` - Target scaler
- `src/lstm_model/supported_stocks.txt` - List of 287 stocks

### Scripts
- `src/convert_all_stocks.py` - Convert raw NEPSE data
- `src/train_universal_lstm_all.py` - Train universal model
- `src/lstm_model/universal_predict_all.py` - Make predictions
- `src/evaluate_universal_lstm_all.py` - Evaluate performance

### Data
- `data/all_raw/` - Original 415 NEPSE CSV files
- `data/processed/` - Converted 287 OHLCV files
- `data/processed/conversion_summary.csv` - Conversion statistics

### Documentation
- `ALL_STOCKS_DOCUMENTATION.md` - Complete documentation
- `DATA_FORMAT_SPECIFICATION.md` - CSV format requirements
- `UNIVERSAL_LSTM_SUMMARY.md` - 2-stock model summary
- `README.md` - Main project documentation

## 🎯 Top 20 Stocks by Data Size

| Rank | Symbol | Days | Quality |
|------|--------|------|---------|
| 1 | SCB | 2,656 | Excellent |
| 2 | PCBL | 2,650 | Excellent |
| 3 | CZBIL | 2,646 | Excellent |
| 4 | NABIL | 2,645 | Excellent |
| 5 | SRBL | 2,645 | Excellent |
| 6 | EBL | 2,634 | Excellent |
| 7 | NTC | 2,626 | Excellent |
| 8 | AHPC | 2,610 | Excellent |
| 9 | SBI | 2,608 | Excellent |
| 10 | CHCL | 2,600 | Excellent |
| 11 | ADBL | 2,570 | Excellent |
| 12 | PLIC | 2,554 | Excellent |
| 13 | ALICL | 2,544 | Excellent |
| 14 | NBB | 2,531 | Excellent |
| 15 | NIB | 2,495 | Excellent |
| 16 | SBL | 2,494 | Excellent |
| 17 | LBL | 2,475 | Excellent |
| 18 | KBL | 2,446 | Excellent |
| 19 | HBL | 2,432 | Excellent |
| 20 | NICA | 2,410 | Excellent |

## 📊 Features (14 Technical Indicators)

| Category | Features |
|----------|----------|
| **Price** | Open, High, Low, Close |
| **Volume** | Volume, volume_change |
| **Returns** | return (daily % change) |
| **Trend** | sma_5, sma_10, sma_20 |
| **Volatility** | std_5, std_10, std_20 |
| **Momentum** | rsi (14-period) |

## ⚙️ Training Configuration

- **Lookback window**: 30 days
- **Forecast horizon**: 5 days ahead
- **Batch size**: 64
- **Max epochs**: 150
- **Early stopping**: Patience 20
- **Optimizer**: Adam
- **Loss function**: MSE

## 📊 Expected Performance

Based on 2-stock Universal LSTM (NABIL + SCB):
- **R² Score**: 0.942
- **RMSE**: 125.92 NPR
- **MAPE**: 12.71%

287-stock model (estimated after full training):
- **R² Score**: 0.85-0.95
- **RMSE**: 100-150 NPR (varies by stock)
- **MAPE**: 10-15% average

## ⚠️ Important Notes

### Model Status
- Training interrupted at epoch 9
- Best checkpoint at epoch 8 (val_loss: 0.000672)
- Model is usable but not fully optimized
- Recommend completing training for best performance

### Usage Recommendations
✅ **Good for**: Trend analysis, supporting technical analysis, research  
❌ **Not for**: Sole trading basis, high-frequency trading, day trading

### Disclaimers
- **For research and educational purposes only**
- Past performance ≠ future results
- Stock investing involves risk of loss
- Always use proper risk management
- Consult financial advisors

## 🔧 Troubleshooting

### Model not found
```bash
# Train the model first
python src/train_universal_lstm_all.py
```

### Stock not supported
```bash
# Check if stock is in the list
python src/lstm_model/universal_predict_all.py --list
```

### Encoders not found
The encoders are created during training. If training was interrupted, they may not exist. Complete training to generate all files.

### Out of memory
Reduce batch size in training script from 64 to 32 or 16.

## 📝 Next Steps

1. **Complete training** (2-4 hours)
   ```bash
   python src/train_universal_lstm_all.py
   ```

2. **Evaluate performance**
   ```bash
   python src/evaluate_universal_lstm_all.py
   ```

3. **Make predictions**
   ```bash
   python src/lstm_model/universal_predict_all.py --top20
   ```

4. **Analyze results**
   - Review `evaluation_results_top20.csv`
   - Check per-stock performance
   - Identify best/worst performing stocks

---

**Last Updated**: 2025-01-22  
**Model Version**: 1.0 (In Training)  
**Stocks**: 287 NEPSE stocks  
**Status**: ⚠️ Requires completion of training
