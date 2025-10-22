# Documentation Index

## 📋 Quick Navigation

### Getting Started
1. **[README.md](README.md)** - Project overview and main entry point
2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick start commands and usage examples

### Comprehensive Guides
3. **[ALL_STOCKS_DOCUMENTATION.md](ALL_STOCKS_DOCUMENTATION.md)** - Complete technical documentation for 287-stock model
4. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current project status and next actions
5. **[DATA_FORMAT_SPECIFICATION.md](DATA_FORMAT_SPECIFICATION.md)** - CSV format requirements and conversion guide

### Historical Reference
6. **[UNIVERSAL_LSTM_SUMMARY.md](UNIVERSAL_LSTM_SUMMARY.md)** - Original 2-stock Universal LSTM implementation
7. **[archive/FINAL_COMPARISON.md](archive/FINAL_COMPARISON.md)** - Comparison of archived models (LightGBM, RF)

### Model-Specific
8. **[src/lstm_model/README.md](src/lstm_model/README.md)** - LSTM models documentation
9. **[src/lstm_model/ADDING_STOCKS.md](src/lstm_model/ADDING_STOCKS.md)** - Guide for adding new stocks

---

## 📁 File Organization

### Root Documentation (Current Work)
```
price-prediction/
├── README.md                           # Main project overview ⭐
├── QUICK_REFERENCE.md                  # Quick start guide ⭐
├── ALL_STOCKS_DOCUMENTATION.md         # 287-stock complete docs ⭐
├── PROJECT_STATUS.md                   # Current status report ⭐
├── DATA_FORMAT_SPECIFICATION.md        # CSV format guide
├── UNIVERSAL_LSTM_SUMMARY.md           # 2-stock historical reference
└── DOCUMENTATION_INDEX.md              # This file
```

### Source Code Documentation
```
src/
├── lstm_model/
│   ├── README.md                       # LSTM models overview
│   └── ADDING_STOCKS.md                # Adding stocks guide
```

### Archived Documentation
```
archive/
├── FINAL_COMPARISON.md                 # Historical model comparison
└── src/
    └── random_forest_model/
        ├── README.md
        ├── COMPARISON.md
        └── EXPERIMENTS.md
```

---

## 🎯 Use Case Guide

### I want to...

#### Make Predictions
→ Start with **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**
- See command examples
- Learn prediction options
- Understand output format

#### Understand the Model
→ Read **[ALL_STOCKS_DOCUMENTATION.md](ALL_STOCKS_DOCUMENTATION.md)**
- Model architecture
- Training process
- Performance metrics
- Technical specifications

#### Check Current Status
→ Review **[PROJECT_STATUS.md](PROJECT_STATUS.md)**
- Training progress
- Completed tasks
- Next actions
- Known limitations

#### Add New Stocks or Convert Data
→ Follow **[DATA_FORMAT_SPECIFICATION.md](DATA_FORMAT_SPECIFICATION.md)**
- Required CSV format
- Conversion examples
- Validation rules
- Common issues

#### Learn About Original Model
→ See **[UNIVERSAL_LSTM_SUMMARY.md](UNIVERSAL_LSTM_SUMMARY.md)**
- 2-stock proof-of-concept
- Initial performance
- Evolution to 287 stocks

#### Compare with Old Models
→ Check **[archive/FINAL_COMPARISON.md](archive/FINAL_COMPARISON.md)**
- LightGBM performance
- Random Forest results
- Why LSTM was chosen

---

## 📊 Documentation Priority

### Essential (Must Read)
1. ⭐⭐⭐ **README.md** - Start here!
2. ⭐⭐⭐ **QUICK_REFERENCE.md** - For immediate usage
3. ⭐⭐⭐ **PROJECT_STATUS.md** - Current state

### Important (For Development)
4. ⭐⭐ **ALL_STOCKS_DOCUMENTATION.md** - Deep technical details
5. ⭐⭐ **DATA_FORMAT_SPECIFICATION.md** - Data requirements

### Reference (As Needed)
6. ⭐ **UNIVERSAL_LSTM_SUMMARY.md** - Historical context
7. ⭐ **src/lstm_model/ADDING_STOCKS.md** - Extending the model
8. ⭐ **archive/FINAL_COMPARISON.md** - Model comparisons

---

## 🔧 Key Scripts Reference

### Data Processing
- `src/convert_all_stocks.py` - Convert NEPSE → OHLCV format
  - Input: `data/all_raw/*.csv`
  - Output: `data/processed/stock_daily_*.csv`

### Model Training
- `src/train_universal_lstm_all.py` - Train on all 287 stocks
  - Input: `data/processed/stock_daily_*.csv`
  - Output: `src/lstm_model/universal_lstm_all_stocks.h5`

### Prediction
- `src/lstm_model/universal_predict_all.py` - Make predictions
  - Options: `--stock`, `--batch`, `--top20`, `--all`, `--list`

### Evaluation
- `src/evaluate_universal_lstm_all.py` - Evaluate model
  - Output: Per-stock performance metrics

---

## 📈 Quick Stats

### Dataset
- **Raw files**: 415 NEPSE stocks
- **Converted**: 287 stocks (69%)
- **Total days**: 293,924 trading days
- **Date range**: 2010-2021

### Model
- **Architecture**: Universal LSTM with stock embeddings
- **Parameters**: 141,745 (553.69 KB)
- **Sequences**: 278,319 training samples
- **Stocks**: 282 (287 converted, 5 failed feature engineering)

### Performance (Expected)
- **R² Score**: 0.85-0.95 (based on 2-stock: 0.942)
- **RMSE**: 100-150 NPR (varies by stock)
- **MAPE**: 10-15% average

---

## 🚀 Quick Commands

```bash
# List all supported stocks
python src/lstm_model/universal_predict_all.py --list

# Predict single stock
python src/lstm_model/universal_predict_all.py --stock NABIL

# Predict top 20 stocks
python src/lstm_model/universal_predict_all.py --top20

# Evaluate model
python src/evaluate_universal_lstm_all.py

# Resume training
python src/train_universal_lstm_all.py
```

---

## 📝 Version History

### v1.0 (Current) - January 2025
- Scaled to 287 NEPSE stocks
- 278,319 training sequences
- 8 epochs completed (in training)
- Comprehensive documentation

### v0.2 - Previous
- 2-stock Universal LSTM (NABIL, SCB)
- R² = 0.942 achieved
- Proof-of-concept validated

### v0.1 - Initial
- Single-stock LSTM models
- Tree-based models (LightGBM, RF)
- Transaction-level predictions

---

## 🔗 Related Files

### Configuration
- Model checkpoints: `src/lstm_model/*.h5`
- Encoders: `src/lstm_model/*.pkl`
- Stock list: `src/lstm_model/supported_stocks.txt`

### Data
- Raw: `data/all_raw/*.csv`
- Processed: `data/processed/stock_daily_*.csv`
- Summary: `data/processed/conversion_summary.csv`

### Results
- `evaluation_results_top20.csv` (after evaluation)
- `batch_predictions.csv` (after batch prediction)

---

## 💡 Tips

1. **New users**: Start with README.md → QUICK_REFERENCE.md
2. **Developers**: Read ALL_STOCKS_DOCUMENTATION.md for technical details
3. **Quick predictions**: Use QUICK_REFERENCE.md command examples
4. **Data issues**: Consult DATA_FORMAT_SPECIFICATION.md
5. **Status check**: Review PROJECT_STATUS.md regularly

---

## 📞 Support

For issues or questions:
1. Check relevant documentation first
2. Review error messages carefully
3. Verify data format with DATA_FORMAT_SPECIFICATION.md
4. Check PROJECT_STATUS.md for known limitations

---

**Last Updated**: January 22, 2025  
**Documentation Version**: 1.0  
**Total Documents**: 9 core files + 4 archived
