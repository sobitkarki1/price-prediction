# Adding New Stocks to Universal LSTM Model

This guide shows you how to add new stocks to the Universal LSTM prediction model.

## Prerequisites

Your stock data CSV file must have these columns:
- **Date**: Date in YYYY-MM-DD format
- **Open**: Opening price
- **High**: Highest price
- **Low**: Lowest price
- **Close**: Closing price
- **Volume**: Trading volume

## Step-by-Step Process

### Step 1: Prepare Your CSV File

Example format (`data/stock_daily_SYMBOL.csv`):
```csv
Date,Open,High,Low,Close,Volume
2010-04-15,150.00,152.50,148.00,151.00,50000
2010-04-16,151.50,153.00,150.00,152.00,45000
...
```

**Important**:
- At least 50+ days of data (more is better)
- No missing dates if possible
- Sorted by date chronologically
- Clean data (no zeros, no nulls in price columns)

### Step 2: Save to Data Folder

Save your file as:
```
data/stock_daily_<SYMBOL>.csv
```

Examples:
- `data/stock_daily_NABIL.csv` (already exists)
- `data/stock_daily_SCB.csv` (already exists)
- `data/stock_daily_NBL.csv` (your new stock)
- `data/stock_daily_NICA.csv` (your new stock)

**Naming Convention**: Use stock ticker symbol in uppercase

### Step 3: Update Universal LSTM Script

Edit `src/lstm_model/universal_lstm.py` to include your new stock:

Find this section (around line 36):
```python
# Load NABIL
nabil_path = os.path.join(data_dir, "stock_daily.csv")
if os.path.exists(nabil_path):
    nabil_df = pd.read_csv(nabil_path, parse_dates=['Date'])
    nabil_df['Stock_Symbol'] = 'NABIL'
    stocks_data.append(nabil_df)
    print(f"✓ Loaded NABIL: {len(nabil_df)} days")

# Load SCB
scb_path = os.path.join(data_dir, "stock_daily_scb.csv")
if os.path.exists(scb_path):
    scb_df = pd.read_csv(scb_path, parse_dates=['Date'])
    scb_df['Stock_Symbol'] = 'SCB'
    stocks_data.append(scb_df)
    print(f"✓ Loaded SCB: {len(scb_df)} days")
```

Add your new stock:
```python
# Load NBL (your new stock)
nbl_path = os.path.join(data_dir, "stock_daily_NBL.csv")
if os.path.exists(nbl_path):
    nbl_df = pd.read_csv(nbl_path, parse_dates=['Date'])
    nbl_df['Stock_Symbol'] = 'NBL'
    stocks_data.append(nbl_df)
    print(f"✓ Loaded NBL: {len(nbl_df)} days")
```

### Step 4: Update Universal Predict Script

Edit `src/lstm_model/universal_predict.py` to map your stock symbol:

Find this section (around line 70):
```python
# Mapping of stock symbols to file names
stock_files = {
    'NABIL': 'stock_daily.csv',
    'SCB': 'stock_daily_scb.csv'
}
```

Add your new stock:
```python
stock_files = {
    'NABIL': 'stock_daily.csv',
    'SCB': 'stock_daily_scb.csv',
    'NBL': 'stock_daily_NBL.csv',
    'NICA': 'stock_daily_NICA.csv'
}
```

### Step 5: Retrain the Model

Run the universal LSTM training script:
```bash
python src/lstm_model/universal_lstm.py
```

This will:
1. Load all stock data (including new ones)
2. Create stock embeddings for each symbol
3. Train on combined dataset
4. Save updated model: `universal_lstm_model.h5`
5. Save updated encoders: `stock_encoder.pkl`, `scaler_X.pkl`, `scaler_y.pkl`

**Expected Output**:
```
============================================================
Universal LSTM Model - Multi-Stock Prediction
============================================================

Loading stock data...
✓ Loaded NABIL: 2649 days
✓ Loaded SCB: 2660 days
✓ Loaded NBL: 1850 days
✓ Loaded NICA: 2100 days

Combined Dataset:
  Total records: 9259
  Stocks: ['NABIL', 'SCB', 'NBL', 'NICA']
  Date range: 2010-04-15 to 2021-12-29
```

### Step 6: Make Predictions

Now you can predict for your new stocks:

```bash
# Predict all stocks
python src/lstm_model/universal_predict.py

# Or use programmatically
python -c "from src.lstm_model.universal_predict import predict_stock; predict_stock('NBL')"
```

## Automated Approach (Recommended)

Instead of manually editing the script each time, you can make it auto-detect all CSV files:

### Option A: Auto-load all stock_daily_*.csv files

Replace the manual loading in `universal_lstm.py` with:

```python
import glob

# Auto-load all stock data files
print("Loading stock data...")
stocks_data = []

# Load NABIL (special case - different filename)
nabil_path = os.path.join(data_dir, "stock_daily.csv")
if os.path.exists(nabil_path):
    nabil_df = pd.read_csv(nabil_path, parse_dates=['Date'])
    nabil_df['Stock_Symbol'] = 'NABIL'
    stocks_data.append(nabil_df)
    print(f"✓ Loaded NABIL: {len(nabil_df)} days")

# Auto-load all stock_daily_*.csv files
pattern = os.path.join(data_dir, "stock_daily_*.csv")
for file_path in glob.glob(pattern):
    filename = os.path.basename(file_path)
    symbol = filename.replace('stock_daily_', '').replace('.csv', '').upper()
    
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['Stock_Symbol'] = symbol
    stocks_data.append(df)
    print(f"✓ Loaded {symbol}: {len(df)} days")
```

Now just drop CSV files in data folder and retrain!

## Example: Adding Nepal Bank Limited (NBL)

### 1. Get Data
Download NBL historical data and save as `data/stock_daily_NBL.csv`

### 2. Verify Format
```bash
head data/stock_daily_NBL.csv
```

Should show:
```
Date,Open,High,Low,Close,Volume
2015-01-01,250.00,255.00,248.00,252.00,35000
```

### 3. Update Scripts
Add NBL loading code to both files (as shown above)

### 4. Retrain
```bash
python src/lstm_model/universal_lstm.py
```

### 5. Predict
```bash
python src/lstm_model/universal_predict.py
```

You'll see:
```
Stock      Current      Predicted    Change          Signal
------------------------------------------------------------
NABIL      1450.00      1454.96       +0.34% 🟢 BUY
SCB        447.00       628.01       +40.50% 🟢 BUY
NBL        315.00       320.50        +1.75% 🟢 BUY
------------------------------------------------------------
```

## Tips for Best Results

### Data Quality
- ✅ **Minimum**: 100+ trading days
- ✅ **Recommended**: 500+ trading days (2 years)
- ✅ **Ideal**: 1000+ trading days (4 years)
- ✅ **Clean data**: No extreme outliers or errors
- ✅ **Complete**: Minimal missing dates

### Stock Selection
- ✅ **Similar sector**: Banking stocks work well together
- ✅ **Active trading**: High volume, liquid stocks
- ❌ **Avoid**: Penny stocks, newly listed stocks
- ❌ **Avoid**: Stocks with major corporate actions (splits, mergers)

### Training Tips
- **More stocks = better generalization** (but slower training)
- **Rebalance**: If one stock has 10x data, others may be underrepresented
- **Monitor per-stock performance** in training output
- **Retrain regularly**: Monthly or quarterly to capture new patterns

## Troubleshooting

### "Error: Data file not found"
- Check file name matches pattern: `stock_daily_<SYMBOL>.csv`
- Verify file is in `data/` folder
- Check file permissions

### "Need at least 30 days of data"
- Your CSV has too few rows
- Increase data history or remove that stock

### "Model performance degraded"
- New stock may be too volatile or different sector
- Try training without it
- Check if data has errors/outliers

### "Stock not in trained model"
- You added CSV but didn't retrain
- Run `universal_lstm.py` to retrain
- Encoders need updating

## Advanced: Batch Adding Multiple Stocks

If you have many stocks to add:

### 1. Prepare all CSVs
```
data/
  stock_daily_NBL.csv
  stock_daily_NICA.csv
  stock_daily_GBIME.csv
  stock_daily_NMB.csv
  stock_daily_SBI.csv
```

### 2. Use auto-loading (shown above)

### 3. Single retrain
```bash
python src/lstm_model/universal_lstm.py
```

All stocks will be loaded and trained in one go!

## Model Versioning

Consider backing up your model before retraining:

```bash
# Backup current model
cp src/lstm_model/universal_lstm_model.h5 src/lstm_model/backup/universal_lstm_model_v1.h5
cp src/lstm_model/stock_encoder.pkl src/lstm_model/backup/stock_encoder_v1.pkl

# Now retrain safely
python src/lstm_model/universal_lstm.py

# If new model is worse, restore backup
cp src/lstm_model/backup/universal_lstm_model_v1.h5 src/lstm_model/universal_lstm_model.h5
```

## Next Steps

After adding stocks:
1. ✅ Verify predictions look reasonable
2. ✅ Check per-stock R² scores in training output
3. ✅ Backtest on historical data
4. ✅ Compare with single-stock models if available
5. ✅ Monitor prediction errors over time

---

**Happy stock trading! 📈**

Questions? Check the main README or LSTM README for more details.
