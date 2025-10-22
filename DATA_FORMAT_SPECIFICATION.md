# Stock Data Format Specification

This document describes the exact format required for stock data files to be compatible with the Universal LSTM model.

## Required File Format

### File Naming Convention

Stock data files must follow this pattern:
```
data/stock_daily_<SYMBOL>.csv
```

**Examples:**
- `data/stock_daily.csv` → NABIL (legacy filename, special case)
- `data/stock_daily_SCB.csv` → Standard Chartered Bank
- `data/stock_daily_NBL.csv` → Nepal Bank Limited
- `data/stock_daily_NICA.csv` → NIC Asia Bank

**Rules:**
- Use uppercase for stock symbols
- No spaces in filename
- Must be `.csv` format
- Must be in the `data/` directory

---

## CSV Structure

### Required Columns (Exact Order)

The CSV file **MUST** have exactly these 6 columns in this order:

| Column # | Column Name | Data Type | Description | Example |
|----------|-------------|-----------|-------------|---------|
| 1 | **Date** | Date | Trading date in YYYY-MM-DD format | 2021-12-29 |
| 2 | **Open** | Float | Opening price in NPR | 1445.00 |
| 3 | **High** | Float | Highest price in NPR | 1455.00 |
| 4 | **Low** | Float | Lowest price in NPR | 1440.00 |
| 5 | **Close** | Float | Closing price in NPR | 1450.00 |
| 6 | **Volume** | Integer | Number of shares traded | 125000 |

### Header Row

**REQUIRED**: First row must be the header with exact column names (case-sensitive):
```csv
Date,Open,High,Low,Close,Volume
```

---

## Complete Example

### Valid CSV Format

**File**: `data/stock_daily_NABIL.csv`

```csv
Date,Open,High,Low,Close,Volume
2010-04-15,585.00,600.00,580.00,595.00,45000
2010-04-16,595.00,610.00,590.00,605.00,52000
2010-04-18,605.00,615.00,600.00,610.00,48000
2010-04-19,610.00,620.00,605.00,615.00,51000
2010-04-20,615.00,625.00,610.00,620.00,49000
2010-04-21,620.00,630.00,615.00,625.00,55000
2010-04-22,625.00,635.00,620.00,630.00,58000
```

---

## Data Requirements

### 1. Date Format

**Required Format**: `YYYY-MM-DD`

✅ **Valid Examples:**
- `2021-12-29`
- `2010-04-15`
- `2020-01-01`

❌ **Invalid Examples:**
- `29/12/2021` (DD/MM/YYYY)
- `12-29-2021` (MM-DD-YYYY)
- `2021/12/29` (wrong separator)
- `29-Dec-2021` (month name)

**Excel Users:** Format cells as Custom: `yyyy-mm-dd`

### 2. Price Columns (Open, High, Low, Close)

**Format**: Decimal number with up to 2 decimal places

**Rules:**
- Must be positive (> 0)
- No currency symbols (₨, NPR, Rs.)
- Use dot (.) for decimals, not comma
- Can be integer or float

✅ **Valid Examples:**
- `1450.00`
- `1450.5`
- `1450` (no decimals)
- `150.75`

❌ **Invalid Examples:**
- `₨1450.00` (currency symbol)
- `1,450.00` (comma separator)
- `1450.000` (more than 2 decimals is okay but unnecessary)
- `0` or negative values

**Logical Constraints:**
- `Low ≤ Open ≤ High`
- `Low ≤ Close ≤ High`
- All prices > 0

### 3. Volume

**Format**: Integer (whole number)

**Rules:**
- Must be positive (> 0)
- No decimal points
- No commas or separators

✅ **Valid Examples:**
- `125000`
- `50000`
- `1500000`

❌ **Invalid Examples:**
- `125,000` (comma separator)
- `125000.00` (decimal point)
- `0` (zero volume)
- Negative values

---

## Data Quality Requirements

### Minimum Data Size

| Requirement | Minimum | Recommended | Ideal |
|-------------|---------|-------------|-------|
| **Trading Days** | 50 | 500 (2 years) | 1000+ (4 years) |
| **For Training** | 100 | 1000 | 2000+ |

**Why?**
- Model uses 30-day lookback window
- Needs enough data for train/test split (80/20)
- More data = better predictions

### Missing Data

**Allowed:**
- ✅ Missing weekend/holiday dates (non-trading days)
- ✅ Gaps due to market holidays

**Not Allowed:**
- ❌ Missing values in any column (blank cells)
- ❌ NULL, NaN, or empty strings
- ❌ Zero values in price columns

### Duplicates

**Rules:**
- Each date should appear only once
- If duplicates exist, model keeps the **first occurrence**
- Better to clean duplicates before uploading

### Sorting

**Required:** Data must be sorted chronologically (oldest to newest)

✅ **Correct:**
```csv
2010-04-15,...
2010-04-16,...
2010-04-18,...
```

❌ **Incorrect:**
```csv
2010-04-18,...
2010-04-15,...
2010-04-16,...
```

---

## Common Data Issues & Solutions

### Issue 1: Wrong Date Format

**Problem:**
```csv
Date,Open,High,Low,Close,Volume
29/12/2021,1450.00,1455.00,1440.00,1450.00,125000
```

**Solution (Excel):**
1. Select Date column
2. Format Cells → Custom
3. Type: `yyyy-mm-dd`
4. Click OK

**Solution (Python):**
```python
import pandas as pd

df = pd.read_csv('input.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df.to_csv('stock_daily_SYMBOL.csv', index=False)
```

### Issue 2: Currency Symbols in Prices

**Problem:**
```csv
Date,Open,High,Low,Close,Volume
2021-12-29,₨1450.00,₨1455.00,₨1440.00,₨1450.00,125000
```

**Solution (Python):**
```python
import pandas as pd

df = pd.read_csv('input.csv')
# Remove currency symbols
for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = df[col].str.replace('₨', '').str.replace('Rs.', '').str.replace(',', '')
    df[col] = pd.to_numeric(df[col])
df.to_csv('stock_daily_SYMBOL.csv', index=False)
```

### Issue 3: Comma Thousands Separator

**Problem:**
```csv
Date,Open,High,Low,Close,Volume
2021-12-29,1,450.00,1,455.00,1,440.00,1,450.00,125,000
```

**Solution (Python):**
```python
import pandas as pd

df = pd.read_csv('input.csv')
# Remove commas from all numeric columns
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if df[col].dtype == 'object':  # If read as string
        df[col] = df[col].str.replace(',', '')
        df[col] = pd.to_numeric(df[col])
df.to_csv('stock_daily_SYMBOL.csv', index=False)
```

### Issue 4: Different Column Names

**Problem:**
```csv
DATE,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,VOL
2021-12-29,1450.00,1455.00,1440.00,1450.00,125000
```

**Solution (Python):**
```python
import pandas as pd

df = pd.read_csv('input.csv')
# Rename columns to match required format
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df.to_csv('stock_daily_SYMBOL.csv', index=False)
```

### Issue 5: Extra Columns

**Problem:**
```csv
Date,Open,High,Low,Close,Volume,Adj_Close,Ticker
2021-12-29,1450.00,1455.00,1440.00,1450.00,125000,1450.00,NABIL
```

**Solution (Python):**
```python
import pandas as pd

df = pd.read_csv('input.csv')
# Keep only required columns
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
df.to_csv('stock_daily_SYMBOL.csv', index=False)
```

---

## Validation Checklist

Before using your CSV file with the model, verify:

- [ ] Filename follows pattern: `stock_daily_<SYMBOL>.csv`
- [ ] File is in `data/` directory
- [ ] Header row exists with exact column names
- [ ] 6 columns in exact order: Date, Open, High, Low, Close, Volume
- [ ] Date format is YYYY-MM-DD
- [ ] All prices are positive numbers without currency symbols
- [ ] Volume is integer without commas
- [ ] No missing values (blank cells)
- [ ] Data sorted chronologically (oldest first)
- [ ] At least 50 rows of data (preferably 500+)
- [ ] No duplicate dates
- [ ] Logical constraints satisfied (Low ≤ Open/Close ≤ High)

---

## Conversion Script Template

Use this Python script to convert your data to the required format:

```python
import pandas as pd
import sys

def convert_to_model_format(input_file, output_file, stock_symbol):
    """
    Convert stock data to Universal LSTM model format
    
    Args:
        input_file: Path to your CSV file
        output_file: Path to save converted file
        stock_symbol: Stock ticker symbol (e.g., 'NABIL', 'SCB')
    """
    
    # Read CSV
    df = pd.read_csv(input_file)
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    # Example: Adjust column mapping based on your file
    # Uncomment and modify as needed:
    
    # If different column names:
    # column_mapping = {
    #     'DATE': 'Date',
    #     'OPEN_PRICE': 'Open',
    #     'HIGH_PRICE': 'High',
    #     'LOW_PRICE': 'Low',
    #     'CLOSE_PRICE': 'Close',
    #     'VOL': 'Volume'
    # }
    # df = df.rename(columns=column_mapping)
    
    # Keep only required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols]
    
    # Convert date to proper format
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    # Clean numeric columns (remove currency symbols, commas)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('₨', '').str.replace('Rs.', '')
            df[col] = df[col].str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Remove rows with zero or negative prices
    df = df[(df['Open'] > 0) & (df['High'] > 0) & 
            (df['Low'] > 0) & (df['Close'] > 0) & (df['Volume'] > 0)]
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Remove duplicates (keep first)
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    # Validate logical constraints
    df = df[(df['Low'] <= df['Open']) & (df['Open'] <= df['High'])]
    df = df[(df['Low'] <= df['Close']) & (df['Close'] <= df['High'])]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Conversion successful!")
    print(f"Output file: {output_file}")
    print(f"Final shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Trading days: {len(df)}")
    
    # Show sample
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return df

# Example usage:
if __name__ == "__main__":
    # Modify these paths for your data
    input_file = "data/your_stock_data.csv"
    stock_symbol = "NBL"  # Change to your stock symbol
    output_file = f"data/stock_daily_{stock_symbol}.csv"
    
    convert_to_model_format(input_file, output_file, stock_symbol)
```

**How to use:**
```bash
# 1. Edit the script with your file paths
# 2. Run the conversion
python convert_stock_data.py

# 3. Verify output
head data/stock_daily_NBL.csv
```

---

## Real-World Examples

### Example 1: NABIL Bank (Correct Format)

**File**: `data/stock_daily.csv` (legacy filename for NABIL)

```csv
Date,Open,High,Low,Close,Volume
2010-04-15,585.00,600.00,580.00,595.00,45123
2010-04-16,595.00,610.00,590.00,605.00,52340
2010-04-18,605.00,615.00,600.00,610.00,48950
```

✅ All requirements met

### Example 2: SCB Bank (Correct Format)

**File**: `data/stock_daily_SCB.csv`

```csv
Date,Open,High,Low,Close,Volume
2010-04-15,220.00,225.00,218.00,223.00,125000
2010-04-16,223.00,228.00,221.00,226.00,130000
2010-04-18,226.00,230.00,224.00,228.00,128000
```

✅ All requirements met

---

## Testing Your Data

After conversion, test your CSV file:

```python
import pandas as pd

# Load the file
df = pd.read_csv('data/stock_daily_SYMBOL.csv', parse_dates=['Date'])

# Run validations
print("Validation Report:")
print(f"✓ Shape: {df.shape}")
print(f"✓ Columns: {df.columns.tolist()}")
print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"✓ Trading days: {len(df)}")
print(f"✓ Missing values: {df.isnull().sum().sum()}")
print(f"✓ Duplicates: {df.duplicated(subset=['Date']).sum()}")
print(f"✓ Zero prices: {(df[['Open', 'High', 'Low', 'Close']] == 0).sum().sum()}")
print(f"✓ Negative prices: {(df[['Open', 'High', 'Low', 'Close']] < 0).sum().sum()}")

# Check logical constraints
valid_ohlc = ((df['Low'] <= df['Open']) & (df['Open'] <= df['High']) & 
              (df['Low'] <= df['Close']) & (df['Close'] <= df['High']))
print(f"✓ Valid OHLC: {valid_ohlc.sum()} / {len(df)}")

print("\nSample data:")
print(df.head())
```

Expected output:
```
Validation Report:
✓ Shape: (2649, 6)
✓ Columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
✓ Date range: 2010-04-15 to 2021-12-29
✓ Trading days: 2649
✓ Missing values: 0
✓ Duplicates: 0
✓ Zero prices: 0
✓ Negative prices: 0
✓ Valid OHLC: 2649 / 2649

Sample data:
        Date    Open    High     Low   Close  Volume
0 2010-04-15  585.00  600.00  580.00  595.00   45123
1 2010-04-16  595.00  610.00  590.00  605.00   52340
...
```

---

## Quick Reference

### Minimum Valid CSV

```csv
Date,Open,High,Low,Close,Volume
2020-01-01,100.00,105.00,98.00,102.00,50000
2020-01-02,102.00,108.00,101.00,106.00,55000
```

### Template for New Stock

1. Copy this template:
```csv
Date,Open,High,Low,Close,Volume
```

2. Add your data rows in YYYY-MM-DD format
3. Save as `data/stock_daily_<SYMBOL>.csv`
4. Run validation script
5. Train model with `python src/lstm_model/universal_lstm.py`

---

## Need Help?

**Common Questions:**
- **"My dates are in different format"** → See Issue 1 above
- **"I have extra columns"** → See Issue 5 above  
- **"Prices have currency symbols"** → See Issue 2 above
- **"Numbers have commas"** → See Issue 3 above
- **"Column names are different"** → See Issue 4 above

**Still stuck?**
1. Check your CSV with a text editor (not Excel)
2. Run the validation script
3. Use the conversion template
4. Verify with testing code

---

**Last Updated**: October 22, 2025  
**Model Compatibility**: Universal LSTM v1.0
