"""
Convert all raw stock data to Universal LSTM compatible format

This script converts stock data from NEPSE format to OHLCV format:
From: S.N., Date, Total Transactions, Total Traded Shares, Total Traded Amount, Max. Price, Min. Price, Close Price
To: Date, Open, High, Low, Close, Volume
"""

import pandas as pd
import os
import glob
from pathlib import Path

# Paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"
raw_dir = data_dir / "all_raw"
processed_dir = data_dir / "processed"

# Create processed directory
processed_dir.mkdir(exist_ok=True)

print(f"{'='*80}")
print(f"NEPSE Stock Data Converter - Raw to OHLCV Format")
print(f"{'='*80}\n")

print(f"Source: {raw_dir}")
print(f"Destination: {processed_dir}\n")

# Get all CSV files in raw directory
raw_files = sorted(glob.glob(str(raw_dir / "*_2000-01-01_2021-12-31.csv")))

print(f"Found {len(raw_files)} stock files\n")

conversion_summary = []
failed_conversions = []

for idx, file_path in enumerate(raw_files, 1):
    filename = os.path.basename(file_path)
    # Extract symbol (e.g., "NABIL_2000-01-01_2021-12-31.csv" -> "NABIL")
    symbol = filename.split('_')[0]
    
    print(f"[{idx}/{len(raw_files)}] Processing {symbol}...", end=" ")
    
    try:
        # Read raw data
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_cols = ['Date', 'Max. Price', 'Min. Price', 'Close Price', 'Total Traded Shares']
        
        if not all(col in df.columns for col in required_cols):
            print(f"❌ Missing required columns")
            failed_conversions.append({'symbol': symbol, 'reason': 'Missing columns'})
            continue
        
        # Create OHLCV dataframe
        # Note: NEPSE data doesn't have Open price, so we'll use previous Close as Open
        df_ohlcv = pd.DataFrame()
        
        # Date (already in correct format YYYY-MM-DD)
        df_ohlcv['Date'] = pd.to_datetime(df['Date'])
        
        # Use Close as Open (or previous day's close if available)
        df_ohlcv['Open'] = df['Close Price'].shift(1).fillna(df['Close Price'])
        
        # High = Max Price
        df_ohlcv['High'] = df['Max. Price']
        
        # Low = Min Price
        df_ohlcv['Low'] = df['Min. Price']
        
        # Close = Close Price
        df_ohlcv['Close'] = df['Close Price']
        
        # Volume = Total Traded Shares
        df_ohlcv['Volume'] = df['Total Traded Shares']
        
        # Clean data
        df_ohlcv = df_ohlcv.dropna()
        df_ohlcv = df_ohlcv[df_ohlcv['Close'] > 0]  # Remove zero prices
        df_ohlcv = df_ohlcv[df_ohlcv['Volume'] > 0]  # Remove zero volume
        
        # Sort chronologically (oldest first)
        df_ohlcv = df_ohlcv.sort_values('Date')
        
        # Remove duplicates
        df_ohlcv = df_ohlcv.drop_duplicates(subset=['Date'], keep='first')
        
        # Format date as string
        df_ohlcv['Date'] = df_ohlcv['Date'].dt.strftime('%Y-%m-%d')
        
        # Validate minimum data requirement
        if len(df_ohlcv) < 50:
            print(f"⚠️  Only {len(df_ohlcv)} days (minimum 50 required)")
            failed_conversions.append({'symbol': symbol, 'reason': f'Insufficient data ({len(df_ohlcv)} days)'})
            continue
        
        # Save to processed directory
        output_file = processed_dir / f"stock_daily_{symbol}.csv"
        df_ohlcv.to_csv(output_file, index=False)
        
        # Summary
        date_range = f"{df_ohlcv['Date'].iloc[0]} to {df_ohlcv['Date'].iloc[-1]}"
        print(f"✅ {len(df_ohlcv):4d} days ({date_range})")
        
        conversion_summary.append({
            'symbol': symbol,
            'days': len(df_ohlcv),
            'start_date': df_ohlcv['Date'].iloc[0],
            'end_date': df_ohlcv['Date'].iloc[-1],
            'file': output_file.name
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        failed_conversions.append({'symbol': symbol, 'reason': str(e)})

# Print summary
print(f"\n{'='*80}")
print(f"Conversion Summary")
print(f"{'='*80}\n")

print(f"Total files processed: {len(raw_files)}")
print(f"Successfully converted: {len(conversion_summary)}")
print(f"Failed conversions: {len(failed_conversions)}\n")

if conversion_summary:
    # Sort by number of days descending
    conversion_summary.sort(key=lambda x: x['days'], reverse=True)
    
    print(f"Top 20 stocks by data size:")
    print(f"{'-'*80}")
    print(f"{'Rank':<6} {'Symbol':<12} {'Days':<8} {'Date Range':<30} {'Status'}")
    print(f"{'-'*80}")
    
    for idx, item in enumerate(conversion_summary[:20], 1):
        date_range = f"{item['start_date']} to {item['end_date']}"
        status = "✅ Excellent" if item['days'] >= 1000 else "✅ Good" if item['days'] >= 500 else "⚠️  Limited"
        print(f"{idx:<6} {item['symbol']:<12} {item['days']:<8} {date_range:<30} {status}")
    
    print(f"\n... and {len(conversion_summary) - 20} more stocks\n")
    
    # Statistics
    total_days = sum(item['days'] for item in conversion_summary)
    avg_days = total_days / len(conversion_summary)
    min_days = min(item['days'] for item in conversion_summary)
    max_days = max(item['days'] for item in conversion_summary)
    
    print(f"Data Statistics:")
    print(f"  Total trading days across all stocks: {total_days:,}")
    print(f"  Average days per stock: {avg_days:.0f}")
    print(f"  Min days: {min_days} ({[s['symbol'] for s in conversion_summary if s['days'] == min_days][0]})")
    print(f"  Max days: {max_days} ({[s['symbol'] for s in conversion_summary if s['days'] == max_days][0]})")
    
    # Stock categories
    excellent = sum(1 for s in conversion_summary if s['days'] >= 1000)
    good = sum(1 for s in conversion_summary if 500 <= s['days'] < 1000)
    limited = sum(1 for s in conversion_summary if 50 <= s['days'] < 500)
    
    print(f"\nStock Categories:")
    print(f"  Excellent (1000+ days): {excellent} stocks")
    print(f"  Good (500-999 days): {good} stocks")
    print(f"  Limited (50-499 days): {limited} stocks")

if failed_conversions:
    print(f"\n{'='*80}")
    print(f"Failed Conversions ({len(failed_conversions)} stocks)")
    print(f"{'='*80}\n")
    
    for item in failed_conversions[:10]:  # Show first 10
        print(f"  {item['symbol']:<12} - {item['reason']}")
    
    if len(failed_conversions) > 10:
        print(f"  ... and {len(failed_conversions) - 10} more")

print(f"\n{'='*80}")
print(f"✅ Conversion Complete!")
print(f"{'='*80}\n")

print(f"Converted files saved to: {processed_dir}")
print(f"Ready for Universal LSTM training!\n")

# Save summary to CSV for reference
summary_df = pd.DataFrame(conversion_summary)
summary_file = processed_dir / "conversion_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary report saved to: {summary_file}\n")
