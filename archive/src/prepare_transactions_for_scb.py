import pandas as pd
import os

# Get the directory paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")

# Read the SCB data
input_path = os.path.join(data_dir, "scb.csv")
df = pd.read_csv(input_path)

print("Original SCB data structure:")
print(df.head())
print(f"\nColumns: {df.columns.tolist()}")
print(f"Total rows: {len(df)}")

# Convert to OHLCV format for prediction models
df_converted = pd.DataFrame({
    'Date': pd.to_datetime(df['Date']),
    'Open': df['Min. Price'],  # Using Min as Open (approximation)
    'High': df['Max. Price'],
    'Low': df['Min. Price'],
    'Close': df['Close Price'],
    'Volume': df['Total Traded Shares']
})

# Sort by date in ascending order (oldest first)
df_converted = df_converted.sort_values('Date')

print("\n" + "="*60)
print("Converted to daily OHLCV format:")
print(df_converted.head())
print(f"\nDate range: {df_converted['Date'].min()} to {df_converted['Date'].max()}")
print(f"Total trading days: {len(df_converted)}")

# Save to stock_daily_scb.csv
output_path = os.path.join(data_dir, "stock_daily_scb.csv")
df_converted.to_csv(output_path, index=False)

print(f"\n✓ Saved to {output_path}")
print(f"  Format: Daily OHLCV data ready for SCB prediction models")
print(f"\nColumns in output file: {df_converted.columns.tolist()}")
