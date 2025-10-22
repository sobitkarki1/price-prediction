import pandas as pd
import os

# Get the directory of this script and construct absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")

# Read train and test data
train_df = pd.read_csv(os.path.join(data_dir, "NEPSE136_train.csv"), header=None)
test_df = pd.read_csv(os.path.join(data_dir, "NEPSE136_test.csv"), header=None)

# Combine train and test
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Assign column names that match what example-pipeline.py expects
combined_df.columns = [
    "Index",
    "Transaction ID", 
    "Stock Symbol",
    "Buyer ID",
    "Seller ID",
    "Quantity",
    "Price",
    "Total Amount"
]

# Select only the columns needed by the pipeline
output_df = combined_df[["Transaction ID", "Stock Symbol", "Quantity", "Price"]]

# Save to transactions.csv in data folder
output_path = os.path.join(data_dir, "transactions.csv")
output_df.to_csv(output_path, index=False)

print(f"Created transactions.csv with {len(output_df)} rows")
print(f"Columns: {list(output_df.columns)}")
print(f"\nFirst few rows:")
print(output_df.head())
print(f"\nData info:")
print(output_df.info())
