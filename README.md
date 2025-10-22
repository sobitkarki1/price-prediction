# NEPSE Stock Price Prediction

## Dataset Overview

This project contains trading data from the Nepal Stock Exchange (NEPSE), focusing on stock ticker **NBB**. The dataset includes 177,209 transaction records.

## Data Structure

Each row represents a single trade transaction with 9 columns:

1. **Index** - Sequential record number
2. **Transaction ID** - Unique identifier for the trade
3. **Ticker** - Stock symbol (NBB)
4. **Buyer ID** - Identifier for the buying party
5. **Seller ID** - Identifier for the selling party
6. **Quantity** - Number of shares traded
7. **Price** - Price per share
8. **Total Amount** - Total transaction value (Quantity × Price)
9. **Reference ID** - Additional transaction reference

## Files

- `NEPSE136.csv` - Complete dataset (177,209 records)
- `NEPSE136_part1.csv` - First 24,128 records
- `NEPSE136_part2.csv` - Remaining 153,081 records
- `NEPSE136_train.csv` - Training set (122,464 records, 80%)
- `NEPSE136_test.csv` - Test set (30,617 records, 20%)

## Usage

The data is split for machine learning purposes with an 80/20 train-test split, suitable for time-series prediction or trading pattern analysis.
