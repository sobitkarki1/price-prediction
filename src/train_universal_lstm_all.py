"""
Universal LSTM Model - Multi-Stock Price Prediction (All NEPSE Stocks)
Automatically loads all stocks from processed/ directory
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import glob
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print(f"{'='*80}")
print(f"Universal LSTM Model - All NEPSE Stocks Training")
print(f"{'='*80}\n")

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")
processed_dir = os.path.join(data_dir, "processed")

# Auto-load all stock data from processed directory
print("Loading stock data from processed directory...")
print(f"Source: {processed_dir}\n")

stocks_data = []
stock_files = sorted(glob.glob(os.path.join(processed_dir, "stock_daily_*.csv")))

print(f"Found {len(stock_files)} processed stock files")
print(f"{'-'*80}\n")

# Load all stocks
for idx, file_path in enumerate(stock_files, 1):
    filename = os.path.basename(file_path)
    # Extract symbol from filename (e.g., "stock_daily_NABIL.csv" -> "NABIL")
    symbol = filename.replace('stock_daily_', '').replace('.csv', '')
    
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df['Stock_Symbol'] = symbol
        stocks_data.append(df)
        
        if idx <= 20:  # Show first 20
            print(f"  [{idx:3d}] {symbol:<12} - {len(df):4d} days ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"  [ERR] {symbol:<12} - Failed to load: {str(e)}")

if len(stocks_data) > 20:
    print(f"  ... and {len(stocks_data) - 20} more stocks")

# Combine all stocks
df_all = pd.concat(stocks_data, ignore_index=True)
df_all = df_all.sort_values(['Stock_Symbol', 'Date']).reset_index(drop=True)

print(f"\n{'='*80}")
print(f"Combined Dataset Summary")
print(f"{'='*80}\n")
print(f"  Total records: {len(df_all):,}")
print(f"  Number of stocks: {df_all['Stock_Symbol'].nunique()}")
print(f"  Date range: {df_all['Date'].min()} to {df_all['Date'].max()}")
print(f"  Unique trading days: {df_all['Date'].nunique()}")

# Encode stock symbols
label_encoder = LabelEncoder()
df_all['Stock_ID'] = label_encoder.fit_transform(df_all['Stock_Symbol'])

print(f"\n{'='*80}")
print(f"Stock Encoding (First 20)")
print(f"{'='*80}\n")
for idx, symbol in enumerate(sorted(df_all['Stock_Symbol'].unique())[:20]):
    stock_id = df_all[df_all['Stock_Symbol'] == symbol]['Stock_ID'].iloc[0]
    count = len(df_all[df_all['Stock_Symbol'] == symbol])
    print(f"  {symbol:<12} → ID {stock_id:3d} ({count:4d} records)")

if df_all['Stock_Symbol'].nunique() > 20:
    print(f"  ... and {df_all['Stock_Symbol'].nunique() - 20} more stocks")

# Feature Engineering (per stock)
print(f"\n{'='*80}")
print(f"Creating Features and Sequences")
print(f"{'='*80}\n")

all_features = []
all_targets = []
all_stock_ids = []
stock_sequence_counts = {}

for stock_symbol in df_all['Stock_Symbol'].unique():
    stock_df = df_all[df_all['Stock_Symbol'] == stock_symbol].copy()
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates and invalid data
    stock_df = stock_df.drop_duplicates(subset=['Date'], keep='first')
    stock_df = stock_df[stock_df['Close'] > 0]
    
    # Skip stocks with insufficient data
    if len(stock_df) < 60:  # Need at least 60 days for 30-day lookback + forecast
        continue
    
    # Technical indicators
    stock_df['return'] = stock_df['Close'].pct_change()
    stock_df['volume_change'] = stock_df['Volume'].pct_change()
    
    for n in [5, 10, 20]:
        stock_df[f'sma_{n}'] = stock_df['Close'].rolling(n).mean()
        stock_df[f'std_{n}'] = stock_df['Close'].rolling(n).std()
    
    # RSI
    delta = stock_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    stock_df['rsi'] = 100 - (100 / (1 + rs))
    
    stock_df = stock_df.dropna()
    
    if len(stock_df) < 40:  # After feature engineering, need minimum data
        continue
    
    # Select features
    tech_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 
                     'volume_change', 'sma_5', 'sma_10', 'sma_20', 
                     'std_5', 'std_10', 'std_20', 'rsi']
    
    stock_data = stock_df[tech_features].values
    stock_target = stock_df['Close'].values
    stock_id = stock_df['Stock_ID'].iloc[0]
    
    # Create sequences
    lookback = 30
    forecast = 5
    
    sequences_created = 0
    for i in range(lookback, len(stock_data) - forecast):
        all_features.append(stock_data[i-lookback:i])
        all_targets.append(stock_target[i + forecast])
        all_stock_ids.append(stock_id)
        sequences_created += 1
    
    stock_sequence_counts[stock_symbol] = sequences_created

# Convert to arrays
X_features = np.array(all_features)
y_target = np.array(all_targets)
X_stock_ids = np.array(all_stock_ids)

print(f"Feature Engineering Complete:")
print(f"  Total sequences created: {len(X_features):,}")
print(f"  Stocks with sequences: {len(stock_sequence_counts)}")
print(f"  Feature shape: {X_features.shape}")
print(f"  Target shape: {y_target.shape}")
print(f"  Stock IDs shape: {X_stock_ids.shape}")

# Top stocks by sequence count
print(f"\nTop 10 stocks by sequence count:")
top_stocks = sorted(stock_sequence_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for rank, (symbol, count) in enumerate(top_stocks, 1):
    print(f"  {rank:2d}. {symbol:<12} - {count:5d} sequences")

# Normalize features
print(f"\n{'='*80}")
print(f"Normalizing Features")
print(f"{'='*80}\n")

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Reshape for scaling
n_samples, n_timesteps, n_features = X_features.shape
X_features_reshaped = X_features.reshape(-1, n_features)
X_features_scaled = scaler_X.fit_transform(X_features_reshaped)
X_features_scaled = X_features_scaled.reshape(n_samples, n_timesteps, n_features)

y_target_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1))

# Train-test split (80-20)
train_size = int(len(X_features_scaled) * 0.8)

X_train = X_features_scaled[:train_size]
X_train_stock_ids = X_stock_ids[:train_size]
y_train = y_target_scaled[:train_size]

X_test = X_features_scaled[train_size:]
X_test_stock_ids = X_stock_ids[train_size:]
y_test = y_target_scaled[train_size:]

print(f"Train-Test Split:")
print(f"  Training samples: {len(X_train):,} ({len(X_train)/len(X_features_scaled)*100:.1f}%)")
print(f"  Testing samples: {len(X_test):,} ({len(X_test)/len(X_features_scaled)*100:.1f}%)")

# Build Universal LSTM Model
print(f"\n{'='*80}")
print(f"Building Universal LSTM Architecture")
print(f"{'='*80}\n")

# Input layers
sequence_input = Input(shape=(n_timesteps, n_features), name='sequence_input')
stock_id_input = Input(shape=(1,), name='stock_id_input')

# Stock embedding (learn stock-specific features)
n_stocks = len(df_all['Stock_Symbol'].unique())
embedding_dim = 16  # Increased from 8 for more stocks
stock_embedding = Embedding(input_dim=n_stocks, output_dim=embedding_dim, 
                           name='stock_embedding')(stock_id_input)
stock_embedding = tf.keras.layers.Flatten()(stock_embedding)

# LSTM branch for sequences
lstm_out = LSTM(128, return_sequences=True)(sequence_input)
lstm_out = Dropout(0.3)(lstm_out)
lstm_out = LSTM(64, return_sequences=True)(lstm_out)
lstm_out = Dropout(0.3)(lstm_out)
lstm_out = LSTM(32, return_sequences=False)(lstm_out)
lstm_out = Dropout(0.3)(lstm_out)

# Combine LSTM output with stock embedding
combined = Concatenate()([lstm_out, stock_embedding])

# Dense layers
dense = Dense(32, activation='relu')(combined)
dense = Dropout(0.3)(dense)
dense = Dense(16, activation='relu')(dense)
output = Dense(1, name='output')(dense)

# Create model
model = Model(inputs=[sequence_input, stock_id_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(model.summary())

# Callbacks
model_path = os.path.join(os.path.dirname(__file__), "lstm_model", "universal_lstm_all_stocks.h5")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)

# Train model
print(f"\n{'='*80}")
print(f"Training Universal LSTM Model")
print(f"{'='*80}\n")

history = model.fit(
    [X_train, X_train_stock_ids],
    y_train,
    epochs=150,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print(f"\n✅ Training completed!")
print(f"Best epoch: {len(history.history['loss']) - early_stop.patience}")
print(f"Best validation loss: {min(history.history['val_loss']):.6f}")

# Evaluate
print(f"\n{'='*80}")
print(f"Evaluation Results")
print(f"{'='*80}\n")

# Overall performance
test_pred_scaled = model.predict([X_test, X_test_stock_ids], verbose=0)
test_pred = scaler_y.inverse_transform(test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

overall_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
overall_mae = mean_absolute_error(y_test_actual, test_pred)
overall_r2 = r2_score(y_test_actual, test_pred)
overall_mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100

print(f"Overall Test Performance (All {n_stocks} Stocks):")
print(f"  RMSE: {overall_rmse:.2f} NPR")
print(f"  MAE: {overall_mae:.2f} NPR")
print(f"  MAPE: {overall_mape:.2f}%")
print(f"  R² Score: {overall_r2:.4f}")

# Per-stock performance (top 10 stocks)
print(f"\n{'-'*80}")
print(f"Top 10 Stocks Performance:")
print(f"{'-'*80}\n")

stock_performance = []
for stock_id in np.unique(X_test_stock_ids):
    stock_symbol = label_encoder.inverse_transform([stock_id])[0]
    stock_mask = X_test_stock_ids == stock_id
    
    if np.sum(stock_mask) > 0:
        y_stock_actual = y_test_actual[stock_mask]
        y_stock_pred = test_pred[stock_mask]
        
        stock_rmse = np.sqrt(mean_squared_error(y_stock_actual, y_stock_pred))
        stock_mae = mean_absolute_error(y_stock_actual, y_stock_pred)
        stock_r2 = r2_score(y_stock_actual, y_stock_pred)
        stock_mape = np.mean(np.abs((y_stock_actual - y_stock_pred) / y_stock_actual)) * 100
        
        stock_performance.append({
            'symbol': stock_symbol,
            'samples': np.sum(stock_mask),
            'r2': stock_r2,
            'rmse': stock_rmse,
            'mae': stock_mae,
            'mape': stock_mape
        })

# Sort by R² score
stock_performance.sort(key=lambda x: x['r2'], reverse=True)

for idx, perf in enumerate(stock_performance[:10], 1):
    print(f"{idx:2d}. {perf['symbol']:<12} (n={perf['samples']:4d})")
    print(f"    R²: {perf['r2']:.4f} | RMSE: {perf['rmse']:7.2f} | MAE: {perf['mae']:7.2f} | MAPE: {perf['mape']:5.2f}%")

print(f"\n{'='*80}")
print(f"Universal LSTM Model Training Complete!")
print(f"{'='*80}\n")

print(f"✅ Model saved to: {model_path}")
print(f"✅ Trained on {n_stocks} NEPSE stocks")
print(f"✅ Total training sequences: {len(X_train):,}")
print(f"✅ Ready for predictions!\n")

# Save encoders and scalers
import pickle

encoder_path = os.path.join(os.path.dirname(__file__), "lstm_model", "stock_encoder_all.pkl")
scaler_x_path = os.path.join(os.path.dirname(__file__), "lstm_model", "scaler_X_all.pkl")
scaler_y_path = os.path.join(os.path.dirname(__file__), "lstm_model", "scaler_y_all.pkl")

with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
with open(scaler_x_path, 'wb') as f:
    pickle.dump(scaler_X, f)
with open(scaler_y_path, 'wb') as f:
    pickle.dump(scaler_y, f)

print(f"✅ Saved encoders and scalers for future predictions")
print(f"✅ Stock encoder includes all {n_stocks} symbols\n")

# Save stock list
stock_list_path = os.path.join(os.path.dirname(__file__), "lstm_model", "supported_stocks.txt")
with open(stock_list_path, 'w') as f:
    for symbol in sorted(label_encoder.classes_):
        f.write(f"{symbol}\n")

print(f"✅ Supported stocks list saved to: {stock_list_path}\n")
