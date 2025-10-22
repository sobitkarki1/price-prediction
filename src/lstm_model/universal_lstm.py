"""
Universal LSTM Model - Multi-Stock Price Prediction
Uses stock symbol as a categorical feature to predict any stock
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print(f"{'='*60}")
print(f"Universal LSTM Model - Multi-Stock Prediction")
print(f"{'='*60}\n")

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")

# Load all stock data
print("Loading stock data...")

stocks_data = []

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

# Combine all stocks
df_all = pd.concat(stocks_data, ignore_index=True)
df_all = df_all.sort_values(['Stock_Symbol', 'Date']).reset_index(drop=True)

print(f"\nCombined Dataset:")
print(f"  Total records: {len(df_all)}")
print(f"  Stocks: {df_all['Stock_Symbol'].unique().tolist()}")
print(f"  Date range: {df_all['Date'].min()} to {df_all['Date'].max()}\n")

# Encode stock symbols
label_encoder = LabelEncoder()
df_all['Stock_ID'] = label_encoder.fit_transform(df_all['Stock_Symbol'])

print(f"Stock Encoding:")
for symbol in df_all['Stock_Symbol'].unique():
    stock_id = df_all[df_all['Stock_Symbol'] == symbol]['Stock_ID'].iloc[0]
    count = len(df_all[df_all['Stock_Symbol'] == symbol])
    print(f"  {symbol} → ID {stock_id} ({count} records)")
print()

# Feature Engineering (per stock)
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
all_features = []
all_targets = []
all_stock_ids = []

print("Creating features and sequences...")

for stock_symbol in df_all['Stock_Symbol'].unique():
    stock_df = df_all[df_all['Stock_Symbol'] == stock_symbol].copy()
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates and invalid data
    stock_df = stock_df.drop_duplicates(subset=['Date'], keep='first')
    stock_df = stock_df[stock_df['Close'] > 0]
    
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
    
    for i in range(lookback, len(stock_data) - forecast):
        all_features.append(stock_data[i-lookback:i])
        all_targets.append(stock_target[i + forecast])
        all_stock_ids.append(stock_id)
    
    print(f"  {stock_symbol}: Created {len(all_features) - (0 if not all_features else len([f for f in all_features if f is not None]))} sequences")

# Convert to arrays
X_features = np.array(all_features)
y_target = np.array(all_targets)
X_stock_ids = np.array(all_stock_ids)

print(f"\nDataset Summary:")
print(f"  Feature sequences: {X_features.shape}")
print(f"  Stock IDs: {X_stock_ids.shape}")
print(f"  Targets: {y_target.shape}")
print(f"  Lookback: 30 days")
print(f"  Forecast: 5 days ahead\n")

# Normalize features
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
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}\n")

# Build Universal LSTM Model with Stock Embedding
print(f"{'='*60}")
print("Building Universal LSTM Model")
print(f"{'='*60}\n")

# Input layers
sequence_input = Input(shape=(n_timesteps, n_features), name='sequence_input')
stock_id_input = Input(shape=(1,), name='stock_id_input')

# Stock embedding (learn stock-specific features)
n_stocks = len(df_all['Stock_Symbol'].unique())
embedding_dim = 8
stock_embedding = Embedding(input_dim=n_stocks, output_dim=embedding_dim, 
                           name='stock_embedding')(stock_id_input)
stock_embedding = tf.keras.layers.Flatten()(stock_embedding)

# LSTM branch for sequences
lstm_out = LSTM(128, return_sequences=True)(sequence_input)
lstm_out = Dropout(0.2)(lstm_out)
lstm_out = LSTM(64, return_sequences=True)(lstm_out)
lstm_out = Dropout(0.2)(lstm_out)
lstm_out = LSTM(32, return_sequences=False)(lstm_out)
lstm_out = Dropout(0.2)(lstm_out)

# Combine LSTM output with stock embedding
combined = Concatenate()([lstm_out, stock_embedding])

# Dense layers
dense = Dense(32, activation='relu')(combined)
dense = Dropout(0.2)(dense)
dense = Dense(16, activation='relu')(dense)
output = Dense(1, name='output')(dense)

# Create model
model = Model(inputs=[sequence_input, stock_id_input], outputs=output)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(model.summary())
print()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_path = os.path.join(os.path.dirname(__file__), "universal_lstm_model.h5")
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')

# Train model
print(f"{'='*60}")
print("Training Universal LSTM Model...")
print(f"{'='*60}\n")

history = model.fit(
    [X_train, X_train_stock_ids],
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print(f"\nTraining completed!")
print(f"Best epoch: {early_stop.stopped_epoch - early_stop.patience + 1}")
print(f"Best validation loss: {min(history.history['val_loss']):.6f}\n")

# Evaluate
print(f"{'='*60}")
print("Evaluation Results")
print(f"{'='*60}\n")

# Overall performance
test_pred_scaled = model.predict([X_test, X_test_stock_ids], verbose=0)
test_pred = scaler_y.inverse_transform(test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

overall_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
overall_mae = mean_absolute_error(y_test_actual, test_pred)
overall_r2 = r2_score(y_test_actual, test_pred)
overall_mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100

print(f"Overall Test Performance:")
print(f"  RMSE: {overall_rmse:.2f} NPR")
print(f"  MAE: {overall_mae:.2f} NPR")
print(f"  MAPE: {overall_mape:.2f}%")
print(f"  R² Score: {overall_r2:.4f}\n")

# Per-stock performance
print(f"{'-'*60}")
print("Per-Stock Performance:")
print(f"{'-'*60}\n")

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
        
        print(f"{stock_symbol}:")
        print(f"  Test samples: {np.sum(stock_mask)}")
        print(f"  RMSE: {stock_rmse:.2f} NPR")
        print(f"  MAE: {stock_mae:.2f} NPR")
        print(f"  MAPE: {stock_mape:.2f}%")
        print(f"  R² Score: {stock_r2:.4f}\n")

print(f"{'='*60}")
print("Universal LSTM Model Complete!")
print(f"{'='*60}\n")

print(f"✓ Model saved to: {model_path}")
print(f"✓ Can now predict: {', '.join(df_all['Stock_Symbol'].unique())}")
print(f"✓ Ready to add more stocks - just provide CSV files!\n")

# Save encoders for future use
import pickle
encoder_path = os.path.join(os.path.dirname(__file__), "stock_encoder.pkl")
scaler_x_path = os.path.join(os.path.dirname(__file__), "scaler_X.pkl")
scaler_y_path = os.path.join(os.path.dirname(__file__), "scaler_y.pkl")

with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
with open(scaler_x_path, 'wb') as f:
    pickle.dump(scaler_X, f)
with open(scaler_y_path, 'wb') as f:
    pickle.dump(scaler_y, f)

print(f"✓ Saved encoders and scalers for future predictions")
