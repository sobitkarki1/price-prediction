"""
LSTM Model for SCB Bank Stock Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")
data_path = os.path.join(data_dir, "stock_daily_scb.csv")

print(f"{'='*60}")
print(f"LSTM Model - SCB Bank Stock Price Prediction")
print(f"{'='*60}\n")

df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates(subset=['Date'], keep='first')
df = df[df['Close'] > 0]

print(f"Dataset: {len(df)} trading days")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")

# Features
df['return'] = df['Close'].pct_change()
df['volume_change'] = df['Volume'].pct_change()

for n in [5, 10, 20]:
    df[f'sma_{n}'] = df['Close'].rolling(n).mean()
    df[f'std_{n}'] = df['Close'].rolling(n).std()

df['rsi'] = 50
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

df = df.dropna()

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 
                'volume_change', 'sma_5', 'sma_10', 'sma_20', 'std_5', 
                'std_10', 'std_20', 'rsi']

data = df[feature_cols].values
target = df['Close'].values

print(f"Features: {len(feature_cols)}\n")

# Scale data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

data_scaled = scaler_X.fit_transform(data)
target_scaled = scaler_y.fit_transform(target.reshape(-1, 1))

# Create sequences
def create_sequences(data, target, lookback=30, forecast=5):
    X, y = [], []
    for i in range(lookback, len(data) - forecast):
        X.append(data[i-lookback:i])
        y.append(target[i + forecast])
    return np.array(X), np.array(y)

lookback = 30
forecast = 5
X, y = create_sequences(data_scaled, target_scaled, lookback, forecast)

print(f"Sequences: {X.shape}, Target: {y.shape}\n")

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

# Build model
print(f"{'='*60}")
print("Building LSTM Model")
print(f"{'='*60}\n")

model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(lookback, len(feature_cols))),
    Dropout(0.2),
    LSTM(units=64, return_sequences=True),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(units=16, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print(f"Training LSTM...\n")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nTraining complete!")
print(f"Best epoch: {early_stop.stopped_epoch - early_stop.patience + 1}\n")

# Evaluate
print(f"{'='*60}")
print("Evaluation Results")
print(f"{'='*60}\n")

test_pred_scaled = model.predict(X_test, verbose=0)
test_pred = scaler_y.inverse_transform(test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
test_mae = mean_absolute_error(y_test_actual, test_pred)
test_r2 = r2_score(y_test_actual, test_pred)
test_mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100

print(f"SCB Bank - LSTM Test Performance:")
print(f"  RMSE: {test_rmse:.2f} NPR")
print(f"  MAE: {test_mae:.2f} NPR")
print(f"  MAPE: {test_mape:.2f}%")
print(f"  R²: {test_r2:.4f}\n")

print(f"{'-'*60}")

# Save
model_path = os.path.join(os.path.dirname(__file__), "lstm_scb_model.h5")
model.save(model_path)
print(f"✓ Model saved to: {model_path}")

print(f"\n{'='*60}")
print("SCB LSTM Complete!")
print(f"{'='*60}")
