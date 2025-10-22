"""
LSTM (Long Short-Term Memory) Model for Stock Price Prediction
Uses deep learning to capture temporal patterns in stock prices
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "data")
data_path = os.path.join(data_dir, "stock_daily.csv")

print(f"{'='*60}")
print(f"LSTM Model - NABIL Stock Price Prediction")
print(f"{'='*60}\n")

# Load data
df = pd.read_csv(data_path, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop_duplicates(subset=['Date'], keep='first')
df = df[df['Close'] > 0]

print(f"Dataset: {len(df)} trading days")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")

# Create features
df['return'] = df['Close'].pct_change()
df['volume_change'] = df['Volume'].pct_change()

for n in [5, 10, 20]:
    df[f'sma_{n}'] = df['Close'].rolling(n).mean()
    df[f'std_{n}'] = df['Close'].rolling(n).std()

df['rsi'] = 50  # Simplified RSI initialization
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

df = df.dropna()

# Select features for LSTM
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 
                'volume_change', 'sma_5', 'sma_10', 'sma_20', 'std_5', 
                'std_10', 'std_20', 'rsi']

data = df[feature_cols].values
target = df['Close'].values

print(f"Features used: {len(feature_cols)}")
print(f"Feature names: {feature_cols}\n")

# Normalize data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

data_scaled = scaler_X.fit_transform(data)
target_scaled = scaler_y.fit_transform(target.reshape(-1, 1))

# Create sequences for LSTM
def create_sequences(data, target, lookback=30, forecast=5):
    """
    Create sequences for LSTM training
    lookback: number of days to look back
    forecast: number of days to predict ahead
    """
    X, y = [], []
    for i in range(lookback, len(data) - forecast):
        X.append(data[i-lookback:i])
        y.append(target[i + forecast])
    return np.array(X), np.array(y)

lookback = 30  # Use past 30 days
forecast = 5   # Predict 5 days ahead

X, y = create_sequences(data_scaled, target_scaled, lookback, forecast)

print(f"Sequence shape: {X.shape}")
print(f"Target shape: {y.shape}\n")

# Train-test split (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}\n")

# Build LSTM Model
print(f"{'='*60}")
print("Building LSTM Neural Network")
print(f"{'='*60}\n")

model = Sequential([
    # First LSTM layer with dropout
    LSTM(units=128, return_sequences=True, input_shape=(lookback, len(feature_cols))),
    Dropout(0.2),
    
    # Second LSTM layer
    LSTM(units=64, return_sequences=True),
    Dropout(0.2),
    
    # Third LSTM layer
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    
    # Dense layers
    Dense(units=16, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(model.summary())
print()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
print(f"{'='*60}")
print("Training LSTM Model...")
print(f"{'='*60}\n")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nTraining completed!")
print(f"Best epoch: {early_stop.stopped_epoch - early_stop.patience + 1}")
print(f"Final training loss: {history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}\n")

# Make predictions
print(f"{'='*60}")
print("Evaluating LSTM Model")
print(f"{'='*60}\n")

# Predictions on training set
train_pred_scaled = model.predict(X_train, verbose=0)
train_pred = scaler_y.inverse_transform(train_pred_scaled)
y_train_actual = scaler_y.inverse_transform(y_train)

train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
train_mae = mean_absolute_error(y_train_actual, train_pred)
train_r2 = r2_score(y_train_actual, train_pred)

print(f"Training Set Performance:")
print(f"  RMSE: {train_rmse:.2f} NPR")
print(f"  MAE: {train_mae:.2f} NPR")
print(f"  R² Score: {train_r2:.4f}\n")

# Predictions on test set
test_pred_scaled = model.predict(X_test, verbose=0)
test_pred = scaler_y.inverse_transform(test_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
test_mae = mean_absolute_error(y_test_actual, test_pred)
test_r2 = r2_score(y_test_actual, test_pred)
test_mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100

print(f"Test Set Performance:")
print(f"  RMSE: {test_rmse:.2f} NPR")
print(f"  MAE: {test_mae:.2f} NPR")
print(f"  MAPE: {test_mape:.2f}%")
print(f"  R² Score: {test_r2:.4f}\n")

print(f"{'-'*60}")
print(f"LSTM Model Summary:")
print(f"  Lookback: {lookback} days")
print(f"  Forecast: {forecast} days ahead")
print(f"  Architecture: 128 → 64 → 32 → 16 → 1")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.2f} NPR")
print(f"{'-'*60}")

# Show sample predictions
print(f"\n{'='*60}")
print("Sample Predictions (Last 10 from Test Set)")
print(f"{'='*60}\n")
print(f"{'Actual':>12} | {'Predicted':>12} | {'Error':>12} | {'Error %':>10}")
print(f"{'-'*60}")
for i in range(-10, 0):
    actual = y_test_actual[i][0]
    predicted = test_pred[i][0]
    error = predicted - actual
    error_pct = (error / actual) * 100
    print(f"{actual:>12.2f} | {predicted:>12.2f} | {error:>12.2f} | {error_pct:>9.2f}%")

print(f"\n{'='*60}")
print("LSTM Model Complete!")
print(f"{'='*60}")

# Save model
model_path = os.path.join(os.path.dirname(__file__), "lstm_nabil_model.h5")
model.save(model_path)
print(f"\n✓ Model saved to: {model_path}")
