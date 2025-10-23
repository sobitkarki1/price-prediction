"""
Optimized Universal LSTM Training Script
Works on both CPU and GPU (auto-detects)

For CPU: Uses optimized batch processing and multiprocessing
For GPU: Uses mixed precision and larger batches

Author: Price Prediction Project
Date: October 2025
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from datetime import datetime

# ========== GPU/CPU CONFIGURATION ==========
print("\n" + "="*80)
print("HARDWARE CONFIGURATION")
print("="*80)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ {len(gpus)} GPU(s) detected - Using GPU acceleration")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ GPU memory growth enabled")
        print("✅ Mixed precision (FP16) enabled for faster training")
        BATCH_SIZE = 256  # Larger batch for GPU
    except RuntimeError as e:
        print(f"⚠️  GPU configuration warning: {e}")
        BATCH_SIZE = 128
else:
    print("⚠️  No GPU detected - Using optimized CPU training")
    print("   Training will be slower but will still work")
    print("   Consider: Google Colab (free GPU) or reduce dataset size")
    BATCH_SIZE = 64  # Smaller batch for CPU

print(f"   Batch size: {BATCH_SIZE}")
print("="*80 + "\n")

# ========== CONFIGURATION ==========
LOOKBACK = 30
FORECAST_HORIZON = 5
MAX_EPOCHS = 150
EARLY_STOP_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
EMBEDDING_DIM = 8

# ========== LOAD DATA ==========
print("📂 Loading stock data...")

def load_stock_data(symbol):
    """Load and preprocess data for a single stock"""
    file_path = f'data/processed/stock_daily_{symbol}.csv'
    
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Feature engineering
        df['return'] = df['Close'].pct_change()
        df['volume_change'] = df['Volume'].pct_change()
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        
        # Volatility
        df['std_5'] = df['Close'].rolling(window=5).std()
        df['std_10'] = df['Close'].rolling(window=10).std()
        df['std_20'] = df['Close'].rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df = df.dropna()
        
        if len(df) < LOOKBACK + FORECAST_HORIZON:
            return None
        
        return df
    except Exception as e:
        print(f"⚠️  Error loading {symbol}: {e}")
        return None

# Load supported stocks
print("📋 Loading stock symbols...")
with open('src/lstm_model/supported_stocks.txt', 'r') as f:
    stock_symbols = [line.strip() for line in f if line.strip()]

print(f"📊 Found {len(stock_symbols)} stock symbols")
print("📂 Loading stock data (this may take a minute)...")

all_data = []
failed_stocks = []
progress_interval = max(1, len(stock_symbols) // 10)

for i, symbol in enumerate(stock_symbols, 1):
    df = load_stock_data(symbol)
    if df is not None:
        df['stock_symbol'] = symbol
        all_data.append(df)
    else:
        failed_stocks.append(symbol)
    
    if i % progress_interval == 0 or i == len(stock_symbols):
        print(f"   Progress: {i}/{len(stock_symbols)} ({i*100//len(stock_symbols)}%)")

print(f"✅ Successfully loaded {len(all_data)} stocks")
if failed_stocks:
    print(f"⚠️  Failed: {len(failed_stocks)} stocks (insufficient data)")

# Combine all data
df_all = pd.concat(all_data, ignore_index=True)
print(f"📊 Combined dataset: {len(df_all):,} records")
print(f"📅 Date range: {df_all['Date'].min().date()} to {df_all['Date'].max().date()}")

# ========== CREATE SEQUENCES ==========
print("\n🔄 Creating training sequences...")

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'volume_change',
                'sma_5', 'sma_10', 'sma_20', 'std_5', 'std_10', 'std_20', 'rsi']

X_seq_list = []
X_stock_list = []
y_list = []

# Encode stocks
stock_encoder = LabelEncoder()
stock_encoder.fit(df_all['stock_symbol'].unique())
n_stocks = len(stock_encoder.classes_)

print(f"📊 Processing {n_stocks} stocks...")

for idx, symbol in enumerate(stock_encoder.classes_, 1):
    df_stock = df_all[df_all['stock_symbol'] == symbol].copy()
    df_stock = df_stock.sort_values('Date').reset_index(drop=True)
    
    stock_id = stock_encoder.transform([symbol])[0]
    
    for i in range(len(df_stock) - LOOKBACK - FORECAST_HORIZON + 1):
        X_seq = df_stock.iloc[i:i+LOOKBACK][feature_cols].values
        y = df_stock.iloc[i+LOOKBACK+FORECAST_HORIZON-1]['Close']
        
        X_seq_list.append(X_seq)
        X_stock_list.append(stock_id)
        y_list.append(y)
    
    if idx % 25 == 0 or idx == n_stocks:
        print(f"   Processed: {idx}/{n_stocks} stocks ({len(X_seq_list):,} sequences so far)")

X_seq = np.array(X_seq_list, dtype=np.float32)
X_stock = np.array(X_stock_list, dtype=np.int32)
y = np.array(y_list, dtype=np.float32)

print(f"✅ Created {len(X_seq):,} sequences")
print(f"   Sequence shape: {X_seq.shape}")
print(f"   Memory usage: {X_seq.nbytes / 1024 / 1024:.1f} MB")

# ========== SCALE DATA ==========
print("\n📏 Normalizing data...")

scaler_X = StandardScaler()
X_seq_reshaped = X_seq.reshape(-1, X_seq.shape[2])
X_seq_scaled = scaler_X.fit_transform(X_seq_reshaped)
X_seq_scaled = X_seq_scaled.reshape(X_seq.shape).astype(np.float32)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

print("✅ Normalization complete")

# ========== TRAIN-TEST SPLIT ==========
print("\n✂️  Splitting dataset...")

X_seq_train, X_seq_test, X_stock_train, X_stock_test, y_train, y_test = train_test_split(
    X_seq_scaled, X_stock, y_scaled, test_size=0.2, random_state=42, shuffle=True
)

print(f"✅ Training samples: {len(X_seq_train):,}")
print(f"✅ Testing samples: {len(X_seq_test):,}")
print(f"   Split ratio: {len(X_seq_train)*100//len(X_seq):}% train, {len(X_seq_test)*100//len(X_seq):}% test")

# ========== BUILD MODEL ==========
print("\n🏗️  Building Universal LSTM model...")

# Time series input
seq_input = Input(shape=(LOOKBACK, len(feature_cols)), name='sequence_input')
lstm1 = LSTM(128, return_sequences=True)(seq_input)
dropout1 = Dropout(0.2)(lstm1)
lstm2 = LSTM(64, return_sequences=True)(dropout1)
dropout2 = Dropout(0.2)(lstm2)
lstm3 = LSTM(32, return_sequences=False)(dropout2)
dropout3 = Dropout(0.2)(lstm3)

# Stock embedding input
stock_input = Input(shape=(1,), name='stock_input')
stock_embedding = Embedding(n_stocks, EMBEDDING_DIM, name='stock_embedding')(stock_input)
stock_flat = Flatten()(stock_embedding)

# Combine
combined = Concatenate()([dropout3, stock_flat])
dense1 = Dense(64, activation='relu')(combined)
dropout4 = Dropout(0.2)(dense1)
dense2 = Dense(32, activation='relu')(dropout4)
output = Dense(1)(dense2)

model = Model(inputs=[seq_input, stock_input], outputs=output)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("✅ Model built successfully!")
print(f"\n📊 Model Summary:")
print(f"   Total parameters: {model.count_params():,}")
print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
print(f"   Model size: ~{model.count_params() * 4 / 1024:.1f} KB")

# ========== SETUP CALLBACKS ==========
print("\n⚙️  Configuring training callbacks...")

callbacks = [
    ModelCheckpoint(
        'src/lstm_model/universal_lstm_all_stocks.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"✅ Callbacks configured:")
print(f"   - Model checkpoint (save best)")
print(f"   - Early stopping (patience={EARLY_STOP_PATIENCE})")
print(f"   - Learning rate reduction (patience={REDUCE_LR_PATIENCE})")

# ========== TRAIN MODEL ==========
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print(f"Stocks: {n_stocks}")
print(f"Training sequences: {len(X_seq_train):,}")
print(f"Validation sequences: {len(X_seq_test):,}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {MAX_EPOCHS}")
print(f"Early stopping: {EARLY_STOP_PATIENCE} epochs")
print(f"Device: {'GPU' if gpus else 'CPU'}")
print("="*80 + "\n")

start_time = datetime.now()

try:
    history = model.fit(
        [X_seq_train, X_stock_train],
        y_train,
        validation_data=([X_seq_test, X_stock_test], y_test),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        use_multiprocessing=True if not gpus else False,
        workers=4 if not gpus else 1
    )
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"⏱️  Total time: {training_duration/60:.1f} minutes ({training_duration/3600:.2f} hours)")
    print(f"⏱️  Average time per epoch: {training_duration/len(history.history['loss']):.1f} seconds")
    print(f"📊 Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"📊 Best validation MAE: {min(history.history['val_mae']):.6f}")
    print(f"📊 Epochs trained: {len(history.history['loss'])}")
    print("="*80)
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user!")
    print("   Best model checkpoint has been saved.")
    training_duration = (datetime.now() - start_time).total_seconds()
    print(f"   Time elapsed: {training_duration/60:.1f} minutes")

# ========== SAVE ARTIFACTS ==========
print("\n💾 Saving encoders and scalers...")

with open('src/lstm_model/stock_encoder_all.pkl', 'wb') as f:
    pickle.dump(stock_encoder, f)
print("✅ Saved: stock_encoder_all.pkl")

with open('src/lstm_model/scaler_X_all.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
print("✅ Saved: scaler_X_all.pkl")

with open('src/lstm_model/scaler_y_all.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("✅ Saved: scaler_y_all.pkl")

# ========== FINAL EVALUATION ==========
print("\n📊 Evaluating final model on test set...")

# Load best model
best_model = load_model('src/lstm_model/universal_lstm_all_stocks.h5')

test_loss, test_mae = best_model.evaluate(
    [X_seq_test, X_stock_test], 
    y_test, 
    verbose=0,
    batch_size=BATCH_SIZE
)

print(f"\n🎯 Final Test Metrics:")
print(f"   Loss (MSE): {test_loss:.6f}")
print(f"   MAE: {test_mae:.6f}")

# Calculate R² score
y_pred_scaled = best_model.predict(
    [X_seq_test, X_stock_test], 
    verbose=0,
    batch_size=BATCH_SIZE
)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

ss_res = np.sum((y_test_actual - y_pred) ** 2)
ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

rmse = np.sqrt(np.mean((y_test_actual - y_pred) ** 2))

print(f"   R² Score: {r2_score:.4f}")
print(f"   RMSE: {rmse:.2f} NPR")

# ========== SUMMARY ==========
print("\n" + "="*80)
print("🎉 ALL DONE! MODEL READY FOR PREDICTIONS")
print("="*80)
print("\n📁 Saved files:")
print("   - src/lstm_model/universal_lstm_all_stocks.h5")
print("   - src/lstm_model/stock_encoder_all.pkl")
print("   - src/lstm_model/scaler_X_all.pkl")
print("   - src/lstm_model/scaler_y_all.pkl")
print("\n🚀 Next steps:")
print("   1. Make predictions:")
print("      python src/lstm_model/universal_predict_all.py --stock NABIL")
print("   2. Evaluate all stocks:")
print("      python src/evaluate_universal_lstm_all.py")
print("   3. Batch predictions:")
print("      python src/lstm_model/universal_predict_all.py --top20")
print("="*80 + "\n")
