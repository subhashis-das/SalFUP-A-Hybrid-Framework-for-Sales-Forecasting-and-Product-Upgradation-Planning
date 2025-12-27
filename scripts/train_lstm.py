"""
Train an LSTM model directly from a preprocessed CSV file.

Usage:
    python scripts/train_lstm.py <path_to_csv> [output_folder]
Example:
    python scripts/train_lstm.py data/monthly_grouped_shap.csv results/run_shap_1
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
from tensorflow.keras import layers, models

# --- Parse arguments ---
if len(sys.argv) < 2:
    print("Usage: python scripts/train_lstm.py <path_to_csv> [output_folder]")
    sys.exit(1)

csv_path = sys.argv[1]
output_folder = sys.argv[2] if len(sys.argv) > 2 else "results/default_run"

if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    sys.exit(1)

os.makedirs(output_folder, exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Load data ---
df = pd.read_csv(csv_path, parse_dates=['month'])
print(f"Loaded {csv_path}, shape = {df.shape}")

# --- Feature preparation ---
if 'verified_purchase' in df.columns:
    df['verified_purchase'] = df['verified_purchase'].astype(int)

features = [
    'rating', 'helpful_vote', 'verified_purchase',
    'review_title_tfidf', 'review_text_tfidf',
    'review_title_sentiment', 'review_text_sentiment'
]
X = df[features].values
y = df['sales'].values.reshape(-1, 1)
df['log_sales'] = np.log1p(df['sales'])
y = df['log_sales'].values.reshape(-1, 1)

# --- Scaling ---
scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# --- Sequence creation ---
seq_len = 3
X_seq, y_seq = [], []
for i in range(seq_len, len(X_scaled)):
    X_seq.append(X_scaled[i - seq_len:i])
    y_seq.append(y_scaled[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# --- Split ---
split1, split2 = int(0.7 * len(X_seq)), int(0.85 * len(X_seq))
X_train, y_train = X_seq[:split1], y_seq[:split1]
X_val, y_val = X_seq[split1:split2], y_seq[split1:split2]
X_test, y_test = X_seq[split2:], y_seq[split2:]
print(f"Data split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# --- Build model ---
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.LSTM(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=8, verbose=1
)

# --- Predict ---
y_pred_scaled = model.predict(X_test)

# --- Compute metrics on scaled values ---
mse = mean_squared_error(y_test, y_pred_scaled)
msle = mean_squared_log_error(np.clip(y_test, 0, 1), np.clip(y_pred_scaled, 0, 1))

# Ensure numpy arrays
actual_np = np.asarray(y_test).reshape(-1, 1)
forecast_np = np.asarray(y_pred_scaled).reshape(-1, 1)

# Inverse transform
actual_unscaled = scaler_y.inverse_transform(actual_np).ravel()
forecast_unscaled = scaler_y.inverse_transform(forecast_np).ravel()


mape = mean_absolute_percentage_error(actual_unscaled, forecast_unscaled)
smape = smape = np.mean(
    2 * np.abs(y_pred_scaled - y_test) / (np.abs(y_test) + np.abs(y_pred_scaled) + 1e-6)
)

metrics_path = os.path.join(output_folder, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("LSTM Model Evaluation Metrics (on scaled data)\n")
    f.write(f"MSE  : {mse:.6f}\n")
    f.write(f"MSLE : {msle:.6f}\n")
    f.write(f"MAPE : {mape:.6f}\n")
    f.write(f"SMAPE : {smape:.6f}\n")
print(f"Saved metrics -> {metrics_path}")

# --- Inverse transform for visualization ---
y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
y_true_log = scaler_y.inverse_transform(y_test)

y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_true_log)

# --- Forecast plot ---
plt.figure(figsize=(10, 4))
plt.plot(df['month'].iloc[-len(y_true):], y_true, label='True Sales')
plt.plot(df['month'].iloc[-len(y_pred):], y_pred, label='Predicted Sales')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Sales Forecast (LSTM)')
plt.legend()
plt.tight_layout()
forecast_path = os.path.join(output_folder, "lstm_forecast.png")
plt.savefig(forecast_path, dpi=200)
plt.close()
print(f"Saved forecast plot -> {forecast_path}")

# --- Loss curve ---
plt.figure(figsize=(6, 3))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training Loss Curve')
plt.legend()
plt.tight_layout()
loss_path = os.path.join(output_folder, "loss_curve.png")
plt.savefig(loss_path, dpi=200)
plt.close()
print(f"Saved loss curve -> {loss_path}")

# --- Save model ---
model_path = os.path.join(output_folder, "lstm_model.keras")
model.save(model_path)
print(f"Saved trained model -> {model_path}")
