"""
Train a SARIMA (Seasonal Auto-ARIMA) model directly from a preprocessed CSV file.

Usage:
    python scripts/train_sarima.py <path_to_csv> [output_folder]

Example:
    python scripts/train_sarima.py data/monthly_grouped_shap.csv results/run_sarima_1
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

# --- Arguments ---
if len(sys.argv) < 2:
    print("Usage: python train_sarima.py <csv_path> [output_folder]")
    sys.exit(1)

csv_path = sys.argv[1]
output_folder = sys.argv[2] if len(sys.argv) > 2 else "results/sarima"
os.makedirs(output_folder, exist_ok=True)

# --- Load data ---
df = pd.read_csv(csv_path)
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)

# --- Scale sales ---
scaler = MinMaxScaler(feature_range=(0, 1))
df['sales_scaled'] = scaler.fit_transform(df[['sales']])

# --- Train / test split ---
train_size = int(len(df) * 0.7)
train = df['sales_scaled'][:train_size]
test = df['sales_scaled'][train_size:]

# --- Stationarity check (informational only) ---
adf_stat, p_value, *_ = adfuller(train)
print(f"ADF Statistic: {adf_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# --- SARIMA (Seasonal Auto-ARIMA) ---
print("Fitting SARIMA (Seasonal Auto-ARIMA)...")

model = auto_arima(
    train,
    start_p=0,
    start_q=0,
    max_p=5,
    max_q=5,
    start_P=0,
    start_Q=0,
    max_P=2,
    max_Q=2,
    seasonal=True,
    m=12,                  # monthly seasonality
    d=None,
    D=None,
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)

print(f"Selected SARIMA order: {model.order}")
print(f"Selected Seasonal order: {model.seasonal_order}")

# --- Forecast ---
forecast_steps = len(test)
forecasted_sales = model.predict(n_periods=forecast_steps)

# --- Evaluation (last 12 months of test set) ---
actual_sales_last_12 = test[-12:].values
forecasted_sales_last_12 = forecasted_sales[:12]

if len(actual_sales_last_12) != len(forecasted_sales_last_12):
    raise ValueError("Actual and forecast lengths do not match.")

# --- Metrics ---
mse = mean_squared_error(actual_sales_last_12, forecasted_sales_last_12)

# Ensure numpy arrays
actual_np = np.asarray(actual_sales_last_12).reshape(-1, 1)
forecast_np = np.asarray(forecasted_sales_last_12).reshape(-1, 1)

# Inverse transform
actual_unscaled = scaler.inverse_transform(actual_np).ravel()
forecast_unscaled = scaler.inverse_transform(forecast_np).ravel()

mape = np.mean(np.abs((actual_unscaled - forecast_unscaled) / actual_unscaled)) * 100

msle = np.mean(
    (np.log1p(actual_sales_last_12) - np.log1p(forecasted_sales_last_12)) ** 2
)

smape = np.mean(
    2 * np.abs(forecasted_sales_last_12 - actual_sales_last_12)
    / (np.abs(actual_sales_last_12) + np.abs(forecasted_sales_last_12))
) * 100

# --- Save metrics ---
metrics_path = os.path.join(output_folder, "metrics.txt")
with open(metrics_path, "a") as f:
    f.write(f"SARIMA order: {model.order}\n")
    f.write(f"Seasonal order: {model.seasonal_order}\n")
    f.write(f"MSE   : {mse:.6f}\n")
    f.write(f"MSLE  : {msle:.6f}\n")
    f.write(f"MAPE  : {mape:.6f}\n")
    f.write(f"SMAPE : {smape:.6f}\n")

print(f"Saved metrics -> {metrics_path}")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(df.index[:train_size], train, label="Observed Sales (Train)")
plt.plot(df.index[train_size:], test, label="Observed Sales (Test)")
plt.plot(
    df.index[train_size:],
    forecasted_sales,
    linestyle="--",
    label="Forecasted Sales (SARIMA)"
)
plt.legend()
plt.title("Observed vs Forecasted Sales (SARIMA)")
plt.xlabel("Date")
plt.ylabel("Sales (Scaled)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "SARIMA forecast.png"), dpi=200)
plt.close()

print("Saved forecast plot.")
