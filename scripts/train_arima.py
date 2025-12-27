# """
# Train an ARIMA model directly from a preprocessed CSV file.

# Usage:
#     python scripts/train_arima.py <path_to_csv> [output_folder]
# Example:
#     python scripts/train_arima.py data/monthly_grouped_shap.csv results/run_arima_1
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
# from sklearn.preprocessing import MinMaxScaler
# from statsmodels.tsa.stattools import adfuller

# # --- Arguments ---
# if len(sys.argv) < 2:
#     print("Usage: python train_arima_auto.py <csv_path> [output_folder]")
#     sys.exit(1)

# csv_path = sys.argv[1]
# output_folder = sys.argv[2] if len(sys.argv) > 2 else "results/auto_arima"
# os.makedirs(output_folder, exist_ok=True)

# # --- Load data ---
# df = pd.read_csv(csv_path)
# df['month'] = pd.to_datetime(df['month'])
# df.set_index('month', inplace=True)

# scaler = MinMaxScaler(feature_range=(0, 1)) 
# df['sales_scaled'] = scaler.fit_transform(df[['sales']])

# train_size = int(len(df) * 0.7)
# train, test = df['sales_scaled'][:train_size], df['sales_scaled'][train_size:]

# train_size = int(len(df) * 0.7)
# train, test = df['sales_scaled'][:train_size], df['sales_scaled'][train_size:]

# # Check for stationarity using the Augmented Dickey-Fuller test on the train set
# result = adfuller(train)
# print(f"ADF Statistic: {result[0]}")
# print(f"p-value: {result[1]}")
# if result[1] < 0.05:
#     print("The series is stationary.")
# else:
#     print("The series is not stationary. Differencing might be required.")

# # Differencing if needed (if the series is not stationary)
# train_diff = train.diff().dropna()
# # Fit ARIMA model (you can choose p, d, q based on ACF and PACF plots)
# model = ARIMA(train, order=(1, 1, 1))  # Example order (p, d, q)
# model_fit = model.fit()

# forecast_steps = len(test)
# forecasted_sales = model_fit.forecast(steps=forecast_steps)

# # Get the last 12 actual sales data points for comparison (if available)
# actual_sales_last_12 = test[-12:]  # Last 12 months from the test set
# actual_sales_last_12_values = actual_sales_last_12.values
# forecasted_sales_values = forecasted_sales[:12]
# # Check the length of both arrays
# print("Length of actual_sales_last_12:", len(actual_sales_last_12_values))
# print("Length of forecasted_sales[:12]:", len(forecasted_sales_values))

# # Ensure both arrays are aligned
# if len(actual_sales_last_12_values) != len(forecasted_sales_values):
#     raise ValueError("The actual and forecasted sales arrays have different lengths.")

# # Calculate MAPE (Mean Absolute Percentage Error) after ensuring both are numeric arrays
# mape = np.mean(np.abs((actual_sales_last_12_values - forecasted_sales_values) / actual_sales_last_12_values)) * 100
# mse = mean_squared_error(actual_sales_last_12_values, forecasted_sales_values)  # MSE
# msle = np.mean((np.log1p(actual_sales_last_12_values) - np.log1p(forecasted_sales_values))**2)  # MSLE
# smape = np.mean(
#     2 * np.abs(forecasted_sales_values - actual_sales_last_12_values)
#     / (np.abs(actual_sales_last_12_values) + np.abs(forecasted_sales_values))
# ) * 100

# metrics_path = os.path.join(output_folder, "metrics.txt")
# with open(metrics_path, "a") as f:
#     f.write(f"ARIMA Results for (1,1,1):\n")
#     f.write(f"MSE   : {mse:.6f}\n")
#     f.write(f"MSLE  : {msle:.6f}\n")
#     f.write(f"MAPE  : {mape:.6f}\n")
#     f.write(f"SMAPE : {smape:.6f}\n")
# print(f"Saved metrics -> {metrics_path}")

# # --- Plot ---
# plt.plot(df.index[:train_size], train, label='Observed Sales (Train)')
# plt.plot(df.index[train_size:], test, label='Observed Sales (Test)')
# plt.plot(df.index[train_size:], forecasted_sales, label='Forecasted Sales(ARIMA)', linestyle='--')
# plt.legend()
# plt.title("Observed vs Forecasted Sales(ARIMA)")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.savefig(os.path.join(output_folder, "forecast.png"), dpi=200)
# plt.close()
# print("Saved forecast plot.")

"""
Train an Auto-ARIMA model directly from a preprocessed CSV file.

Usage:
    python scripts/train_auto_arima.py <path_to_csv> [output_folder]
Example:
    python scripts/train_auto_arima.py data/monthly_grouped_shap.csv results/run_auto_arima_1
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
    print("Usage: python train_auto_arima.py <csv_path> [output_folder]")
    sys.exit(1)

csv_path = sys.argv[1]
output_folder = sys.argv[2] if len(sys.argv) > 2 else "results/auto_arima"
os.makedirs(output_folder, exist_ok=True)

# --- Load data ---
df = pd.read_csv(csv_path)
df['month'] = pd.to_datetime(df['month'])
df.set_index('month', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
df['sales_scaled'] = scaler.fit_transform(df[['sales']])

train_size = int(len(df) * 0.7)
train = df['sales_scaled'][:train_size]
test = df['sales_scaled'][train_size:]

# --- Stationarity check (informational) ---
adf_stat, p_value, *_ = adfuller(train)
print(f"ADF Statistic: {adf_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# --- Auto-ARIMA ---
print("Fitting Auto-ARIMA...")
model = auto_arima(
    train,
    start_p=0,
    start_q=0,
    max_p=5,
    max_q=5,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=True
)

print(f"Selected ARIMA order: {model.order}")

# --- Forecast ---
forecast_steps = len(test)
forecasted_sales = model.predict(n_periods=forecast_steps)

# --- Evaluation (last 12 months) ---
actual_sales_last_12 = test[-12:].values
forecasted_sales_last_12 = forecasted_sales[:12]

if len(actual_sales_last_12) != len(forecasted_sales_last_12):
    raise ValueError("Actual and forecast lengths do not match.")

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
    f.write(f"Auto-ARIMA order: {model.order}\n")
    f.write(f"MSE   : {mse:.6f}\n")
    f.write(f"MSLE  : {msle:.6f}\n")
    f.write(f"MAPE  : {mape:.6f}\n")
    f.write(f"SMAPE : {smape:.6f}\n")

print(f"Saved metrics -> {metrics_path}")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(df.index[:train_size], train, label="Observed Sales (Train)")
plt.plot(df.index[train_size:], test, label="Observed Sales (Test)")
plt.plot(df.index[train_size:], forecasted_sales, "--", label="Forecasted Sales (Auto-ARIMA)")
plt.legend()
plt.title("Observed vs Forecasted Sales (Auto-ARIMA)")
plt.xlabel("Date")
plt.ylabel("Sales (Scaled)")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "ARIMA forecast.png"), dpi=200)
plt.close()

print("Saved forecast plot.")
