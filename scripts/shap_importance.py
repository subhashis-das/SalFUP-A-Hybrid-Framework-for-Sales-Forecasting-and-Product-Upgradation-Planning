"""
Compute SHAP feature importances for ALL months in the dataset (not just 3)
by sliding window attribution — each month gets importance based on the
sequence ending at that month.
"""
import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# === Load model & data ===
model_path = "results/pre_tuning/lstm_model.keras"
data_path = "data/monthly_grouped.csv"

assert os.path.exists(model_path), f"Model not found: {model_path}"
assert os.path.exists(data_path), f"Data not found: {data_path}"

model = load_model(model_path)
df = pd.read_csv(data_path, parse_dates=["month"])

# === Prepare features & sequences ===
features = [
    'rating', 'helpful_vote', 'verified_purchase',
    'review_title_tfidf', 'review_text_tfidf',
    'review_title_sentiment', 'review_text_sentiment'
]

X = df[features].values
seq_len = 3  # same as training
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Build all sequences (rolling)
X_seq, month_labels = [], []
for i in range(seq_len, len(X_scaled)):
    X_seq.append(X_scaled[i-seq_len:i])
    month_labels.append(df['month'].iloc[i])
X_seq = np.array(X_seq)
month_labels = np.array(month_labels)

# === Flatten for SHAP ===
X_train_flat = X_seq.reshape(X_seq.shape[0], -1)
n_features = len(features)

# === SHAP explainer ===
def model_wrapper(x_flat):
    x_reshaped = x_flat.reshape(-1, seq_len, n_features)
    return model.predict(x_reshaped)

# sample small background
background = X_train_flat[np.random.choice(X_train_flat.shape[0],
                                           min(50, X_train_flat.shape[0]), replace=False)]
explainer = shap.KernelExplainer(model_wrapper, background)

# === Run SHAP for entire timeline (batched) ===
print("Running SHAP across all months (may take time)...")
all_importance = []

for i in tqdm(range(0, len(X_train_flat), 10)):  # process in small batches
    batch = X_train_flat[i:i+10]
    shap_values = explainer.shap_values(batch, nsamples=100)
    sv = np.array(shap_values[0] if isinstance(shap_values, list) else shap_values)
    abs_sv = np.abs(sv).reshape(batch.shape[0], seq_len, n_features)
    # assign importance to last timestep (current month)
    current_month_imp = abs_sv[:, -1, :]
    all_importance.append(current_month_imp)

all_importance = np.vstack(all_importance)

# === Aggregate results per month ===
importance_df = pd.DataFrame(all_importance, columns=features)
importance_df['month'] = month_labels
monthly_importance = importance_df.groupby('month')[features].mean().reset_index()

# === Normalize each month to % ===
monthly_importance_norm = monthly_importance.copy()
monthly_importance_norm[features] = monthly_importance[features].div(
    monthly_importance[features].sum(axis=1), axis=0
) * 100

# === Save results ===
os.makedirs("results", exist_ok=True)
raw_csv = "results/shap_monthly_feature_importance_raw.csv"
norm_csv = "results/shap_monthly_feature_importance_normalized.csv"
monthly_importance.to_csv(raw_csv, index=False)
monthly_importance_norm.to_csv(norm_csv, index=False)

print(f"Saved per-month SHAP importances:")
print(f" - Raw values: {raw_csv}")
print(f" - Normalized %: {norm_csv}")

# === Plot trends ===
plt.figure(figsize=(10,6))
for f in features:
    plt.plot(monthly_importance_norm['month'], monthly_importance_norm[f], label=f)
plt.xlabel("Month")
plt.ylabel("Relative Importance (%)")
plt.title("SHAP Feature Importance Trend Over Time")
plt.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/shap_monthly_feature_trends.png", dpi=200)
plt.close()
print("Saved trend plot to results/shap_monthly_feature_trends.png")
