"""
Apply normalized SHAP feature weights to the training part of monthly_grouped.csv
and save the weighted dataset as monthly_grouped_shap.csv
"""
import pandas as pd
import numpy as np
import os

# === File paths ===
base_csv = "data/monthly_grouped.csv"
shap_csv = "results/shap_monthly_feature_importance_normalized.csv"
out_csv  = "data/monthly_grouped_shap.csv"

assert os.path.exists(base_csv), f"Missing: {base_csv}"
assert os.path.exists(shap_csv), f"Missing: {shap_csv}"

# === Load data ===
df = pd.read_csv(base_csv, parse_dates=["month"])
shap_norm = pd.read_csv(shap_csv, parse_dates=["month"])

# Align both DataFrames
merged = pd.merge(df, shap_norm, on="month", suffixes=("", "_shap"))
print(f"Merged shape: {merged.shape}")

# === Define feature columns ===
features = [
    'rating', 'helpful_vote', 'verified_purchase',
    'review_title_tfidf', 'review_text_tfidf',
    'review_title_sentiment', 'review_text_sentiment'
]

# Convert SHAP % to fractional weights
for f in features:
    merged[f + "_shap"] = merged[f + "_shap"] / 100.0

# === Apply weights (elementwise multiply feature × shap weight) ===
for f in features:
    merged[f + "_weighted"] = merged[f] * merged[f + "_shap"]

# === Keep only weighted features + target ===
weighted_cols = ['month'] + [f for f in features] + ['sales']
weighted_df = merged[weighted_cols]

# === Save result ===
weighted_df.to_csv(out_csv, index=False)
print(f"Saved SHAP-weighted training data to: {out_csv}")
