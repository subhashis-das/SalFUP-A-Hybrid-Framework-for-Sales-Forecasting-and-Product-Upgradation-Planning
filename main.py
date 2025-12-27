import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--meta_url", required=True)
parser.add_argument("--useVader", required=True)

args = parser.parse_args()

url = args.url
meta_url = args.meta_url
useVader = args.useVader.upper()

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=== Step 1: Preprocess Data ===")
subprocess.run(["python", "utils/data_preprocess.py", url,meta_url,useVader])

print("\n=== Step 2: Train on various Models(LSTM,ARIMA,SARIMA) ===")
os.system("python scripts/train_lstm.py data/monthly_grouped.csv results/pre_tuning")
os.system("python scripts/train_arima.py data/monthly_grouped.csv results/pre_tuning")
os.system("python scripts/train_sarima.py data/monthly_grouped.csv results/pre_tuning")
os.system("python scripts/train_sentitsmixer.py data/monthly_grouped.csv results/pre_tuning")

print("\n=== Step 3: Compute SHAP importance for features each month ===")
os.system("python scripts/shap_importance.py")

print("\n=== Step 4: Apply SHAP weights ===")
os.system("python scripts/apply_shap_weights.py")

print("\n=== Step 5: Train on various Models(LSTM,ARIMA,SARIMA) post tuning ===")
os.system("python scripts/train_lstm.py data/monthly_grouped_shap.csv results/post_tuning")
os.system("python scripts/train_arima.py data/monthly_grouped_shap.csv results/post_tuning")
os.system("python scripts/train_sarima.py data/monthly_grouped_shap.csv results/post_tuning")
os.system("python scripts/train_sentitsmixer.py data/monthly_grouped_shap.csv results/post_tuning")

print("\n=== Step 6: Find negative reviews during decline months ===")
os.system("python -u scripts/review_decline_analysis.py data/monthly_grouped.csv data/raw_reviews.csv")

print("\n=== Step 7: Find positive reviews during increasing sales months ===")
os.system("python scripts/review_growth_analysis.py data/monthly_grouped.csv data/raw_reviews.csv")

print("\n Pipeline complete! Results and model saved.")
