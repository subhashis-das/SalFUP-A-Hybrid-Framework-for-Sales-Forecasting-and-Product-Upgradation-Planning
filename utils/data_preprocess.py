import os
import gzip
import json
import re
import sys
import numpy as np
import pandas as pd
import swifter
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
from tqdm.auto import tqdm
from multiprocessing import Pool



def download_file(url, path):
    """Download file only if not already present"""
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    else:
        print(f"{path} exists, skipping download.")

def read_jsonl_gzip(path):
    """Read gzipped JSONL into DataFrame"""
    data = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return pd.DataFrame(data)

def load_and_prepare_data(url,meta_url,useVader):
    """Preprocess Amazon reviews into monthly and raw-level CSVs"""
    os.makedirs("data", exist_ok=True)
    monthly_csv = "data/monthly_grouped.csv"
    raw_csv = "data/raw_reviews.csv"

    # --- Load cached if present ---
    if os.path.exists(monthly_csv) and os.path.exists(raw_csv):
        print(f"Loaded cached CSVs: {monthly_csv}, {raw_csv}")
        return pd.read_csv(monthly_csv, parse_dates=['month'])

    print("Preprocessing data (first-time only)...")

    # --- Download review + meta files ---
    raw_path, meta_path = "data/review_data.jsonl.gz", "data/meta_review_data.jsonl.gz"
    download_file(url, raw_path)
    download_file(meta_url, meta_path)

    # --- Load & merge ---
    df = read_jsonl_gzip(raw_path)
    meta = read_jsonl_gzip(meta_path)
    merged = pd.merge(df, meta, on='parent_asin', how='left')

    merged = merged.rename(columns={
        'title_x': 'review_title',
        'title_y': 'product_title',
        'text': 'review_text'
    })
    merged['timestamp'] = pd.to_datetime(merged['timestamp'], unit='ms', errors='coerce')
    merged = merged.dropna(subset=['timestamp'])
    merged = merged.set_index('timestamp')

    # --- Text cleaning + TF-IDF + Vader sentiment ---
    merged['review_title'] = merged['review_title'].fillna('').astype(str)
    merged['review_text'] = merged['review_text'].fillna('').astype(str)
    clean = lambda x: re.sub(r'[^a-zA-Z ]', '', x.lower())

    print("Computing TF-IDF features...")
    title_clean = merged['review_title'].swifter.apply(clean)
    tfidf_title = TfidfVectorizer(max_features=500, stop_words='english')
    merged['review_title_tfidf'] = tfidf_title.fit_transform(title_clean).mean(axis=1).A1

    text_clean = merged['review_text'].swifter.apply(clean)
    tfidf_text = TfidfVectorizer(max_features=1000, stop_words='english')
    merged['review_text_tfidf'] = tfidf_text.fit_transform(text_clean).mean(axis=1).A1

    if(useVader):
        print("Calculating Vader sentiment for title and text...")
        sia = SentimentIntensityAnalyzer()
        merged['review_title_sentiment'] = merged['review_title'].swifter.apply(lambda t: sia.polarity_scores(t)['compound'])
        merged['review_text_sentiment'] = merged['review_text'].swifter.apply(lambda t: sia.polarity_scores(t)['compound'])
    else:
        print("Calculating BERT sentiment for title and text...")
        device = 0 if torch.cuda.is_available() else -1
        bert_sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            truncation=True,
            padding=True,
            max_length=256
        )
        def bert_batch(texts, batch_size=32):
            scores = []
            for i in tqdm(range(0, len(texts), batch_size), desc="BERT sentiment"):
                batch = texts[i:i+batch_size]
                results = bert_sentiment(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=256
                    )
                for r in results:
                    scores.append(
                        r["score"] if r["label"] == "POSITIVE" else -r["score"]
                    )
                
            return scores

        all_texts = (
            merged['review_title'].fillna("").tolist()
            + merged['review_text'].fillna("").tolist()
        )

        scores = bert_batch(all_texts, batch_size=64)

        n = len(merged)
        merged['review_title_sentiment'] = scores[:n]
        merged['review_text_sentiment'] = scores[n:]

    # --- Save raw-level CSV (includes sentiments) ---
    merged_reset = merged.reset_index()
    raw_cols = [
        'timestamp', 'review_title', 'review_text',
        'rating', 'helpful_vote', 'verified_purchase',
        'review_title_tfidf', 'review_text_tfidf',
        'review_title_sentiment', 'review_text_sentiment'
    ]
    merged_reset[raw_cols].to_csv(raw_csv, index=False)
    print(f"Saved raw-level data (with sentiment) to: {raw_csv}")

    # --- Group monthly ---
    print("Aggregating monthly data...")
    merged['month'] = merged.index.to_period('M').to_timestamp()
    monthly_grouped = merged.groupby('month').agg({
        'rating': 'mean',
        'helpful_vote': 'sum',
        'verified_purchase': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'review_title_tfidf': 'mean',
        'review_text_tfidf': 'mean',
        'review_title_sentiment': 'mean',
        'review_text_sentiment': 'mean',
        'review_title': 'count'  # monthly review count proxy = sales
    }).rename(columns={'review_title': 'sales'}).reset_index()

    # --- Save monthly CSV ---
    monthly_grouped.to_csv(monthly_csv, index=False)
    print(f"Saved monthly grouped data to: {monthly_csv}")

    return monthly_grouped

if __name__ == "__main__":
    url = sys.argv[1]
    meta_url = sys.argv[2]
    useVader = sys.argv[3]
    useVader = True if useVader == 'T' else False
    load_and_prepare_data(url,meta_url,useVader=useVader)
