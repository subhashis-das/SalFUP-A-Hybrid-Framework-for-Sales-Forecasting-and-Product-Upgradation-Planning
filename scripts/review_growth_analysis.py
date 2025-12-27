"""
Positive review analysis using parallel spaCy

Usage:
    python scripts/review_increase_analysis.py <monthly_grouped.csv> <raw_reviews.csv>
"""

import multiprocessing
import os
import sys
import re
import pandas as pd
import contractions
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from tqdm import tqdm
from rapidfuzz import fuzz
from spellchecker import SpellChecker

import spacy
import nltk
from nltk.corpus import wordnet as wn

# ---------------- Config ----------------
EXCLUDED_SUBJECTS = {"t", "beauty", "d", "cr", "tv", "gq", ""}
POSITIVE_THRESHOLD = 0.3
TOP_N_PER_MONTH = 100
FUZZY_THRESHOLD = 90

FAMILY_TERMS = {
    "husband","wife","son","daughter","child","kid","kids",
    "father","mother","parent","brother","sister","sibling",
    "grandfather","grandmother","grandparent","uncle","aunt",
    "family","relative"
}

ROLE_TERMS = {
    "person","people","someone","anyone","everyone",
    "customer","buyer","user","reader","client",
    "employee","staff","worker","team",
    "man","woman","men","women","boy","girl"
}

PHRASE_PATTERNS = [
    ("fast", "delivery"), ("great", "service"),
    ("friendly", "staff"), ("easy", "refund"),
    ("easy", "payment"), ("quick", "response")
]

SYNONYM_MAP = {
    "fast delivery": "fast delivery",
    "quick response": "quick response",
    "great service": "good service",
    "friendly staff": "friendly staff",
    "easy refund": "smooth refund",
    "easy payment": "smooth payment"
}

# ---------------- NLP setup ----------------
nltk.download("wordnet", quiet=True)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
nlp.enable_pipe("lemmatizer")
sia = SentimentIntensityAnalyzer()

# ---------------- Caches ----------------
human_cache = {}
generic_cache = {}

# ---------------- Helpers ----------------
def normalize(word):
    return re.sub(r"s$", "", re.sub(r"'s$", "", word.lower()))

def is_human_noun(word):
    if word in human_cache:
        return human_cache[word]
    synsets = wn.synsets(word, pos=wn.NOUN)
    result = any(
        h.name().startswith("person.n.01")
        for s in synsets
        for path in s.hypernym_paths()
        for h in path
    )
    human_cache[word] = result
    return result

def is_generic(word):
    if word in generic_cache:
        return generic_cache[word]
    synsets = wn.synsets(word, pos=wn.NOUN)
    result = synsets and min(s.min_depth() for s in synsets) < 6
    generic_cache[word] = result
    return result

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    try:
        text = contractions.fix(text)
    except Exception:
        pass
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-zA-Z0-9 .,!?]', ' ', text)
    return text.lower().strip()

def extract_phrase(token):
    mods = [t.text.lower() for t in token.lefts if t.dep_ in ("amod", "compound")]
    return " ".join(mods + [token.text.lower()])

def normalize_praise(phrase):
    for w1, w2 in PHRASE_PATTERNS:
        if w1 in phrase and w2 in phrase:
            phrase = f"{w1} {w2}"
    return SYNONYM_MAP.get(phrase, phrase)

def extract_subjects(texts):
    praises = []

    docs = nlp.pipe(
        texts,
        batch_size=64,
        n_process=max(1, multiprocessing.cpu_count()-60)
    )

    for doc in tqdm(
        docs,
        total=len(texts),
        desc="Extracting subjects",
        smoothing=0.1,
        disable=False,
        dynamic_ncols=True
    ):
        found = None
        for tok in doc:
            if tok.pos_ == "NOUN":
                lemma = normalize(tok.lemma_)
                if (
                    lemma in EXCLUDED_SUBJECTS
                    or lemma in FAMILY_TERMS
                    or lemma in ROLE_TERMS
                    or len(lemma) <= 2
                    or is_human_noun(lemma)
                    or is_generic(lemma)
                ):
                    continue
                found = normalize_praise(extract_phrase(tok))
                break

        praises.append(found)

    return praises

# ---------------- Main ----------------
if len(sys.argv) < 3:
    print("Usage: python review_increase_analysis.py monthly.csv raw.csv")
    sys.exit(1)

monthly = pd.read_csv(sys.argv[1], parse_dates=["month"])
raw = pd.read_csv(sys.argv[2], parse_dates=["timestamp"])

monthly["sales_diff"] = monthly["sales"].diff()
increase_months = set(monthly.loc[monthly["sales_diff"] > 0, "month"])

raw["month"] = raw["timestamp"].dt.to_period("M").dt.to_timestamp()
increase_reviews = raw[raw["month"].isin(increase_months)]

pos = increase_reviews[
    (increase_reviews["review_text_sentiment"] >= POSITIVE_THRESHOLD) |
    (increase_reviews["review_title_sentiment"] >= POSITIVE_THRESHOLD)
]

print(f"Processing {len(pos)} positive reviews...")

texts = [clean_text(t) for t in tqdm(pos["review_text"], desc="Cleaning text")]
subjects = extract_subjects(texts)
subjects = [s for s in subjects if s is not None]

praise_df = pd.DataFrame({
    "month": pos["month"].values[:len(subjects)],
    "praise": subjects
})

ranked = (
    praise_df.groupby(["month", "praise"])
    .size()
    .reset_index(name="frequency")
    .sort_values(["month", "frequency"], ascending=[True, False])
)

# ---------------- Spell correction ONLY on top praises ----------------
top = ranked.groupby("month").head(TOP_N_PER_MONTH).copy()
spell = SpellChecker()
top = top[top["praise"].notna() & (top["praise"].str.strip() != "")]
unique = top["praise"].unique()

def correct_phrase(phrase):
    if not phrase or not isinstance(phrase, str):
        return ""
    corrected_words = []
    for w in phrase.split():
        if w.isalpha() and len(w) > 3:
            c = spell.correction(w)
            if c is None:
                c = w
            corrected_words.append(c)
        else:
            corrected_words.append(w)
    return " ".join(corrected_words)

correction_map = {c: correct_phrase(c) for c in tqdm(unique, desc="Spell-correcting")}
top["praise_corrected"] = top["praise"].map(correction_map)

# ---------------- Fuzzy merge ----------------
def canonicalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

top["praise_norm"] = top["praise_corrected"].map(canonicalize)
labels = top["praise_norm"].unique()

cluster_map = {}
seen = set()
for c in labels:
    if c in seen:
        continue
    cluster = [c]
    seen.add(c)
    for o in labels:
        if o not in seen and fuzz.token_sort_ratio(c, o) >= FUZZY_THRESHOLD:
            cluster.append(o)
            seen.add(o)
    root = min(cluster, key=len)
    for k in cluster:
        cluster_map[k] = root

top["praise_root"] = top["praise_norm"].map(cluster_map)

final_ranked = (
    top.groupby(["month", "praise_root"])["frequency"]
    .sum()
    .reset_index()
    .sort_values(["month", "frequency"], ascending=[True, False])
)

# ---------------- Save outputs ----------------
os.makedirs("results", exist_ok=True)
ranked.to_csv("results/praise_ranking_by_month.csv", index=False)
final_ranked.to_csv("results/praise_ranking_by_month_root_causes.csv", index=False)
pd.DataFrame(subjects, columns=["subject"]).to_csv("results/positive_subjects_wordnet.csv", index=False)
print("Saved CSV outputs.")

# ---------------- Wordcloud ----------------
wc_text = " ".join(final_ranked["praise_root"].tolist())

if wc_text:
    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap="Greens"
    ).generate(wc_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Root Praise Topics During Increase Months")
    plt.savefig("results/praise_wordcloud_root_causes.png", dpi=200)
    plt.close()
    print("Saved wordcloud.")
