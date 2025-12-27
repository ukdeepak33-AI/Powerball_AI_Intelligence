# ml/train_patterns.py

import os
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------- CONFIG ----------------
MODEL_VERSION = "v1"
N_CLUSTERS = 6
RANDOM_STATE = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", MODEL_VERSION)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "powerball_draws.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["draw_date"])

# ---------------- FREQUENCY & GAP ----------------
def build_freq_gap(df):
    freq = defaultdict(int)
    last_seen = {}

    df = df.sort_values("draw_date")

    for _, row in df.iterrows():
        for i in range(1, 6):
            n = row[f"n{i}"]
            freq[n] += 1
            last_seen[n] = row["draw_date"]

    max_date = df["draw_date"].max()
    gap = {n: (max_date - last_seen[n]).days for n in freq}
    return freq, gap

# ---------------- FEATURES ----------------
def extract_features(row, freq, gap):
    white = sorted([row[f"n{i}"] for i in range(1, 6)])
    return {
        "sum": sum(white),
        "range": max(white) - min(white),
        "mean": np.mean(white),
        "std": np.std(white),
        "odd_count": sum(n % 2 for n in white),
        "even_count": 5 - sum(n % 2 for n in white),
        "low_count": sum(n <= 35 for n in white),
        "high_count": sum(n > 35 for n in white),
        "avg_freq": np.mean([freq[n] for n in white]),
        "max_freq": max(freq[n] for n in white),
        "avg_gap": np.mean([gap[n] for n in white]),
        "max_gap": max(gap[n] for n in white),
        "powerball": row["powerball"]
    }

# ---------------- LABELS ----------------
def label_draw(f):
    if f["sum"] < 130:
        return "low_sum"
    if f["sum"] > 190:
        return "high_sum"
    if f["odd_count"] >= 4:
        return "odd_heavy"
    if f["even_count"] >= 4:
        return "even_heavy"
    if f["avg_gap"] > 20:
        return "cold_numbers"
    return "balanced"

# ---------------- TRAIN ----------------
def train():
    print("📥 Loading data")
    df = load_data()

    freq, gap = build_freq_gap(df)

    print("🧮 Feature engineering")
    rows, labels = [], []

    for _, row in df.iterrows():
        f = extract_features(row, freq, gap)
        rows.append(f)
        labels.append(label_draw(f))

    X = pd.DataFrame(rows)
    y = pd.Series(labels)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print("🤖 Training classifiers")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE
    )
    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

    rf.fit(Xs, y)
    gb.fit(Xs, y)

    print("🔎 Training KNN")
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(Xs)

    print("🧠 Clustering")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(Xs)
    print("Silhouette:", silhouette_score(Xs, clusters))

    print("💾 Saving models")
    joblib.dump(rf, f"{MODEL_DIR}/rf.joblib")
    joblib.dump(gb, f"{MODEL_DIR}/gb.joblib")
    joblib.dump(knn, f"{MODEL_DIR}/knn.joblib")
    joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

    print("✅ Training complete")

if __name__ == "__main__":
    train()
