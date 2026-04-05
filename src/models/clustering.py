"""
src/models/clustering.py
------------------------
Customer segmentation using K-Means and DBSCAN.
Includes elbow method, silhouette analysis, and segment profiling.
Run: python src/models/clustering.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_raw_data
from data.preprocessor import ChurnPreprocessor, TARGET, ID_COL

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Optimal K Selection ────────────────────────────────────────────────────────
def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11)) -> int:
    """Use elbow method + silhouette scores to select best k."""
    inertias    = []
    silhouettes = []

    print("\n🔍 Finding optimal number of clusters...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels, sample_size=2000))
        print(f"   k={k:2d} | Inertia: {km.inertia_:,.0f} | Silhouette: {silhouettes[-1]:.4f}")

    best_k = k_range[int(np.argmax(silhouettes))]
    print(f"\n✅ Optimal k = {best_k} (highest silhouette score: {max(silhouettes):.4f})")
    return best_k, inertias, silhouettes, list(k_range)


# ── K-Means ────────────────────────────────────────────────────────────────────
def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """Fit K-Means and return cluster labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score  = silhouette_score(X, labels, sample_size=2000)
    print(f"\n🎯 K-Means (k={k}) | Silhouette: {score:.4f}")
    return labels, km, score


# ── DBSCAN ─────────────────────────────────────────────────────────────────────
def run_dbscan(X: np.ndarray, eps: float = 0.8, min_samples: int = 10) -> np.ndarray:
    """Fit DBSCAN and return cluster labels (-1 = noise/outliers)."""
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"\n🔵 DBSCAN | Clusters: {n_clusters} | Noise points: {n_noise} ({n_noise/len(labels):.1%})")
    if n_clusters > 1:
        non_noise = labels != -1
        score = silhouette_score(X[non_noise], labels[non_noise], sample_size=min(2000, non_noise.sum()))
        print(f"   Silhouette (non-noise): {score:.4f}")
    return labels, db


# ── Segment Profiling ──────────────────────────────────────────────────────────
SEGMENT_NAMES = {
    0: "Low-Risk Loyalists",
    1: "High-Value Engaged",
    2: "At-Risk Churners",
    3: "New Uncertain",
}

def profile_segments(df_orig: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Attach cluster labels to original (unscaled) dataframe and
    compute per-segment summary statistics.
    """
    df = df_orig.copy()
    df["segment"] = labels

    numeric_cols = ["tenure", "monthly_charges", "total_charges",
                    "support_calls", "num_products"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    profile = df.groupby("segment").agg(
        count=("segment", "count"),
        churn_rate=(TARGET, "mean") if TARGET in df.columns else ("segment", "count"),
        **{col: (col, "mean") for col in numeric_cols},
    ).round(2)

    profile["segment_name"] = profile.index.map(lambda i: SEGMENT_NAMES.get(i, f"Segment {i}"))
    print("\n📊 Segment Profile:")
    print(profile.to_string())
    return profile


# ── PCA for 2D Viz ─────────────────────────────────────────────────────────────
def reduce_to_2d(X: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


# ── Full Pipeline ──────────────────────────────────────────────────────────────
def run_clustering_pipeline():
    df = load_raw_data()

    # Preprocess (keep target for profiling)
    y = df[TARGET].values if TARGET in df.columns else None
    prep = ChurnPreprocessor(scale=True)
    X_proc = prep.fit_transform(df.drop(columns=[TARGET, ID_COL], errors="ignore"))
    X = X_proc.values

    # Find optimal k
    best_k, inertias, silhouettes, k_range = find_optimal_k(X)

    # K-Means
    km_labels, km_model, km_score = run_kmeans(X, best_k)

    # DBSCAN
    db_labels, db_model = run_dbscan(X)

    # Profile
    df_with_target = df.copy()
    profile = profile_segments(df_with_target, km_labels)

    # 2D coords for dashboard
    coords_2d = reduce_to_2d(X)

    # Save enriched dataset
    df_out = df.copy()
    df_out["kmeans_segment"]     = km_labels
    df_out["dbscan_segment"]     = db_labels
    df_out["pca_x"]              = coords_2d[:, 0]
    df_out["pca_y"]              = coords_2d[:, 1]
    df_out["segment_name"]       = pd.Series(km_labels).map(SEGMENT_NAMES).fillna("Other").values

    out_path = OUTPUT_DIR / "customers_segmented.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n💾 Segmented data saved → {out_path}")

    return df_out, profile, {
        "km_model": km_model,
        "db_model": db_model,
        "best_k":   best_k,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "k_range":  k_range,
    }


if __name__ == "__main__":
    run_clustering_pipeline()
