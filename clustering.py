"""Product clustering module using K-Means.

Implements KPDL.md §3 (Khai phá dữ liệu - Phân nhóm sản phẩm bằng K-Means)
and supports §1.3 (Đặc trưng nhóm sản phẩm từ K-Means).

Features aggregated per product strictly on training set (Date <= TRAIN_END):
  1. AvgDailyDemand: Trung bình số lượng bán trong ngày
  2. ActiveDays:     Tần suất giao dịch (số ngày phát sinh giao dịch)
  3. AvgUnitPrice:   Mức giá trung bình của sản phẩm
  4. DemandStd:      Độ biến động số lượng bán

Evaluation:
  - Elbow method (Inertia / WCSS) for k = 2..8
  - Silhouette score for k = 2..8

Outputs:
  - data/processed/product_clusters.parquet (and .csv)
  - reports/clustering_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
TRAIN_FILE = ROOT / "data" / "model" / "train.parquet"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

CLUSTER_NAMES_K3 = {
    0: "Phổ thông (Nhu cầu thấp - vừa)",
    1: "Bán chạy (Nhu cầu trung bình - cao, ổn định)",
    2: "Chủ lực (Nhu cầu đột biến, số lượng lớn)",
}


def build_clustering_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute product-level summary features strictly on training data."""
    stats = (
        train_df.groupby("StockCode")
        .agg(
            AvgDailyDemand=("DailyDemand", "mean"),
            ActiveDays=("DailyDemand", lambda x: int((x > 0).sum())),
            AvgUnitPrice=("UnitPrice", "mean"),
            DemandStd=("DailyDemand", "std"),
        )
        .fillna(0.0)
    )
    return stats


def evaluate_k_range(
    X_scaled: np.ndarray, k_range: range = range(2, 9)
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute inertia (Elbow) and silhouette scores across k range."""
    inertias: dict[int, float] = {}
    silhouettes: dict[int, float] = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = round(float(km.inertia_), 2)
        silhouettes[k] = round(float(silhouette_score(X_scaled, labels)), 4)

    return inertias, silhouettes


def fit_product_kmeans(
    prod_features: pd.DataFrame, n_clusters: int = 3
) -> tuple[pd.DataFrame, KMeans, StandardScaler, dict[str, Any]]:
    """Fit K-Means clustering and produce labeled product DataFrame and evaluation report."""
    feature_cols = ["AvgDailyDemand", "ActiveDays", "AvgUnitPrice", "DemandStd"]
    X = prod_features[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias, silhouettes = evaluate_k_range(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Sort cluster labels by AvgDailyDemand ascending for intuitive ordering (0: lowest, 2: highest)
    temp = prod_features.copy()
    temp["_raw_cluster"] = labels
    cluster_order = (
        temp.groupby("_raw_cluster")["AvgDailyDemand"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    remap = {old_c: new_c for new_c, old_c in enumerate(cluster_order)}
    final_labels = np.array([remap[lbl] for lbl in labels], dtype=np.int32)

    result_df = prod_features.copy()
    result_df["ProductCluster"] = final_labels
    result_df["ClusterName"] = result_df["ProductCluster"].map(
        lambda c: CLUSTER_NAMES_K3.get(c, f"Cluster {c}")
    )

    # Build cluster profile summary
    cluster_profiles = {}
    for c in range(n_clusters):
        subset = result_df[result_df["ProductCluster"] == c]
        cluster_profiles[f"cluster_{c}"] = {
            "name": CLUSTER_NAMES_K3.get(c, f"Cluster {c}"),
            "product_count": int(len(subset)),
            "pct_of_total": round(float(len(subset) / len(result_df) * 100), 2),
            "metrics": {
                col: {
                    "mean": round(float(subset[col].mean()), 4),
                    "std": round(float(subset[col].std()), 4),
                    "min": round(float(subset[col].min()), 4),
                    "max": round(float(subset[col].max()), 4),
                }
                for col in feature_cols
            },
        }

    report = {
        "n_products": int(len(result_df)),
        "feature_columns": feature_cols,
        "selected_k": n_clusters,
        "elbow_inertias": {str(k): v for k, v in inertias.items()},
        "silhouette_scores": {str(k): v for k, v in silhouettes.items()},
        "selected_silhouette": silhouettes.get(n_clusters),
        "cluster_profiles": cluster_profiles,
        "notes": [
            "Features are aggregated strictly on the training partition (Date <= 2010-11-30) to prevent leakage.",
            "Features were standardized using StandardScaler prior to K-Means.",
            "Clusters are ordered monotonically by mean AvgDailyDemand.",
        ],
    }

    return result_df, kmeans, scaler, report


def run_clustering() -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training set not found: {TRAIN_FILE}. Run feature_engineering.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {TRAIN_FILE.name} …")
    train_df = pd.read_parquet(TRAIN_FILE)

    print("Computing product aggregation features …")
    prod_features = build_clustering_features(train_df)

    print("Evaluating K-Means (Elbow & Silhouette) for k in [2..8] …")
    clustered_df, _, _, report = fit_product_kmeans(prod_features, n_clusters=3)

    out_parquet = OUTPUT_DIR / "product_clusters.parquet"
    out_csv = OUTPUT_DIR / "product_clusters.csv"
    clustered_df.to_parquet(out_parquet)
    clustered_df.to_csv(out_csv)

    report_path = REPORT_DIR / "clustering_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nClustering completed successfully!")
    print(f"Saved: {out_parquet.relative_to(ROOT)}")
    print(f"Saved: {out_csv.relative_to(ROOT)}")
    print(f"Report: {report_path.relative_to(ROOT)}")
    print(f"Selected k = {report['selected_k']}, Silhouette = {report['selected_silhouette']:.4f}")
    for c_id, profile in report["cluster_profiles"].items():
        print(f"  [{c_id}] {profile['name']}: {profile['product_count']} products ({profile['pct_of_total']}%)")

    return clustered_df


if __name__ == "__main__":
    run_clustering()
