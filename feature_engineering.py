"""Feature engineering and dataset preparation for demand forecasting.

Implements KPDL.md §1.3 (biến đổi dữ liệu) and §4 (phân chia tập dữ liệu).

Reads
-----
  data/processed/product_daily_demand.parquet

Writes
------
  data/model/train.parquet      — Training   (2009-12-01 → 2010-11-30)
  data/model/val.parquet        — Validation (2010-12-01 → 2011-02-28)
  data/model/test.parquet       — Test       (2011-03-01 → 2011-12-09)
  data/model/features_full.parquet
  reports/feature_engineering_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import holidays as _holidays_lib

    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "processed" / "product_daily_demand.parquet"
OUTPUT_DIR = ROOT / "data" / "model"
REPORT_DIR = ROOT / "reports"

# §4 — Chronological split boundaries.
TRAIN_END = pd.Timestamp("2010-11-30")
VAL_END = pd.Timestamp("2011-02-28")

# Products with fewer active selling days in training are dropped (§4).
MIN_ACTIVE_DAYS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _describe(series: pd.Series) -> dict:
    """Concise distribution summary for a numeric series."""
    d = series.describe()
    return {k: round(float(d[k]), 4) for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}


# ---------------------------------------------------------------------------
# Step 1 — UK holiday flag  (§1.1.4 & §1.3)
# ---------------------------------------------------------------------------


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Mark UK bank holidays."""
    if not _HAS_HOLIDAYS:
        print("⚠ 'holidays' package not installed — IsHoliday set to 0.")
        df["IsHoliday"] = np.int8(0)
        return df

    years = sorted(int(y) for y in df["Date"].dt.year.unique())
    uk = _holidays_lib.UK(years=years)
    holiday_dates = set(uk.keys())
    df["IsHoliday"] = df["Date"].dt.date.isin(holiday_dates).astype("int8")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Low-frequency product filter  (§4)
# ---------------------------------------------------------------------------


def filter_low_frequency(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove products with too few active selling days in the training window."""
    train_active = (
        df.loc[(df["Date"] <= TRAIN_END) & (df["DailyDemand"] > 0)]
        .groupby("StockCode")["Date"]
        .nunique()
    )
    keep = train_active[train_active >= MIN_ACTIVE_DAYS].index

    n_before = int(df["StockCode"].nunique())
    rows_before = len(df)
    df = df.loc[df["StockCode"].isin(keep)].copy()
    n_after = int(df["StockCode"].nunique())

    info = {
        "products_before": n_before,
        "products_after": n_after,
        "products_removed": n_before - n_after,
        "rows_before": rows_before,
        "rows_after": len(df),
        "min_active_days_threshold": MIN_ACTIVE_DAYS,
    }
    return df, info


# ---------------------------------------------------------------------------
# Step 3 — Outlier detection  (§4)
# ---------------------------------------------------------------------------


def flag_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag per-product outlier demand days using IQR on positive demand."""

    def _iqr(s: pd.Series) -> pd.Series:
        pos = s[s > 0]
        if len(pos) < 4:
            return pd.Series(np.int8(0), index=s.index)
        q1, q3 = pos.quantile(0.25), pos.quantile(0.75)
        return ((s > q3 + 1.5 * (q3 - q1)) & (s > 0)).astype("int8")

    df["IsOutlierDemand"] = df.groupby("StockCode")["DailyDemand"].transform(_iqr)
    n = int(df["IsOutlierDemand"].sum())
    info = {
        "method": "per-product IQR (Q3 + 1.5 × IQR) on positive demand",
        "outlier_rows": n,
        "outlier_pct": round(n / len(df) * 100, 4),
    }
    return df, info


# ---------------------------------------------------------------------------
# Step 4 — History features  (§1.3 — đặc trưng lịch sử)
# ---------------------------------------------------------------------------


def build_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, rolling, and recency features.

    Every feature uses ``shift(≥1)`` so only past data is visible — no leakage.
    """
    g = "StockCode"
    demand = df.groupby(g)["DailyDemand"]

    # Lag features
    df["Lag_1"] = demand.shift(1)
    df["Lag_7"] = demand.shift(7)
    df["Lag_14"] = demand.shift(14)
    df["Lag_28"] = demand.shift(28)

    # Rolling features on the shifted series (shift prevents leakage)
    df["_s"] = demand.shift(1)
    grp = df.groupby(g)["_s"]
    for w in (7, 14, 28):
        df[f"RollingMean_{w}"] = grp.transform(lambda x, _w=w: x.rolling(_w, min_periods=1).mean())
    df["RollingSum_7"] = grp.transform(lambda x: x.rolling(7, min_periods=1).sum())
    df["RollingStd_28"] = grp.transform(lambda x: x.rolling(28, min_periods=1).std())
    df.drop(columns="_s", inplace=True)

    # Days since last sale (ngày không bán gần nhất)
    df["_sd"] = df["Date"].where(df["DailyDemand"] > 0)
    df["_ls"] = df.groupby(g)["_sd"].transform(lambda x: x.ffill().shift(1))
    df["DaysSinceLastSale"] = (df["Date"] - df["_ls"]).dt.days
    df.drop(columns=["_sd", "_ls"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Step 5 — Price features  (§1.3 — đặc trưng giá)
# ---------------------------------------------------------------------------


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged price level and day-over-day price change.

    Price is forward-filled across zero-sale days, then shifted by 1 day so
    only yesterday's price is visible.
    """
    g = "StockCode"
    df["_pf"] = df.groupby(g)["UnitPrice"].transform("ffill")
    df["PriceLag_1"] = df.groupby(g)["_pf"].shift(1)
    _lag2 = df.groupby(g)["_pf"].shift(2)
    df["PriceChange"] = df["PriceLag_1"] - _lag2
    df.drop(columns="_pf", inplace=True)
    # Contemporaneous price and invoice count are only known after the target
    # day closes; expose their prior-day values as predictive inputs.
    df["InvoiceCountLag_1"] = df.groupby(g)["InvoiceCount"].shift(1)
    return df


# ---------------------------------------------------------------------------
# Step 6 — Target transforms  (§1.3 — biến đổi biến đích)
# ---------------------------------------------------------------------------


def transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """Log1p and sqrt transforms for the right-skewed DailyDemand."""
    df["DailyDemand_Log"] = np.log1p(df["DailyDemand"])
    df["DailyDemand_Sqrt"] = np.sqrt(df["DailyDemand"])
    # Descriptive classes are useful for EDA, but must never be model inputs.
    positive = df.loc[df["DailyDemand"] > 0, "DailyDemand"]
    cutoff = float(positive.median()) if not positive.empty else 0.0
    df["DailyDemand_Category"] = np.select(
        [df["DailyDemand"].eq(0), df["DailyDemand"].le(cutoff)],
        [0, 1], default=2,
    ).astype("int8")
    df["DailyDemand_IsHigh"] = df["DailyDemand"].gt(cutoff).astype("int8")
    return df


# ---------------------------------------------------------------------------
# Step 7 — Product-level features from training data  (§1.3 — sản phẩm)
# ---------------------------------------------------------------------------


def build_product_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate product features computed exclusively from the training set
    to prevent information leakage into validation / test."""
    stats = train.groupby("StockCode").agg(
        TotalActiveDays=("DailyDemand", lambda x: int((x > 0).sum())),
        TotalQuantitySold=("DailyDemand", "sum"),
        AvgUnitPrice=("UnitPrice", "mean"),
    )
    for frame in (train, val, test):
        frame["TotalActiveDays"] = frame["StockCode"].map(stats["TotalActiveDays"])
        frame["TotalQuantitySold"] = frame["StockCode"].map(stats["TotalQuantitySold"])
        frame["AvgUnitPrice"] = frame["StockCode"].map(stats["AvgUnitPrice"])
    # Product clusters are fitted from training-only aggregates, then mapped
    # to all partitions. Keep the estimator implementation local to avoid
    # making preprocessing depend on a separately executed script.
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    cluster_cols = ["AvgDailyDemand", "ActiveDays", "AvgUnitPrice", "DemandStd"]
    cluster_stats = train.groupby("StockCode").agg(
        AvgDailyDemand=("DailyDemand", "mean"),
        ActiveDays=("DailyDemand", lambda x: int((x > 0).sum())),
        AvgUnitPrice=("UnitPrice", "mean"),
        DemandStd=("DailyDemand", "std"),
    ).fillna(0.0)
    if len(cluster_stats) >= 3:
        scaler = StandardScaler()
        x = scaler.fit_transform(cluster_stats[cluster_cols])
        model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(x)
        order = (cluster_stats.assign(_label=model.labels_).groupby("_label")["AvgDailyDemand"]
                 .mean().sort_values().index.tolist())
        remap = {old: new for new, old in enumerate(order)}
        cluster_map = pd.Series([remap[int(label)] for label in model.labels_], index=cluster_stats.index)
        for frame in (train, val, test):
            frame["ProductCluster"] = frame["StockCode"].map(cluster_map).astype("Int8")
    else:
        for frame in (train, val, test):
            frame["ProductCluster"] = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("1/8  Loading daily demand data …")
    df = pd.read_parquet(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["StockCode", "Date"], kind="stable").reset_index(drop=True)

    print("2/8  Adding UK holiday flag …")
    df = add_holiday_flag(df)

    print("3/8  Filtering low-frequency products …")
    df, low_freq_info = filter_low_frequency(df)

    print("4/8  Flagging outlier demand …")
    df, outlier_info = flag_outliers(df)

    print("5/8  Building history features (lag, rolling, recency) …")
    df = build_history_features(df)

    print("6/8  Building price features …")
    df = build_price_features(df)

    print("7/8  Transforming target variable …")
    df = transform_target(df)

    # §4 — Chronological split
    print("8/8  Splitting train / val / test …")
    train = df[df["Date"] <= TRAIN_END].copy()
    val = df[(df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END)].copy()
    test = df[df["Date"] > VAL_END].copy()

    print("     Building product-level features from training data …")
    train, val, test = build_product_features(train, val, test)

    # ---- Save outputs ----
    print("     Saving datasets …")
    df.to_parquet(OUTPUT_DIR / "features_full.parquet", index=False)
    train.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    test.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    # ---- Report ----
    report = {
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "low_frequency_filter": low_freq_info,
        "outlier_detection": outlier_info,
        "split": {
            "train": {
                "rows": len(train),
                "date_range": [str(train["Date"].min().date()), str(train["Date"].max().date())],
            },
            "val": {
                "rows": len(val),
                "date_range": [str(val["Date"].min().date()), str(val["Date"].max().date())],
            },
            "test": {
                "rows": len(test),
                "date_range": [str(test["Date"].min().date()), str(test["Date"].max().date())],
            },
        },
        "demand_distribution_by_split": {
            "train": _describe(train["DailyDemand"]),
            "val": _describe(val["DailyDemand"]),
            "test": _describe(test["DailyDemand"]),
        },
        "feature_columns": sorted(c for c in train.columns if c not in (
            "StockCode", "Date", "DailyDemand", "DailyDemand_Log", "DailyDemand_Sqrt",
            "DailyDemand_Category", "DailyDemand_IsHigh", "IsOutlierDemand",
            "UnitPrice", "InvoiceCount",
        )),
        "holiday_source": "holidays.UK" if _HAS_HOLIDAYS else "unavailable",
        "notes": [
            "All lag and rolling features use shift(≥1) to prevent data leakage.",
            "Product-level features are computed from training data only.",
            "UnitPrice is forward-filled across zero-sale days before computing price features.",
            "DailyDemand_Log uses log1p(x) = ln(1+x) to handle zero demand.",
            "ProductCluster is fitted from training-only product aggregates.",
            "Contemporaneous UnitPrice and InvoiceCount are retained for reporting but excluded from forecasting predictors.",
            "Target-derived transformations and outlier flags are excluded from model feature_columns.",
        ],
    }

    report_path = REPORT_DIR / "feature_engineering_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "feature_count": len(report["feature_columns"]),
        "low_freq_products_removed": low_freq_info["products_removed"],
        "outlier_rows_flagged": outlier_info["outlier_rows"],
    }
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nReport: {report_path.relative_to(ROOT)}")
    print("Done ✓")


if __name__ == "__main__":
    main()
