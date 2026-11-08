"""Weekly Demand Aggregation, Feature Engineering, and Forecasting Pipeline.

Remediated for strict QA compliance:
  1. Uses the exact same 3,271 modeling products (active days >= 10 in train) as daily pipeline.
  2. Strict non-overlapping 7-day weeks within each split (WeekEnd <= split end) — zero boundary leakage
     and no partial 5-day final week distortion.
  3. Zero future price leakage (.ffill() + shift(1) only; no .bfill() across splits).
  4. Includes TwoStageWeekly (Hurdle) and L1 HistGradientBoosting alongside Ridge, RF, and XGBoost,
     plus direct comparison against Daily TwoStageDemand predictions summed over the same test weeks.

Outputs:
  - data/model/weekly_train.parquet
  - data/model/weekly_val.parquet
  - data/model/weekly_test.parquet
  - reports/weekly_modeling_report.json
  - reports/figures/weekly_model_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_MODEL = ROOT / "data" / "model"
PRED_DIR = DATA_MODEL / "predictions"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TRAIN_END = pd.Timestamp("2010-11-30")
VAL_START = pd.Timestamp("2010-12-01")
VAL_END = pd.Timestamp("2011-02-28")
TEST_START = pd.Timestamp("2011-03-01")
MIN_ACTIVE_DAYS = 10


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, WAPE, and R2."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sum_actual = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / sum_actual) if sum_actual > 0 else float("nan")
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE": round(wape, 4),
        "R2": round(r2, 4),
    }


def main():
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    FIGURE_DIR.mkdir(exist_ok=True, parents=True)
    DATA_MODEL.mkdir(exist_ok=True, parents=True)

    print("Loading product daily demand...")
    daily_file = DATA_PROCESSED / "product_daily_demand.parquet"
    daily_df = pd.read_parquet(
        daily_file,
        columns=["StockCode", "Date", "DailyDemand", "UnitPrice", "InvoiceCount"],
    )
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])

    # 1. Filter to exact same modeling cohort (MIN_ACTIVE_DAYS >= 10 in training window)
    train_active = (
        daily_df.loc[(daily_df["Date"] <= TRAIN_END) & (daily_df["DailyDemand"] > 0)]
        .groupby("StockCode")["Date"]
        .nunique()
    )
    keep_products = set(train_active[train_active >= MIN_ACTIVE_DAYS].index)
    daily_df = daily_df[daily_df["StockCode"].isin(keep_products)].copy()
    max_date = daily_df["Date"].max()

    # Align Date to Monday of each week and keep only complete 7-day weeks
    daily_df["WeekStart"] = daily_df["Date"] - pd.to_timedelta(daily_df["Date"].dt.dayofweek, unit="D")
    daily_df["WeekEnd"] = daily_df["WeekStart"] + pd.Timedelta(days=6)
    daily_df = daily_df[daily_df["WeekEnd"] <= max_date].copy()

    print(f"Aggregating {len(keep_products)} modeled products to complete 7-day weeks...")
    weekly_df = (
        daily_df.groupby(["StockCode", "WeekStart", "WeekEnd"], as_index=False)
        .agg(
            WeeklyDemand=("DailyDemand", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            InvoiceCount=("InvoiceCount", "sum"),
            ActiveDaysInWeek=("DailyDemand", lambda x: int(np.sum(x > 0))),
        )
        .sort_values(["StockCode", "WeekStart"], kind="stable")
        .reset_index(drop=True)
    )

    # Forward-fill UnitPrice strictly forward in time (NEVER bfill across future splits!)
    weekly_df["_pf"] = weekly_df.groupby("StockCode")["UnitPrice"].ffill()
    weekly_df["PriceLag_1W"] = weekly_df.groupby("StockCode")["_pf"].shift(1)
    weekly_df.drop(columns=["_pf"], inplace=True)

    zero_ratio = float((weekly_df["WeeklyDemand"] == 0).mean())
    print(f"Weekly dataset total rows: {len(weekly_df)}, zero demand ratio: {zero_ratio * 100:.2f}%")

    # Calendar features
    weekly_df["Year"] = weekly_df["WeekStart"].dt.year
    weekly_df["Month"] = weekly_df["WeekStart"].dt.month
    weekly_df["WeekOfYear"] = weekly_df["WeekStart"].dt.isocalendar().week.astype(int)

    # Lag and rolling features (strictly shifted >= 1 week)
    grp = weekly_df.groupby("StockCode")["WeeklyDemand"]
    weekly_df["Lag_1W"] = grp.shift(1)
    weekly_df["Lag_2W"] = grp.shift(2)
    weekly_df["Lag_4W"] = grp.shift(4)
    weekly_df["Lag_8W"] = grp.shift(8)

    weekly_df["_s"] = grp.shift(1)
    s_grp = weekly_df.groupby("StockCode")["_s"]
    weekly_df["RollingMean_4W"] = s_grp.transform(lambda s: s.rolling(4, min_periods=1).mean())
    weekly_df["RollingMean_8W"] = s_grp.transform(lambda s: s.rolling(8, min_periods=1).mean())
    weekly_df["RollingStd_4W"] = s_grp.transform(lambda s: s.rolling(4, min_periods=1).std()).fillna(0.0)
    weekly_df["RollingSum_4W"] = s_grp.transform(lambda s: s.rolling(4, min_periods=1).sum())
    weekly_df.drop(columns=["_s"], inplace=True)

    weekly_df["InvoiceCountLag_1W"] = weekly_df.groupby("StockCode")["InvoiceCount"].shift(1)

    # Map ProductCluster fitted strictly on daily train
    clusters_file = DATA_PROCESSED / "product_clusters.parquet"
    if clusters_file.exists():
        clusters_df = pd.read_parquet(clusters_file).reset_index()
        cluster_map = clusters_df.set_index("StockCode")["ProductCluster"]
        weekly_df["ProductCluster"] = weekly_df["StockCode"].map(cluster_map).fillna(0).astype(int)
    else:
        weekly_df["ProductCluster"] = 0

    weekly_df = weekly_df.dropna(subset=["Lag_1W"]).copy()

    # Strict non-overlapping split boundaries: neither start nor end of a week may cross a split boundary
    train_mask = weekly_df["WeekEnd"] <= TRAIN_END
    val_mask = (weekly_df["WeekStart"] >= VAL_START) & (weekly_df["WeekEnd"] <= VAL_END)
    test_mask = weekly_df["WeekStart"] >= TEST_START

    train_df = weekly_df[train_mask].copy()
    val_df = weekly_df[val_mask].copy()
    test_df = weekly_df[test_mask].copy()

    # Compute training-only product statistics
    train_stats = train_df.groupby("StockCode").agg(
        TrainActiveWeeks=("WeeklyDemand", lambda x: int((x > 0).sum())),
        TrainTotalSold=("WeeklyDemand", "sum"),
        TrainAvgPrice=("UnitPrice", "mean"),
    )
    for frame in (train_df, val_df, test_df):
        frame["TrainActiveWeeks"] = frame["StockCode"].map(train_stats["TrainActiveWeeks"]).fillna(0)
        frame["TrainTotalSold"] = frame["StockCode"].map(train_stats["TrainTotalSold"]).fillna(0)
        frame["PriceLag_1W"] = frame["PriceLag_1W"].fillna(frame["StockCode"].map(train_stats["TrainAvgPrice"]))

    print(f"Strict Weekly Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_df.to_parquet(DATA_MODEL / "weekly_train.parquet", index=False)
    val_df.to_parquet(DATA_MODEL / "weekly_val.parquet", index=False)
    test_df.to_parquet(DATA_MODEL / "weekly_test.parquet", index=False)

    feature_cols = [
        "Month", "WeekOfYear",
        "Lag_1W", "Lag_2W", "Lag_4W", "Lag_8W",
        "RollingMean_4W", "RollingMean_8W", "RollingStd_4W", "RollingSum_4W",
        "PriceLag_1W", "InvoiceCountLag_1W",
        "TrainActiveWeeks", "TrainTotalSold",
    ]
    cat_cols = ["ProductCluster"]

    transformer = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), feature_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols),
    ], remainder="drop")

    X_train = train_df[feature_cols + cat_cols]
    y_train = train_df["WeeklyDemand"].to_numpy(dtype=np.float64)

    X_val = val_df[feature_cols + cat_cols]
    y_val = val_df["WeeklyDemand"].to_numpy(dtype=np.float64)

    X_test = test_df[feature_cols + cat_cols]
    y_test = test_df["WeeklyDemand"].to_numpy(dtype=np.float64)

    results = {}

    # 1. Baseline Lag-1W
    lag1_val_pred = val_df["Lag_1W"].to_numpy(dtype=np.float64)
    lag1_test_pred = test_df["Lag_1W"].to_numpy(dtype=np.float64)
    results["Lag1W_Baseline"] = {
        "val": evaluate_metrics(y_val, lag1_val_pred),
        "test": evaluate_metrics(y_test, lag1_test_pred),
    }

    # 2. TwoStageWeekly (Hurdle Classifier + L1 Regressor)
    print("Training TwoStageWeekly (Hurdle) on weekly dataset...")
    w_clf = make_pipeline(
        transformer,
        HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=31, l2_regularization=2.0, random_state=42),
    )
    w_reg = make_pipeline(
        transformer,
        HistGradientBoostingRegressor(loss="absolute_error", max_iter=250, max_leaf_nodes=31, l2_regularization=2.0, random_state=42),
    )
    w_clf.fit(X_train, (y_train > 0).astype(int))
    pos_mask = y_train > 0
    w_reg.fit(X_train.loc[pos_mask], y_train[pos_mask])

    p_val = w_clf.predict_proba(X_val)[:, 1]
    pos_val = np.maximum(0.0, w_reg.predict(X_val))
    best_th = 0.50
    best_th_mae = float("inf")
    for th in np.arange(0.20, 0.85, 0.05):
        cand = (p_val >= th) * pos_val
        m = evaluate_metrics(y_val, cand)
        if m["MAE"] < best_th_mae:
            best_th_mae = m["MAE"]
            best_th = round(float(th), 2)

    hurdle_val_pred = (p_val >= best_th) * pos_val
    p_test = w_clf.predict_proba(X_test)[:, 1]
    pos_test = np.maximum(0.0, w_reg.predict(X_test))
    hurdle_test_pred = (p_test >= best_th) * pos_test
    results["TwoStageWeekly"] = {
        "val": evaluate_metrics(y_val, hurdle_val_pred),
        "test": evaluate_metrics(y_test, hurdle_test_pred),
        "sale_probability_threshold": best_th,
    }

    models = {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(n_estimators=80, min_samples_leaf=5, max_features=0.8, n_jobs=1, random_state=42),
        "HistGB_L1": HistGradientBoostingRegressor(loss="absolute_error", max_iter=250, max_leaf_nodes=31, l2_regularization=2.0, random_state=42),
        "HistGB_L2": HistGradientBoostingRegressor(loss="squared_error", max_iter=200, max_leaf_nodes=31, l2_regularization=2.0, random_state=42),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=42)
    except ImportError:
        pass

    for name, est in models.items():
        print(f"Training {name} on weekly dataset...")
        pipe = make_pipeline(transformer, est)
        pipe.fit(X_train, y_train)
        val_pred = np.maximum(0.0, pipe.predict(X_val))
        test_pred = np.maximum(0.0, pipe.predict(X_test))
        results[name] = {
            "val": evaluate_metrics(y_val, val_pred),
            "test": evaluate_metrics(y_test, test_pred),
        }

    # Compare against Daily TwoStageDemand aggregated over the exact same test weeks
    daily_pred_file = PRED_DIR / "test_predictions.parquet"
    daily_agg_metrics = None
    if daily_pred_file.exists():
        d_pred = pd.read_parquet(daily_pred_file)
        d_pred["Date"] = pd.to_datetime(d_pred["Date"])
        d_pred["WeekStart"] = d_pred["Date"] - pd.to_timedelta(d_pred["Date"].dt.dayofweek, unit="D")
        d_weekly = (
            d_pred.groupby(["StockCode", "WeekStart"], as_index=False)
            .agg(DailySumPred=("PredictedDemand", "sum"))
        )
        merged_test = test_df[["StockCode", "WeekStart", "WeeklyDemand"]].merge(
            d_weekly, on=["StockCode", "WeekStart"], how="left"
        )
        daily_agg_metrics = evaluate_metrics(
            merged_test["WeeklyDemand"].to_numpy(dtype=np.float64),
            merged_test["DailySumPred"].fillna(0.0).to_numpy(dtype=np.float64),
        )
        results["Daily_TwoStage_SummedToWeekly"] = {
            "val": {},
            "test": daily_agg_metrics,
        }

    candidate_models = [m for m in results.keys() if m not in ("Lag1W_Baseline", "Daily_TwoStage_SummedToWeekly")]
    best_model_name = min(candidate_models, key=lambda m: results[m]["val"]["WAPE"])

    report = {
        "description": "Weekly demand forecasting comparison on identical 3,271-product cohort and strict non-overlapping 7-day weeks",
        "weekly_dataset_summary": {
            "modeled_products": len(keep_products),
            "total_rows": len(weekly_df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_week_range": [str(train_df["WeekStart"].min().date()), str(train_df["WeekEnd"].max().date())],
            "val_week_range": [str(val_df["WeekStart"].min().date()), str(val_df["WeekEnd"].max().date())],
            "test_week_range": [str(test_df["WeekStart"].min().date()), str(test_df["WeekEnd"].max().date())],
            "zero_demand_percentage": round(float(zero_ratio * 100), 2),
            "test_zero_demand_percentage": round(float((test_df["WeeklyDemand"] == 0).mean() * 100), 2),
        },
        "selected_best_model_by_val_wape": best_model_name,
        "models_performance": results,
        "qa_remediations_applied": [
            "Filtered to the exact 3,271 products with >=10 active training days for apples-to-apples comparison with daily models.",
            "Eliminated boundary-week leakage by requiring complete 7-day weeks strictly inside each split window.",
            "Removed .bfill() on UnitPrice to prevent future price leakage into training weeks.",
            "Removed raw Year feature to avoid out-of-range extrapolation drift on 2011 test data.",
        ],
    }

    report_file = REPORT_DIR / "weekly_modeling_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Weekly report saved to {report_file}")

    plot_models = [m for m in results.keys() if results[m]["test"]]
    test_wapes = [results[m]["test"]["WAPE"] for m in plot_models]
    test_r2s = [results[m]["test"]["R2"] for m in plot_models]
    test_maes = [results[m]["test"]["MAE"] for m in plot_models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = ["#94a3b8", "#10b981", "#38bdf8", "#818cf8", "#34d399", "#f59e0b", "#ec4899", "#6366f1"][:len(plot_models)]

    axes[0].bar(plot_models, test_wapes, color=colors)
    axes[0].set_title("Weekly Test WAPE (Lower is Better)")
    axes[0].set_ylabel("WAPE")
    axes[0].axhline(1.0, color="r", linestyle="--", alpha=0.6)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(plot_models, test_r2s, color=colors)
    axes[1].set_title("Weekly Test R2 (Higher is Better)")
    axes[1].set_ylabel("R2 Score")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(plot_models, test_maes, color=colors)
    axes[2].set_title("Weekly Test MAE (Lower is Better)")
    axes[2].set_ylabel("MAE")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig_path = FIGURE_DIR / "weekly_model_comparison.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
