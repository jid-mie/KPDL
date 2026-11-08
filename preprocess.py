

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "online_retail_II.xlsx"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country",
]


class RemovalRecord(TypedDict):
    step: str
    removed_rows: int
    percent_of_source: float


def find_column(columns: pd.Index, *candidates: str) -> str:
    """Resolve column names across Online Retail II workbook variants."""
    by_normalized = {"".join(c.lower().split()): c for c in columns}
    for candidate in candidates:
        found = by_normalized.get("".join(candidate.lower().split()))
        if found is not None:
            return found
    raise ValueError(f"Required column not found; expected one of {candidates}. Got {list(columns)}")


def load_workbook(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and combine all sheets, preserving the original source sheet."""
    workbook = pd.ExcelFile(path)
    frames = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        frame["SourceSheet"] = str(sheet)
        frames.append(frame)
    if not frames:
        raise ValueError("Workbook contains no worksheets")
    return pd.concat(frames, ignore_index=True), workbook.sheet_names


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input workbook not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw, sheet_names = load_workbook(INPUT_FILE)
    source_rows = len(raw)

    # Normalize the two common UCI header styles to a single internal schema.
    columns = {
        find_column(raw.columns, "Invoice", "InvoiceNo"): "InvoiceNo",
        find_column(raw.columns, "StockCode"): "StockCode",
        find_column(raw.columns, "Description"): "Description",
        find_column(raw.columns, "Quantity"): "Quantity",
        find_column(raw.columns, "InvoiceDate"): "InvoiceDate",
        find_column(raw.columns, "Price", "UnitPrice"): "UnitPrice",
        find_column(raw.columns, "Customer ID", "CustomerID"): "CustomerID",
        find_column(raw.columns, "Country"): "Country",
    }
    raw = raw.rename(columns=columns)
    keep = EXPECTED_COLUMNS + ["SourceSheet"]
    raw = raw[[c for c in keep if c in raw.columns]]

    # Normalize identifiers and parse fields before applying quality rules.
    raw["InvoiceNo"] = raw["InvoiceNo"].astype("string").str.strip()
    raw["StockCode"] = raw["StockCode"].astype("string").str.strip().str.upper()
    raw["Description"] = raw["Description"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip().str.upper()
    raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
    raw["UnitPrice"] = pd.to_numeric(raw["UnitPrice"], errors="coerce")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce", dayfirst=True)
    raw["CustomerID"] = pd.to_numeric(raw["CustomerID"], errors="coerce").astype("Int64")
    raw["Country"] = raw["Country"].astype("string").str.strip()

    missing_before = {
        col: {"count": int(raw[col].isna().sum()), "percent": round(float(raw[col].isna().mean() * 100), 4)}
        for col in raw.columns if col != "SourceSheet"
    }
    clean = raw.copy()
    removals: list[RemovalRecord] = []

    def remove(mask: pd.Series, name: str) -> None:
        nonlocal clean
        count = int(mask.fillna(False).sum())
        removals.append({"step": name, "removed_rows": count, "percent_of_source": round(count / source_rows * 100, 4)})
        clean = clean.loc[~mask.fillna(False)].copy()

    # Ordered, mutually exclusive steps make the removal report auditable.
    remove(clean["InvoiceNo"].str.upper().str.startswith("C", na=False), "cancelled_invoice")
    remove(clean["Quantity"].lt(0), "negative_quantity")
    remove(clean["UnitPrice"].le(0), "nonpositive_price")
    duplicate_mask = clean.duplicated(subset=EXPECTED_COLUMNS, keep="first")
    remove(duplicate_mask, "exact_duplicate")

    # Fill missing descriptions from the most frequent normalized description
    # for the same product code; rows with no resolvable product description go.
    description_by_code = (
        clean.dropna(subset=["Description"])
        .groupby("StockCode", dropna=True)["Description"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else pd.NA)
    )
    missing_description = clean["Description"].isna()
    clean.loc[missing_description, "Description"] = clean.loc[missing_description, "StockCode"].map(description_by_code)
    unresolved = clean["Description"].isna() | clean["StockCode"].isna()
    remove(unresolved, "unresolved_product_description_or_code")

    # Validate key fields and datetime. Customer ID is intentionally nullable.
    invalid_core = clean[["InvoiceNo", "Quantity", "UnitPrice", "InvoiceDate", "Country"]].isna().any(axis=1)
    remove(invalid_core, "missing_or_invalid_core_field")

    clean["Revenue"] = clean["Quantity"] * clean["UnitPrice"]
    clean["Date"] = clean["InvoiceDate"].dt.normalize()
    clean["Year"] = clean["InvoiceDate"].dt.year.astype("int16")
    clean["Month"] = clean["InvoiceDate"].dt.month.astype("int8")
    clean["Day"] = clean["InvoiceDate"].dt.day.astype("int8")
    clean["DayOfWeek"] = clean["InvoiceDate"].dt.dayofweek.astype("int8")
    clean["WeekOfYear"] = clean["InvoiceDate"].dt.isocalendar().week.astype("int8")

    # Demand modeling excludes non-product service/fee codes. Keep this as a
    # separate branch so the transaction-clean dataset remains useful for EDA.
    product_code = clean["StockCode"].str.contains(r"\d", regex=True, na=False)
    product_transactions = clean.loc[product_code].copy()
    excluded_service_count = int((~product_code).sum())

    daily = (
        product_transactions.groupby(["StockCode", "Date"], as_index=False)
        .agg(
            DailyDemand=("Quantity", "sum"),
            UnitPrice=("UnitPrice", "mean"),
            InvoiceCount=("InvoiceNo", "nunique"),
        )
    )
    if not daily.empty:
        # Make zero-sales days explicit for each product across the full data period.
        all_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D", name="Date")
        products = pd.Index(daily["StockCode"].drop_duplicates(), name="StockCode")
        complete_index = pd.MultiIndex.from_product([products, all_dates], names=["StockCode", "Date"])
        daily = daily.set_index(["StockCode", "Date"]).reindex(complete_index).reset_index()
        daily["DailyDemand"] = daily["DailyDemand"].fillna(0).astype("int64")
        daily["InvoiceCount"] = daily["InvoiceCount"].fillna(0).astype("int32")
        # Price is undefined on no-sale days; retain as missing to avoid inventing observations.
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = daily.sort_values(["StockCode", "Date"], kind="stable").reset_index(drop=True)
        daily["Year"] = daily["Date"].dt.year.astype("int16")
        daily["Month"] = daily["Date"].dt.month.astype("int8")
        daily["Day"] = daily["Date"].dt.day.astype("int8")
        daily["DayOfWeek"] = daily["Date"].dt.dayofweek.astype("int8")
        daily["WeekOfYear"] = daily["Date"].dt.isocalendar().week.astype("int8")

    clean.to_parquet(OUTPUT_DIR / "transactions_clean.parquet", index=False)
    daily.to_parquet(OUTPUT_DIR / "product_daily_demand.parquet", index=False)
    # CSV copies allow easy inspection without a parquet reader.
    clean.to_csv(OUTPUT_DIR / "transactions_clean.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUTPUT_DIR / "product_daily_demand.csv", index=False, encoding="utf-8-sig")

    removal_total = sum(row["removed_rows"] for row in removals)
    report = {
        "input_file": INPUT_FILE.name,
        "sheets": [str(s) for s in sheet_names],
        "source_rows": source_rows,
        "clean_transaction_rows": len(clean),
        "product_transaction_rows": len(product_transactions),
        "daily_demand_rows_including_zero_sales": len(daily),
        "service_or_fee_rows_excluded_from_demand_branch": excluded_service_count,
        "removal_steps": removals,
        "total_removed_by_quality_steps": removal_total,
        "missing_values_before_cleaning": missing_before,
        "missing_customer_id_after_cleaning": int(clean["CustomerID"].isna().sum()),
        "date_range": {
            "start": clean["Date"].min().strftime("%Y-%m-%d") if not clean.empty else None,
            "end": clean["Date"].max().strftime("%Y-%m-%d") if not clean.empty else None,
        },
        "notes": [
            "Customer ID is retained as nullable and is not imputed.",
            "Service and fee codes without digits are excluded only from demand aggregation.",
            "Zero-sales days are explicitly represented; UnitPrice remains null on those days.",
            "Holiday flag is not inferred because the project description does not specify a country calendar.",
        ],
    }
    (REPORT_DIR / "preprocessing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(removals).to_csv(REPORT_DIR / "preprocessing_removals.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({
        "source_rows": source_rows,
        "clean_transaction_rows": len(clean),
        "product_transaction_rows": len(product_transactions),
        "daily_demand_rows": len(daily),
        "removed_by_step": removals,
        "service_rows_excluded_from_demand": excluded_service_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
