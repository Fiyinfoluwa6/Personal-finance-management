"""Bank-statement analytics.

Takes a parsed statement (list of rows with a narration and optional amount)
and produces per-category spend aggregates that a bank or an individual can use
to understand where money goes.
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd

from pfms.predictor import Predictor

# Column names we will try to auto-detect in an uploaded CSV, in priority order.
NARRATION_ALIASES = [
    "narration", "narrations", "description", "details", "particulars",
    "remarks", "transaction details", "transaction_details", "memo",
]
AMOUNT_ALIASES = [
    "amount", "value", "debit", "credit", "amount (ngn)", "amount_ngn",
    "transaction amount", "transaction_amount",
]
DATE_ALIASES = ["date", "transaction date", "transaction_date", "value date", "posted"]


def _find_column(columns: list[str], aliases: list[str]) -> Optional[str]:
    lowered = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    # Fallback: substring match.
    for alias in aliases:
        for low, original in lowered.items():
            if alias in low:
                return original
    return None


def _coerce_amount(series: pd.Series) -> pd.Series:
    """Best-effort conversion of a currency column to float."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", "0")
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def parse_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV into a normalised DataFrame.

    Returns a DataFrame with columns: narration, amount (may be 0), date (opt).
    Raises ValueError if no narration column can be identified.
    """
    df = pd.read_csv(io.BytesIO(raw_bytes))
    if df.empty:
        raise ValueError("The uploaded file contains no rows.")

    narration_col = _find_column(list(df.columns), NARRATION_ALIASES)
    if narration_col is None:
        raise ValueError(
            "Could not find a narration/description column. Expected one of: "
            + ", ".join(NARRATION_ALIASES)
        )

    amount_col = _find_column(list(df.columns), AMOUNT_ALIASES)
    date_col = _find_column(list(df.columns), DATE_ALIASES)

    out = pd.DataFrame()
    out["narration"] = df[narration_col].astype(str)
    out["amount"] = _coerce_amount(df[amount_col]) if amount_col else 0.0
    if date_col:
        out["date"] = df[date_col].astype(str)
    return out


def analyse(df: pd.DataFrame, predictor: Predictor) -> dict:
    """Categorise every row and build spend analytics.

    Returns a dict with:
      - transactions: per-row narration/category/amount
      - summary: totals
      - by_category: aggregated counts and amounts per category, sorted by spend
    """
    predictions = predictor.predict_many(df["narration"].tolist())
    df = df.copy()
    df["category"] = [p["category"] for p in predictions]
    if predictions and "confidence" in predictions[0]:
        df["confidence"] = [p.get("confidence") for p in predictions]

    df["abs_amount"] = df["amount"].abs()

    grouped = (
        df.groupby("category")
        .agg(count=("narration", "size"), total_amount=("abs_amount", "sum"))
        .reset_index()
        .sort_values(["total_amount", "count"], ascending=False)
    )

    total_amount = float(df["abs_amount"].sum())
    by_category = []
    for _, row in grouped.iterrows():
        amt = float(row["total_amount"])
        by_category.append(
            {
                "category": row["category"],
                "count": int(row["count"]),
                "total_amount": round(amt, 2),
                "percentage": round((amt / total_amount * 100) if total_amount else 0.0, 2),
            }
        )

    transactions = []
    for _, row in df.iterrows():
        record = {
            "narration": row["narration"],
            "category": row["category"],
            "amount": round(float(row["amount"]), 2),
        }
        if "confidence" in df.columns:
            record["confidence"] = row["confidence"]
        if "date" in df.columns:
            record["date"] = row["date"]
        transactions.append(record)

    return {
        "summary": {
            "transaction_count": int(len(df)),
            "total_amount": round(total_amount, 2),
            "category_count": int(grouped.shape[0]),
        },
        "by_category": by_category,
        "transactions": transactions,
    }
