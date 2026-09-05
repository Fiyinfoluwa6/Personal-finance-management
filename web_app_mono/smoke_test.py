"""Minimal smoke test exercising the live app in-process.

Run after `python train.py`. Exits non-zero on any failure so CI can gate on it.
"""

from __future__ import annotations

import io
import sys
import warnings

warnings.filterwarnings("ignore")

from app import app  # noqa: E402


def _check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        sys.exit(1)


def main() -> None:
    client = app.test_client()

    # Health
    resp = client.get("/health")
    _check(resp.status_code == 200, "GET /health returns 200")
    _check(resp.get_json().get("status") == "ok", "health reports ok")

    # Single prediction
    resp = client.post("/api/predict", json={"narration": "sms alert charges john doe"})
    _check(resp.status_code == 200, "POST /api/predict (single) returns 200")
    _check("category" in resp.get_json(), "single prediction has a category")

    # Batch prediction
    resp = client.post(
        "/api/predict",
        json={"narrations": ["ATM Withdrawal Lagos", "stamp duty charge"]},
    )
    _check(resp.status_code == 200, "POST /api/predict (batch) returns 200")
    _check(len(resp.get_json()["results"]) == 2, "batch returns 2 results")

    # Bad request
    resp = client.post("/api/predict", json={})
    _check(resp.status_code == 400, "empty predict body returns 400")

    # CSV analysis
    csv = (
        b"date,description,amount\n"
        b"2024-01-01,sms alert charges john doe,52.5\n"
        b"2024-01-02,ATM Withdrawal Lagos,20000\n"
        b"2024-01-03,stamp duty charge,50\n"
    )
    resp = client.post(
        "/api/analyze",
        data={"file": (io.BytesIO(csv), "stmt.csv")},
        content_type="multipart/form-data",
    )
    _check(resp.status_code == 200, "POST /api/analyze returns 200")
    body = resp.get_json()
    _check(body["summary"]["transaction_count"] == 3, "analyze counts 3 transactions")
    _check(len(body["by_category"]) >= 1, "analyze returns category breakdown")

    # Missing narration column should be a clean 400
    bad_csv = b"foo,bar\n1,2\n"
    resp = client.post(
        "/api/analyze",
        data={"file": (io.BytesIO(bad_csv), "bad.csv")},
        content_type="multipart/form-data",
    )
    _check(resp.status_code == 400, "CSV without narration column returns 400")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
