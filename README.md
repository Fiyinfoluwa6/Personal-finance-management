# PFMS — Personal Finance Management Solution

PFMS is an intelligent solution that reads bank **transaction narrations** and
classifies them into **spend categories** (transfers, ATM withdrawals, bank
charges, salary, food, rent, etc.). It can categorise a single narration or an
entire uploaded bank statement, and it produces a per-category spend breakdown
that banks and individuals can use to understand where money goes.

The classifier was originally prototyped in a Jupyter notebook
(`Mono-personal-finance-management.ipynb`). This repo turns that prototype into
a **reproducible, deployable web application** with a JSON API.

---

## What's inside

```
web_app_mono/
├── app.py                     # Flask app: UI + JSON API
├── train.py                   # Reproducible model training script
├── requirements.txt           # Pinned runtime dependencies
├── Procfile                   # gunicorn entrypoint (for PaaS)
├── Dockerfile                 # Container build
├── model/
│   ├── pfms_pipeline.joblib   # Trained model artifact (produced by train.py)
│   └── pfms_pipeline.meta.json# Metrics + metadata
├── pfms/
│   ├── preprocessing.py       # Single source of truth for text cleaning
│   ├── predictor.py           # Loads the model, serves predictions
│   └── analytics.py           # CSV parsing + spend aggregation
└── templates/main.html        # Tabbed UI (single narration + CSV upload)
```

The model is a scikit-learn `Pipeline`: **TF-IDF (uni+bi-grams)** →
**LinearSVC** (probability-calibrated). Test accuracy on the sample data is
~86% (weighted F1 ~0.85 across 23 categories).

> **Note on the old artifacts:** the original `svc_model` pickle was tied to
> scikit-learn 0.24 and no longer loads on modern versions. `train.py`
> regenerates a clean, version-pinned artifact instead.

---

## Quick start (local)

```bash
cd web_app_mono
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) (Re)train the model — produces model/pfms_pipeline.joblib
python train.py

# 2) Run the app
python app.py          # dev server on http://localhost:5000
# or, production-style:
gunicorn app:app --bind 0.0.0.0:5000
```

---

## API

### `GET /health`
Liveness probe plus model metadata and the list of supported categories.

### `POST /api/predict`
Classify one or many narrations.

```bash
# single
curl -X POST localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"narration": "sms alert charges john doe"}'
# -> {"narration": "...", "category": "bank_charges", "confidence": 0.80}

# batch
curl -X POST localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"narrations": ["ATM Withdrawal Lagos", "stamp duty charge"]}'
```

### `POST /api/analyze`
Upload a bank-statement CSV and get a spend breakdown. The parser
auto-detects the narration column (`narration`, `description`, `details`,
`particulars`, …) and, if present, `amount` and `date` columns.

```bash
curl -X POST localhost:5000/api/analyze -F 'file=@statement.csv'
```

Response shape:

```json
{
  "summary": { "transaction_count": 5, "total_amount": 38302.5, "category_count": 5 },
  "by_category": [
    { "category": "transfer", "count": 1, "total_amount": 15000.0, "percentage": 39.16 }
  ],
  "transactions": [
    { "narration": "...", "category": "transfer", "amount": 15000.0, "confidence": 0.78, "date": "2024-01-02" }
  ]
}
```

---

## Deployment

The app is a standard WSGI (Flask) app served by gunicorn, so it runs anywhere
that runs Python or Docker.

### Option A — Docker (portable, recommended)

```bash
cd web_app_mono
docker build -t pfms .
docker run -p 8000:8000 pfms
```

### Option B — Render / Railway / Fly.io (managed PaaS)

These read the `Procfile` (`web: gunicorn app:app`) or the `Dockerfile`
directly. Recommended settings:

- **Build:** `pip install -r requirements.txt && python train.py`
  (or commit `model/pfms_pipeline.joblib` and skip the train step).
- **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT`
- **Health check path:** `/health`

**Render** and **Railway** are the lowest-friction choices for this app: connect
the GitHub repo, point the service at `web_app_mono/`, and they build and host it
automatically. For heavier/bank-internal use, the Docker image can go to any
container platform (AWS ECS/Fargate, Google Cloud Run, Azure Container Apps).

### Environment variables

| Variable            | Default                        | Purpose                          |
|---------------------|--------------------------------|----------------------------------|
| `PORT`              | `5000` (dev) / `8000` (Docker) | Port to bind                     |
| `PFMS_MODEL_PATH`   | `model/pfms_pipeline.joblib`   | Override model artifact location |
| `PFMS_MAX_UPLOAD_MB`| `10`                           | Max CSV upload size (MB)         |

---

## Retraining on your own data

`train.py` expects a narrations CSV and a label-mapping CSV (see
`export interview data.csv` and `label.csv`). Point it at new files and it will
retrain, evaluate, and write a fresh artifact + metadata:

```bash
python train.py --data path/to/narrations.csv --labels path/to/labels.csv \
  --out model/pfms_pipeline.joblib
```
