"""Train the PFMS spend-category classifier.

Reproducibly rebuilds the model from the labelled sample data. This replaces
the fragile pickle that was produced inside the research notebook (which was
tied to scikit-learn 0.24 and no longer loads on modern versions).

Usage:
    python train.py \
        --data "../export interview data.csv" \
        --labels "../label.csv" \
        --out model/pfms_pipeline.joblib

The saved artifact is a single sklearn Pipeline (TF-IDF -> LinearSVC) that
takes an iterable of RAW narration strings and returns the human-readable
spend category. Preprocessing is applied via pfms.preprocessing.clean_text.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from pfms.preprocessing import STOP_WORDS, clean_text

RANDOM_STATE = 3


def load_dataset(data_path: str, labels_path: str) -> pd.DataFrame:
    """Load narrations + numeric labels and map to readable category names."""
    data = pd.read_csv(data_path)
    label_map = pd.read_csv(labels_path, usecols=["indd", "label2"])
    id_to_name = label_map.set_index("indd")["label2"].to_dict()

    data = data.dropna(subset=["narrations", "label"]).copy()
    data["label_text"] = data["label"].map(id_to_name)
    data = data.dropna(subset=["label_text"])
    data["clean"] = data["narrations"].apply(clean_text)
    data = data[data["clean"].str.strip() != ""]
    return data


def build_pipeline(probability: bool) -> Pipeline:
    """TF-IDF + LinearSVC pipeline.

    We pass STOP_WORDS to TF-IDF as a second line of defence; the tokens are
    already removed by clean_text, but this keeps parity with the notebook.
    """
    tfidf = TfidfVectorizer(
        preprocessor=clean_text,
        encoding="latin-1",
        min_df=2,
        ngram_range=(1, 2),
        stop_words=STOP_WORDS,
        sublinear_tf=True,
    )
    svc = LinearSVC(random_state=RANDOM_STATE)
    if probability:
        # Wrap LinearSVC so we can expose confidence scores in the API.
        classifier = CalibratedClassifierCV(svc, cv=3)
    else:
        classifier = svc
    return Pipeline([("vectorizer", tfidf), ("classifier", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PFMS classifier")
    parser.add_argument("--data", default="../export interview data.csv")
    parser.add_argument("--labels", default="../label.csv")
    parser.add_argument("--out", default="model/pfms_pipeline.joblib")
    parser.add_argument(
        "--no-probability",
        action="store_true",
        help="Skip probability calibration (faster, no confidence scores)",
    )
    args = parser.parse_args()

    print(f"scikit-learn version: {sklearn.__version__}")
    df = load_dataset(args.data, args.labels)
    print(f"Loaded {len(df)} usable rows across {df['label_text'].nunique()} categories")

    X = df["narrations"]
    y = df["label_text"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=None
    )

    pipeline = build_pipeline(probability=not args.no_probability)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit on the full dataset for the deployed artifact.
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(pipeline, args.out)

    metadata = {
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_samples": int(len(df)),
        "categories": sorted(y.unique().tolist()),
        "test_accuracy": round(float(acc), 4),
        "test_weighted_f1": round(float(f1), 4),
        "has_probability": not args.no_probability,
    }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nSaved model -> {args.out}")
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
