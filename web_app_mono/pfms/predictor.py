"""Model loading and prediction utilities.

Wraps the trained sklearn pipeline behind a small, stable interface so the web
layer never touches sklearn directly.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Iterable

import joblib
import numpy as np

_DEFAULT_MODEL_PATH = os.environ.get(
    "PFMS_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "model", "pfms_pipeline.joblib"),
)


class ModelNotFoundError(RuntimeError):
    """Raised when the model artifact is missing."""


class Predictor:
    """Loads the pipeline once and serves predictions."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise ModelNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Run `python train.py` to build it."
            )
        self.pipeline = joblib.load(self.model_path)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        meta_path = os.path.splitext(self.model_path)[0] + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                return json.load(fh)
        return {}

    @property
    def categories(self) -> list[str]:
        return list(getattr(self.pipeline, "classes_", []))

    def predict_one(self, narration: str) -> dict:
        """Predict the spend category for a single narration."""
        results = self.predict_many([narration])
        return results[0]

    def predict_many(self, narrations: Iterable[str]) -> list[dict]:
        """Predict spend categories for many narrations.

        Returns a list of {"narration", "category", "confidence"} dicts.
        Confidence is included only when the model supports probabilities.
        """
        items = [("" if n is None else str(n)) for n in narrations]
        if not items:
            return []

        preds = self.pipeline.predict(items)

        confidences: list[float | None]
        if hasattr(self.pipeline, "predict_proba"):
            try:
                probs = self.pipeline.predict_proba(items)
                confidences = [float(np.max(row)) for row in probs]
            except Exception:
                confidences = [None] * len(items)
        else:
            confidences = [None] * len(items)

        out = []
        for narration, category, conf in zip(items, preds, confidences):
            record = {"narration": narration, "category": str(category)}
            if conf is not None:
                record["confidence"] = round(conf, 4)
            out.append(record)
        return out


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Return a process-wide singleton predictor."""
    return Predictor()
