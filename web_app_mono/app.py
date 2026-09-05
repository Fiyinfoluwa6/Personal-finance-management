"""PFMS Flask application.

Endpoints
---------
GET  /                 -> HTML UI (single narration + CSV upload)
POST /                 -> HTML UI form submission (single narration)
GET  /health           -> liveness/readiness probe + model metadata
POST /api/predict      -> JSON: {"narration": "..."} or {"narrations": [...]}
POST /api/analyze      -> multipart CSV upload -> spend analytics JSON
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from pfms.analytics import analyse, parse_csv
from pfms.predictor import ModelNotFoundError, get_predictor

MAX_CONTENT_LENGTH = int(os.environ.get("PFMS_MAX_UPLOAD_MB", "10")) * 1024 * 1024

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _predictor_or_error():
    """Return (predictor, None) or (None, json_error_response)."""
    try:
        return get_predictor(), None
    except ModelNotFoundError as exc:
        return None, (
            jsonify({"error": "model_unavailable", "detail": str(exc)}),
            503,
        )


@app.route("/health", methods=["GET"])
def health():
    try:
        predictor = get_predictor()
        return jsonify(
            {
                "status": "ok",
                "model": predictor.metadata or {"note": "no metadata file"},
                "categories": predictor.categories,
            }
        )
    except ModelNotFoundError as exc:
        return jsonify({"status": "degraded", "detail": str(exc)}), 503


@app.route("/api/predict", methods=["POST"])
def api_predict():
    predictor, err = _predictor_or_error()
    if err:
        return err

    payload = request.get_json(silent=True) or {}
    if "narrations" in payload:
        narrations = payload["narrations"]
        if not isinstance(narrations, list):
            return jsonify({"error": "narrations must be a list"}), 400
        return jsonify({"results": predictor.predict_many(narrations)})

    narration = payload.get("narration")
    if not narration:
        return jsonify(
            {"error": "Provide 'narration' (string) or 'narrations' (list)."}
        ), 400
    return jsonify(predictor.predict_one(narration))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    predictor, err = _predictor_or_error()
    if err:
        return err

    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file under form field 'file'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    try:
        df = parse_csv(file.read())
        result = analyse(df, predictor)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": "processing_failed", "detail": str(exc)}), 500


@app.route("/", methods=["GET", "POST"])
def main():
    predictor, _ = _predictor_or_error()
    context = {"categories": predictor.categories if predictor else []}

    if request.method == "POST":
        narration = request.form.get("narration", "").strip()
        if narration and predictor:
            prediction = predictor.predict_one(narration)
            context.update(
                original_input={"narration": narration},
                result=prediction["category"],
                confidence=prediction.get("confidence"),
            )
    return render_template("main.html", **context)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
