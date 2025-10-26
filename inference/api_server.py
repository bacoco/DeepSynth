"""Flask API server exposing the summarisation model."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from .infer import DeepSynthSummarizer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL_PATH = os.getenv("MODEL_PATH", "./deepsynth-summarizer")
summarizer: DeepSynthSummarizer | None = None


def init_model() -> None:
    global summarizer
    if summarizer is None:
        summarizer = DeepSynthSummarizer(MODEL_PATH)
        LOGGER.info("Model initialised from %s", MODEL_PATH)


@app.before_first_request
def setup() -> None:  # pragma: no cover
    init_model()


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": summarizer is not None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.route("/summarize/text", methods=["POST"])
def summarize_text() -> Any:
    init_model()
    payload = request.get_json() or {}
    text = payload.get("text", "")
    if not text.strip():
        return jsonify({"error": "Missing text"}), 400

    max_length = int(payload.get("max_length", 128))
    temperature = float(payload.get("temperature", 0.7))
    summary = summarizer.summarize_text(text, max_length=max_length, temperature=temperature)  # type: ignore[arg-type]
    response = {
        "summary": summary,
        "original_length": len(text),
        "summary_length": len(summary),
        "compression_ratio": round(len(text) / max(len(summary), 1), 2),
        "parameters": {
            "max_length": max_length,
            "temperature": temperature,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    return jsonify(response)


@app.route("/summarize/file", methods=["POST"])
def summarize_file() -> Any:
    init_model()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(uploaded.filename)
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"{uuid.uuid4()}_{filename}"
    uploaded.save(temp_path)

    try:
        with open(temp_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        max_length = int(request.form.get("max_length", 128))
        temperature = float(request.form.get("temperature", 0.7))
        summary = summarizer.summarize_text(text, max_length=max_length, temperature=temperature)  # type: ignore[arg-type]
    finally:
        temp_path.unlink(missing_ok=True)

    return jsonify({"summary": summary})


@app.route("/summarize/image", methods=["POST"])
def summarize_image() -> Any:
    init_model()
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    uploaded = request.files["file"]
    filename = secure_filename(uploaded.filename or "image.png")
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f"{uuid.uuid4()}_{filename}"
    uploaded.save(temp_path)

    try:
        summary = summarizer.summarize_image(str(temp_path))  # type: ignore[arg-type]
    except Exception as exc:
        LOGGER.error("Image summarisation failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    finally:
        temp_path.unlink(missing_ok=True)

    return jsonify({"summary": summary})


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5000)
