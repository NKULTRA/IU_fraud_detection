import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import pandas as pd

from src.data_ingestion import load_config
from src.preprocessing import transform, ORDINAL_MAPPINGS

tracking_dir = Path(__file__).resolve().parents[2] / "mlruns"
mlflow.set_tracking_uri(tracking_dir.as_uri())

app = Flask(__name__)

cfg = load_config()

with open("feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

MODEL_NAME = "fraud-detection-model"
_model = None


def get_model():
    """Lazily load whichever version currently has the 'production' alias."""
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    df = pd.DataFrame([payload])

    df = transform(
        df,
        drop_columns=cfg["data"]["drop_columns"],
        ordinal_columns=cfg["data"]["ordinal_columns"],
        ordinal_mappings=ORDINAL_MAPPINGS,
        nominal_columns=cfg["data"]["nominal_columns"],
    )
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    model = get_model()
    proba = model.predict_proba(df)[:, 1][0]

    return jsonify({
        "fraud_probability": float(proba),
        "model_name": MODEL_NAME,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)