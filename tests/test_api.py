import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import load_config
from src.api.server import app as flask_app

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_returns_probability(client):
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["raw_path"])
    row = df.drop(columns=[cfg["data"]["target_column"]]).iloc[0].to_dict()

    response = client.post(
        "/predict",
        data=json.dumps(row),
        content_type="application/json",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0