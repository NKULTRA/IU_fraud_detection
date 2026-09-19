from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

tracking_dir = Path(__file__).resolve().parents[1] / "mlruns"
mlflow.set_tracking_uri(tracking_dir.as_uri())

MODEL_NAME = "fraud-detection-model"

client = MlflowClient()

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version))

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=latest_version.version,
)

print(f"Alias 'production' now points to {MODEL_NAME} v{latest_version.version}")