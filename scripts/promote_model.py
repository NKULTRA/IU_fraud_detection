"""Promote the newest registered fraud model to the production alias."""

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

tracking_dir = Path(__file__).resolve().parents[1] / "mlruns"
mlflow.set_tracking_uri(tracking_dir.as_uri())

MODEL_NAME = "fraud-detection-model"
METRIC_TO_COMPARE = "recall" 

# Select the newest registered version and expose it through the production alias.
client = MlflowClient()

# Get the most recently registered version (the one just trained)
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version))

new_run = client.get_run(latest_version.run_id)
new_metric = new_run.data.metrics.get(METRIC_TO_COMPARE)

# Check whether a production version already exists
try:
    current_prod = client.get_model_version_by_alias(MODEL_NAME, "production")
    current_run = client.get_run(current_prod.run_id)
    current_metric = current_run.data.metrics.get(METRIC_TO_COMPARE)
except mlflow.exceptions.MlflowException:
    current_prod = None
    current_metric = None

if current_prod is None:
    print(f"No existing production model — promoting v{latest_version.version} as the first production version.")
    should_promote = True
elif new_metric is None:
    print("New model has no recorded metric — skipping promotion.")
    should_promote = False
elif new_metric >= current_metric:
    print(f"New {METRIC_TO_COMPARE}={new_metric:.3f} >= current production {METRIC_TO_COMPARE}={current_metric:.3f} — promoting.")
    should_promote = True
else:
    print(f"New {METRIC_TO_COMPARE}={new_metric:.3f} < current production {METRIC_TO_COMPARE}={current_metric:.3f} — keeping current model.")
    should_promote = False

if should_promote:
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=latest_version.version,
    )
    print(f"Alias 'production' now points to {MODEL_NAME} v{latest_version.version}")
else:
    print(f"Production alias unchanged — still v{current_prod.version}")