from mlflow.tracking import MlflowClient

MODEL_NAME = "fraud-detection-model"

client = MlflowClient()

# Get the most recently registered version (highest version number)
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version))

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=latest_version.version,
)

print(f"Alias 'production' now points to {MODEL_NAME} v{latest_version.version}")