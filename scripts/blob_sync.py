"""
Utility for persisting the MLflow tracking store (mlflow.db) to Blob
Storage across ephemeral pipeline runs. Hosted DevOps agents are
throwaway VMs, so anything not explicitly synced back to Blob is lost
when the job finishes.
"""
import os

from anyio import Path
from azure.storage.blob import BlobServiceClient


def _client():
    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    return BlobServiceClient.from_connection_string(connection_string)


def download_if_exists(container: str, blob_name: str, local_path: str) -> None:
    blob = _client().get_blob_client(container=container, blob=blob_name)
    if blob.exists():
        with open(local_path, "wb") as f:
            f.write(blob.download_blob().readall())
        print(f"Restored {blob_name} from Blob -> {local_path}")
    else:
        print(f"No existing {blob_name} in Blob — starting fresh.")


def upload(container: str, blob_name: str, local_path: str) -> None:
    blob = _client().get_blob_client(container=container, blob=blob_name)
    with open(local_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)
    print(f"Persisted {local_path} -> Blob ({blob_name})")


if __name__ == "__main__":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data_ingestion import load_config

    cfg = load_config()
    action = sys.argv[1] 
    container = cfg["data"]["azure"]["container"]
    blob_name = "mlflow.db"
    local_path = "mlflow.db"

    if action == "download":
        download_if_exists(container, blob_name, local_path)
    elif action == "upload":
        upload(container, blob_name, local_path)
    else:
        raise ValueError("Usage: python scripts/blob_sync.py [download|upload]")