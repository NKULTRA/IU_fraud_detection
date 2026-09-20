"""Persist the local MLflow file store in Azure Blob Storage."""

import os
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

from azure.storage.blob import BlobServiceClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import load_config


def _client():
    """Create an Azure Blob client from the pipeline connection string."""
    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    return BlobServiceClient.from_connection_string(connection_string)


def download_if_exists(container: str, blob_name: str, local_path: Path) -> None:
    """Restore a zipped MLflow directory from Blob Storage if available."""
    blob = _client().get_blob_client(container=container, blob=blob_name)
    if blob.exists():
        archive_path = PROJECT_ROOT / ".mlruns-download.zip"
        archive_path.write_bytes(blob.download_blob().readall())
        shutil.rmtree(local_path, ignore_errors=True)
        with ZipFile(archive_path) as archive:
            archive.extractall(PROJECT_ROOT)
        archive_path.unlink()
        print(f"Restored {blob_name} from Blob -> {local_path}")
    else:
        print(f"No existing {blob_name} in Blob — starting fresh.")


def upload(container: str, blob_name: str, local_path: Path) -> None:
    """Upload the local MLflow directory as a zip archive to Blob Storage."""
    blob = _client().get_blob_client(container=container, blob=blob_name)
    archive_base = PROJECT_ROOT / ".mlruns-upload"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", PROJECT_ROOT, local_path.name))
    with archive_path.open("rb") as archive:
        blob.upload_blob(archive, overwrite=True)
    archive_path.unlink()
    print(f"Persisted {local_path} -> Blob ({blob_name})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python scripts/blob_sync.py [download|upload]")

    action = sys.argv[1]

    cfg = load_config()
    container = cfg["azure"]["container"]
    blob_name = "mlruns.zip"
    local_path = PROJECT_ROOT / "mlruns"

    if action == "download":
        download_if_exists(container, blob_name, local_path)
    elif action == "upload":
        upload(container, blob_name, local_path)
    else:
        raise ValueError("Usage: python scripts/blob_sync.py [download|upload]")