"""Load configuration and application data from local or Azure storage."""

from pathlib import Path

import pandas as pd
import yaml
from azure.storage.blob import BlobServiceClient
import io
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_data_from_blob(container: str, blob_name: str) -> pd.DataFrame:
    """Download a CSV blob from Azure Storage and return it as a dataframe."""
    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    client = BlobServiceClient.from_connection_string(connection_string)
    blob = client.get_blob_client(container=container, blob=blob_name)
    data = blob.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_config(path: str = "config/config.yaml") -> dict:
    """Load the YAML configuration from a project-relative or absolute path."""
    resolved_path = _resolve_project_path(path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load the raw applications data from disk."""
    resolved_path = _resolve_project_path(raw_path)
    return pd.read_csv(resolved_path)


def validate_schema(df: pd.DataFrame, target_column: str) -> None:
    """Basic sanity checks before handing off to preprocessing."""
    if target_column not in df.columns:
        raise ValueError(f"Expected target column '{target_column}' not found.")
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")


if __name__ == "__main__":
    cfg = load_config()
    df = load_raw_data_from_blob(cfg["azure"]["container"], cfg["azure"]["blob_name"])
    validate_schema(df, cfg["data"]["target_column"])

    Path(cfg["data"]["raw_path"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["data"]["raw_path"], index=False)
    print(f"Loaded {len(df)} rows from Blob, saved to {cfg['data']['raw_path']}")