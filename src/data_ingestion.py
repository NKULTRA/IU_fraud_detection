from pathlib import Path

import pandas as pd
import yaml
from azure.storage.blob import BlobServiceClient
import io

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_raw_data_from_blob(connection_string: str, container: str, blob_name: str) -> pd.DataFrame:
    client = BlobServiceClient.from_connection_string(connection_string)
    blob = client.get_blob_client(container=container, blob=blob_name)
    data = blob.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_config(path: str = "config/config.yaml") -> dict:
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
    df = load_raw_data(cfg["data"]["raw_path"])
    validate_schema(df, cfg["data"]["target_column"])
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")