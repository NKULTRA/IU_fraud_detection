import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import load_config


def main():
    cfg = load_config()
    raw_path = PROJECT_ROOT / cfg["data"]["raw_path"]
    df = pd.read_csv(raw_path)

    # Take one real row, drop the target column, keep everything else exactly as-is
    row = df.drop(columns=[cfg["data"]["target_column"]]).iloc[0]
    payload = row.to_dict()

    print(json.dumps(payload, indent=2))

    # Also print a ready-to-run PowerShell command
    compact = json.dumps(payload).replace('"', '\\"')
    print("\n--- PowerShell ---")
    print(f'Invoke-RestMethod -Uri http://localhost:5001/predict -Method Post '
          f'-ContentType "application/json" -Body "{compact}"')


if __name__ == "__main__":
    main()