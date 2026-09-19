import json
import sys

import pandas as pd

sys.path.append("src")
from data_ingestion import load_config


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["raw_path"])

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