"""Compare reference and incoming feature distributions for data drift."""
import sys

import numpy as np
import pandas as pd
import argparse

from data_ingestion import load_config, validate_schema
from preprocessing import transform, ORDINAL_MAPPINGS


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Measure how much a current numeric distribution differs from a reference."""
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = reference.quantile(quantiles).values
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / len(current))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    """Compare configured feature distributions and signal whether retraining is needed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-data", default=None, help="Path to the new/incoming data CSV")
    args = parser.parse_args()

    cfg = load_config()
    drift_cfg = cfg["drift"]

    reference_df = pd.read_parquet(drift_cfg["reference_path"])
    new_data_path = args.new_data or drift_cfg["new_data_path"]

    try:
        new_raw = pd.read_csv(new_data_path)
    except FileNotFoundError:
        print(f"ERROR: drift comparison data not found at {drift_cfg['new_data_path']}")
        sys.exit(2) 

    validate_schema(new_raw, cfg["data"]["target_column"])
    new_processed = transform(
        new_raw,
        drop_columns=cfg["data"]["drop_columns"],
        ordinal_columns=cfg["data"]["ordinal_columns"],
        ordinal_mappings=ORDINAL_MAPPINGS,
        nominal_columns=cfg["data"]["nominal_columns"],
    )

    results = {}
    for col in drift_cfg["check_columns"]:
        if col in reference_df.columns and col in new_processed.columns:
            results[col] = population_stability_index(reference_df[col], new_processed[col])

    print("PSI per column:", results)

    triggered = any(v > drift_cfg["threshold"] for v in results.values())
    if triggered:
        print("Drift threshold exceeded — retraining should run.")
        sys.exit(1)
    else:
        print("No significant drift detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()