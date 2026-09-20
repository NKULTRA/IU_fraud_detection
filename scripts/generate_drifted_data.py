"""Generate a synthetic later-month dataset with controlled feature drift."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import load_config

SHIFT_FRACTION = 0.5  # fraction of rows to perturb
RNG_SEED = 42


def generate_drifted_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create a reproducible synthetic batch with shifted applicant features."""
    rng = np.random.default_rng(RNG_SEED)
    drifted = df.copy()

    n = len(drifted)
    shift_mask = rng.random(n) < SHIFT_FRACTION

    # Applicant pool skews older
    drifted.loc[shift_mask, "Age"] = drifted.loc[shift_mask, "Age"] + rng.integers(10, 20, shift_mask.sum())
    drifted["Age"] = drifted["Age"].clip(upper=80)

    # Deductible amounts trend higher
    drifted.loc[shift_mask, "Deductible"] = drifted.loc[shift_mask, "Deductible"] * 1.4

    return drifted


if __name__ == "__main__":
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["raw_path"])
    drifted = generate_drifted_data(df)

    out_path = "data/fraud_oracle_drifted.csv"
    drifted.to_csv(out_path, index=False)
    print(f"Saved drifted dataset ({len(drifted)} rows) -> {out_path}")