import pandas as pd


ORDINAL_MAPPINGS = {
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfPolicyHolder": ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
                          "36 to 40", "41 to 50", "51 to 65", "over 65"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"],
    "Year": [1994, 1995, 1996],
}


def build_features(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str],
    ordinal_columns: list[str],
    ordinal_mappings: dict[str, list],
    nominal_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocessing pipeline for the fraud detection dataset."""
    df = df.drop(columns=drop_columns)

    for col in ordinal_columns:
        df[col] = pd.Categorical(
            df[col], categories=ordinal_mappings[col], ordered=True
        ).codes

    df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    y = df.pop(target_column)
    X = df
    return X, y


def save_processed(X: pd.DataFrame, y: pd.Series, out_path: str) -> None:
    out = X.copy()
    out["target"] = y.values
    out.to_parquet(out_path, index=False)


if __name__ == "__main__":
    from data_ingestion import load_config, load_raw_data, validate_schema

    cfg = load_config()
    df = load_raw_data(cfg["data"]["raw_path"])
    validate_schema(df, cfg["data"]["target_column"])

    X, y = build_features(
        df,
        target_column=cfg["data"]["target_column"],
        drop_columns=cfg["data"]["drop_columns"],
        ordinal_columns=cfg["data"]["ordinal_columns"],
        ordinal_mappings=ORDINAL_MAPPINGS, 
        nominal_columns=cfg["data"]["nominal_columns"],
    )
    save_processed(X, y, cfg["data"]["processed_path"])
    print(f"Saved {len(X)} rows, {X.shape[1]} features -> {cfg['data']['processed_path']}")