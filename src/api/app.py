import json
from data_ingestion import load_config
from preprocessing import transform, ORDINAL_MAPPINGS

with open("feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

cfg = load_config()

df = transform(
    df,
    drop_columns=cfg["data"]["drop_columns"],
    ordinal_columns=cfg["data"]["ordinal_columns"],
    ordinal_mappings=ORDINAL_MAPPINGS,
    nominal_columns=cfg["data"]["nominal_columns"],
)
df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)