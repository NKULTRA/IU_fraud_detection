import json

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from data_ingestion import load_config
from preprocessing import load_processed

cfg = load_config()
X, y = load_processed(cfg["data"]["processed_path"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("fraud-detection")

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    mlflow.log_metrics(metrics)
    print("Metrics:", metrics)

    feature_columns = X.columns.tolist()
    with open("feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    mlflow.log_artifact("feature_columns.json")

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="fraud-detection-model",
    )