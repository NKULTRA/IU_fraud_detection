"""Train, evaluate, and register the fraud detection model with MLflow."""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from data_ingestion import load_config
from preprocessing import load_processed

cfg = load_config()
# Load the already-prepared feature matrix and split it for evaluation.
X, y = load_processed(cfg["data"]["processed_path"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tracking_dir = Path(__file__).resolve().parents[1] / "mlruns"
artifact_dir = Path(__file__).resolve().parents[1] / "mlartifacts"
mlflow.set_tracking_uri(tracking_dir.as_uri())

experiment_name = "fraud-detection"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name, artifact_location=artifact_dir.as_uri())

mlflow.set_experiment(experiment_name)

# Train, evaluate, and persist the model and its feature ordering in one run.
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