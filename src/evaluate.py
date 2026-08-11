"""Evaluate the trained churn model."""
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from data_loader import load_data
from feature_engineering import add_features

ROOT = Path(__file__).resolve().parents[1]


def evaluate() -> dict:
    df = add_features(load_data(ROOT / "data" / "customer_data.csv"))
    X = df.drop(columns="churn")
    y = df["churn"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = joblib.load(ROOT / "models" / "churn_pipeline.joblib")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": classification_report(y_test, predictions),
    }
    return metrics


if __name__ == "__main__":
    for key, value in evaluate().items():
        print(f"{key}:\n{value}")
