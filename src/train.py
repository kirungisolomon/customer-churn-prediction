"""Train and persist the churn classification pipeline."""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_data
from feature_engineering import add_features
from preprocessing import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_data.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"


def train() -> None:
    df = add_features(load_data(DATA_PATH))
    X = df.drop(columns="churn")
    y = df["churn"].astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
