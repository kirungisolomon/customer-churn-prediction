"""Reusable prediction interface for the churn model."""

from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_customer(customer: dict) -> dict:
    """Predict churn for one customer record."""
    model = load_model()
    frame = pd.DataFrame([customer])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "label": "Likely to churn" if prediction else "Likely to stay",
    }
