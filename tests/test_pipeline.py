import pandas as pd
from src.feature_engineering import add_features
from src.preprocessing import build_preprocessor


def test_feature_engineering_adds_behavior_features():
    df = pd.DataFrame({
        "tenure_months": [10], "total_charges": [500.0], "support_tickets": [5]
    })
    result = add_features(df)
    assert result.loc[0, "avg_monthly_value"] == 50
    assert result.loc[0, "tickets_per_month"] == 0.5


def test_preprocessor_can_fit_basic_data():
    df = pd.DataFrame({
        "tenure_months": [10, 20], "monthly_charges": [50, 80],
        "total_charges": [500, 1600], "support_tickets": [1, 3],
        "contract_type": ["one-year", "month-to-month"],
        "payment_method": ["card", "electronic"],
        "internet_service": ["dsl", "fiber"],
    })
    transformed = build_preprocessor().fit_transform(df)
    assert transformed.shape[0] == 2
