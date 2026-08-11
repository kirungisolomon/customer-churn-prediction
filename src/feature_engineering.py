"""Feature engineering for customer behavior signals."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["avg_monthly_value"] = result["total_charges"] / result["tenure_months"].clip(lower=1)
    result["tickets_per_month"] = result["support_tickets"] / result["tenure_months"].clip(lower=1)
    return result
