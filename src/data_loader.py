"""Data loading and validation utilities."""
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "tenure_months", "monthly_charges", "total_charges", "support_tickets",
    "contract_type", "payment_method", "internet_service", "churn"
}


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df
