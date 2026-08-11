"""Generate a deterministic synthetic dataset for local development."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = rng.normal(75, 22, n).clip(20, 160)
    tickets = rng.poisson(2, n)
    contracts = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[.55, .25, .20])
    payments = rng.choice(["electronic", "bank_transfer", "card"], n)
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[.48, .35, .17])
    total = monthly * tenure * rng.normal(1, .08, n)
    risk = -1.0 + .035 * tickets + .012 * monthly - .018 * tenure
    risk += (contracts == "month-to-month") * .9
    risk += (internet == "fiber") * .15
    probability = 1 / (1 + np.exp(-risk))
    churn = rng.binomial(1, probability)

    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly.round(2),
        "total_charges": total.round(2),
        "support_tickets": tickets,
        "contract_type": contracts,
        "payment_method": payments,
        "internet_service": internet,
        "churn": churn,
    })


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "customer_data.csv"
    output.parent.mkdir(exist_ok=True)
    generate().to_csv(output, index=False)
    print(f"Generated {len(generate())} records at {output}")
