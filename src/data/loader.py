"""
src/data/loader.py
------------------
Generates a realistic synthetic telecom churn dataset and saves it to data/raw/.
Run directly: python src/data/loader.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_SAMPLES = 5000


def generate_churn_data(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate synthetic customer churn dataset with realistic feature correlations."""
    rng = np.random.default_rng(seed)

    tenure          = rng.integers(1, 73, n)                    # months 1–72
    monthly_charges = rng.uniform(20, 120, n).round(2)
    total_charges   = (tenure * monthly_charges * rng.uniform(0.9, 1.1, n)).round(2)
    num_products    = rng.integers(1, 6, n)
    support_calls   = rng.integers(0, 10, n)
    contract        = rng.choice(["Month-to-month", "One year", "Two year"], n,
                                  p=[0.55, 0.25, 0.20])
    payment_method  = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    gender          = rng.choice(["Male", "Female"], n)
    senior_citizen  = rng.integers(0, 2, n)
    partner         = rng.choice(["Yes", "No"], n)
    dependents      = rng.choice(["Yes", "No"], n)
    paperless       = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    tech_support    = rng.choice(["Yes", "No", "No internet service"], n)
    online_security = rng.choice(["Yes", "No", "No internet service"], n)
    streaming_tv    = rng.choice(["Yes", "No", "No internet service"], n)
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], n)

    # Churn probability model (realistic correlations)
    churn_score = (
        -2.5
        + 0.03  * support_calls
        + 0.015 * monthly_charges
        - 0.04  * tenure
        - 0.5   * (contract == "One year").astype(float)
        - 1.2   * (contract == "Two year").astype(float)
        + 0.4   * (payment_method == "Electronic check").astype(float)
        + 0.3   * (internet_service == "Fiber optic").astype(float)
        + 0.2   * senior_citizen
        - 0.2   * (online_security == "Yes").astype(float)
        - 0.15  * (tech_support == "Yes").astype(float)
        + rng.normal(0, 0.3, n)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id":       [f"CUST{i:05d}" for i in range(n)],
        "gender":            gender,
        "senior_citizen":    senior_citizen,
        "partner":           partner,
        "dependents":        dependents,
        "tenure":            tenure,
        "contract":          contract,
        "paperless_billing": paperless,
        "payment_method":    payment_method,
        "internet_service":  internet_service,
        "online_security":   online_security,
        "tech_support":      tech_support,
        "streaming_tv":      streaming_tv,
        "streaming_movies":  streaming_movies,
        "num_products":      num_products,
        "monthly_charges":   monthly_charges,
        "total_charges":     total_charges,
        "support_calls":     support_calls,
        "churn":             churn,
    })

    print(f"✅ Generated {n:,} records | Churn rate: {churn.mean():.1%}")
    return df


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load raw data from CSV, or generate synthetic data if not found."""
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "raw" / "churn.csv"
    path = Path(path)

    if path.exists():
        df = pd.read_csv(path)
        print(f"📂 Loaded {len(df):,} records from {path}")
        return df

    print(f"⚠️  No data found at {path}. Generating synthetic dataset...")
    df = generate_churn_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved to {path}")
    return df


if __name__ == "__main__":
    load_raw_data()
