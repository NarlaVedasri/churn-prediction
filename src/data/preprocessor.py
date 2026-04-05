"""
src/data/preprocessor.py
------------------------
Full data cleaning, feature engineering, and preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ── Column Groups ──────────────────────────────────────────────────────────────
BINARY_COLS = ["gender", "partner", "dependents", "paperless_billing"]
ORDINAL_COLS = ["contract"]
NOMINAL_COLS = ["payment_method", "internet_service", "online_security",
                "tech_support", "streaming_tv", "streaming_movies"]
NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges", "num_products",
                "support_calls", "senior_citizen"]
TARGET = "churn"
ID_COL = "customer_id"


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing for the churn dataset.

    Steps:
    1. Drop duplicates & irrelevant columns
    2. Handle missing values
    3. Binary encode yes/no columns
    4. Ordinal encode contract type
    5. One-hot encode nominal categoricals
    6. Feature engineering (derived features)
    7. Standard scale numerics
    """

    def __init__(self, scale: bool = True):
        self.scale = scale
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        df = self._clean(X.copy())
        df = self._encode(df)
        df = self._engineer(df)
        numeric = [c for c in df.columns if df[c].dtype in [np.float64, np.int64, float, int]]
        if self.scale:
            self.scaler.fit(df[numeric])
        self.feature_names_ = list(df.columns)
        self._numeric_cols = numeric
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = self._clean(X.copy())
        df = self._encode(df)
        df = self._engineer(df)
        if self.scale:
            df[self._numeric_cols] = self.scaler.transform(df[self._numeric_cols])
        # Align columns to training set
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_names_]

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove noise and fix basic issues."""
        df = df.drop_duplicates()
        if ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])

        # Coerce total_charges to numeric (can be blank in real Telco dataset)
        df["total_charges"] = pd.to_numeric(df.get("total_charges", pd.Series()), errors="coerce")
        df["total_charges"] = df["total_charges"].fillna(
            df["monthly_charges"] * df["tenure"]
        )

        df = df.dropna(subset=["tenure", "monthly_charges"])
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        # Binary: Yes/No → 1/0
        yes_no_cols = [c for c in BINARY_COLS if c in df.columns]
        for col in yes_no_cols:
            if df[col].dtype == object:
                df[col] = df[col].map({"Yes": 1, "No": 1, "Male": 1, "Female": 0}).fillna(0).astype(int)

        # Gender specifically
        if "gender" in df.columns and df["gender"].dtype == object:
            df["gender"] = (df["gender"] == "Male").astype(int)

        # Ordinal: contract
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        if "contract" in df.columns:
            df["contract"] = df["contract"].map(contract_map).fillna(0)

        # One-hot encode nominal columns
        nominal = [c for c in NOMINAL_COLS if c in df.columns]
        if nominal:
            df = pd.get_dummies(df, columns=nominal, drop_first=True, dtype=int)

        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        if "monthly_charges" in df.columns and "tenure" in df.columns:
            df["charges_per_month"] = df["monthly_charges"]
            df["lifetime_value"]    = df["monthly_charges"] * df["tenure"]
            df["avg_monthly_total"] = df.get("total_charges", df["monthly_charges"]) / (df["tenure"] + 1)

        if "support_calls" in df.columns and "tenure" in df.columns:
            df["calls_per_month"] = df["support_calls"] / (df["tenure"] + 1)

        if "num_products" in df.columns:
            df["is_multi_product"] = (df["num_products"] > 2).astype(int)

        return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Full split + preprocessing pipeline.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in DataFrame.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = ChurnPreprocessor(scale=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    print(f"📐 Train: {X_train_proc.shape} | Test: {X_test_proc.shape}")
    print(f"⚖️  Class balance — Train: {y_train.mean():.1%} churn | Test: {y_test.mean():.1%} churn")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


if __name__ == "__main__":
    from loader import load_raw_data
    df = load_raw_data()
    X_train, X_test, y_train, y_test, prep = prepare_data(df)
    print("Feature names:", prep.feature_names_[:10], "...")
