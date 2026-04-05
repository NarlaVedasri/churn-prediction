"""
tests/test_preprocessor.py
--------------------------
Unit tests for the data preprocessing pipeline.
Run: pytest tests/
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.loader import generate_churn_data
from data.preprocessor import ChurnPreprocessor, prepare_data, TARGET


@pytest.fixture
def sample_df():
    return generate_churn_data(n=500, seed=0)


def test_generate_churn_data():
    df = generate_churn_data(n=200, seed=42)
    assert len(df) == 200
    assert TARGET in df.columns
    assert df[TARGET].isin([0, 1]).all()
    assert df["monthly_charges"].between(0, 200).all()


def test_preprocessor_fit_transform(sample_df):
    prep = ChurnPreprocessor(scale=True)
    X = sample_df.drop(columns=[TARGET])
    X_proc = prep.fit_transform(X)

    assert isinstance(X_proc, pd.DataFrame)
    assert len(X_proc) == len(sample_df)
    assert not X_proc.isnull().any().any(), "Preprocessed data should have no NaNs"


def test_no_target_leakage(sample_df):
    prep = ChurnPreprocessor()
    X = sample_df.drop(columns=[TARGET])
    X_proc = prep.fit_transform(X)
    assert TARGET not in X_proc.columns, "Target column must not appear in features"


def test_feature_names_consistent(sample_df):
    prep = ChurnPreprocessor()
    X = sample_df.drop(columns=[TARGET])
    prep.fit(X)
    X1 = prep.transform(X.head(100))
    X2 = prep.transform(X.tail(100))
    assert list(X1.columns) == list(X2.columns), "Feature names must be consistent across transforms"


def test_prepare_data_shapes(sample_df):
    X_train, X_test, y_train, y_test, prep = prepare_data(sample_df, test_size=0.2)
    total = len(sample_df)
    assert len(X_train) + len(X_test) == total
    assert len(y_train) + len(y_test) == total
    assert X_train.shape[1] == X_test.shape[1], "Train/test must have same feature count"


def test_prepare_data_stratification(sample_df):
    _, _, y_train, y_test, _ = prepare_data(sample_df, test_size=0.2)
    train_rate = y_train.mean()
    test_rate  = y_test.mean()
    assert abs(train_rate - test_rate) < 0.05, "Stratification should balance churn rates"


def test_engineered_features(sample_df):
    prep = ChurnPreprocessor()
    X = sample_df.drop(columns=[TARGET])
    X_proc = prep.fit_transform(X)
    engineered = ["lifetime_value", "calls_per_month", "avg_monthly_total"]
    for feat in engineered:
        assert feat in X_proc.columns, f"Engineered feature '{feat}' missing"
