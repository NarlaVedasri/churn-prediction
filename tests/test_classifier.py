"""
tests/test_classifier.py
-----------------------
Unit tests for the classification pipeline.
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.loader import generate_churn_data
from data.preprocessor import prepare_data
from models.classifier import build_models, get_feature_importance


@pytest.fixture(scope="module")
def prepared_data():
    df = generate_churn_data(n=800, seed=42)
    return prepare_data(df, test_size=0.2)


def test_build_models():
    models = build_models()
    assert len(models) == 3
    assert "Random Forest" in models
    assert "XGBoost" in models
    assert "Logistic Regression" in models


def test_model_fit_predict(prepared_data):
    X_train, X_test, y_train, y_test, _ = prepared_data
    models = build_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        assert len(preds) == len(X_test), f"{name}: prediction count mismatch"
        assert set(preds).issubset({0, 1}), f"{name}: predictions should be binary"
        assert probs.shape == (len(X_test), 2), f"{name}: proba shape should be (n, 2)"
        assert np.allclose(probs.sum(axis=1), 1.0), f"{name}: probabilities must sum to 1"


def test_accuracy_above_baseline(prepared_data):
    """Model should beat the majority-class baseline."""
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    X_train, X_test, y_train, y_test, _ = prepared_data
    baseline = max(y_test.mean(), 1 - y_test.mean())

    model = XGBClassifier(n_estimators=50, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    assert acc > baseline, f"XGBoost accuracy {acc:.3f} should beat baseline {baseline:.3f}"
    assert acc > 0.70, f"Expected accuracy > 70%, got {acc:.3f}"


def test_feature_importance_output(prepared_data):
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test, _ = prepared_data
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)

    fi = get_feature_importance(model, list(X_train.columns))
    assert isinstance(fi, pd.DataFrame)
    assert "feature" in fi.columns and "importance" in fi.columns
    assert len(fi) == X_train.shape[1]
    assert fi["importance"].sum() > 0
