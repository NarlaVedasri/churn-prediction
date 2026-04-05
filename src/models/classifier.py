"""
src/models/classifier.py
------------------------
End-to-end churn classification pipeline.
Trains Random Forest, XGBoost, and Logistic Regression; evaluates all; saves the best.
Run: python src/models/classifier.py
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_raw_data
from data.preprocessor import prepare_data

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Model Definitions ──────────────────────────────────────────────────────────
def build_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,     # handle class imbalance
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=0.5,
        ),
    }


# ── Training & Evaluation ──────────────────────────────────────────────────────
def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, str]:
    """
    Train all models and return results dict + name of best model.
    """
    models   = build_models()
    results  = {}
    best_auc = 0
    best_name = None

    print("\n" + "=" * 60)
    print("  CLASSIFICATION PIPELINE — CHURN PREDICTION")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train, y_train)

        y_pred      = model.predict(X_test)
        y_prob      = model.predict_proba(X_test)[:, 1]
        acc         = accuracy_score(y_test, y_pred)
        auc         = roc_auc_score(y_test, y_prob)
        f1          = f1_score(y_test, y_pred)
        cm          = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model":     model,
            "accuracy":  acc,
            "auc_roc":   auc,
            "f1_score":  f1,
            "confusion_matrix": cm,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
        }

        print(f"   Accuracy : {acc:.4f}")
        print(f"   AUC-ROC  : {auc:.4f}")
        print(f"   F1-Score : {f1:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

        if auc > best_auc:
            best_auc  = auc
            best_name = name

    print(f"\n🏆 Best Model: {best_name} (AUC-ROC = {best_auc:.4f})")
    return results, best_name


# ── Feature Importance ─────────────────────────────────────────────────────────
def get_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importances from tree-based or linear models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ── Save / Load ────────────────────────────────────────────────────────────────
def save_model(model, name: str, path: Path | None = None):
    path = path or MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, path)
    print(f"💾 Saved model → {path}")


def load_model(name: str, path: Path | None = None):
    path = path or MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
    model = joblib.load(path)
    print(f"📂 Loaded model from {path}")
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def run_pipeline() -> tuple[dict, str, object, list[str]]:
    """Full pipeline: load → preprocess → train → evaluate → save best."""
    df = load_raw_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    results, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_model = results[best_name]["model"]
    save_model(best_model, best_name)
    save_model(preprocessor, "preprocessor")

    feature_names = list(X_train.columns)
    fi = get_feature_importance(best_model, feature_names)
    print(f"\n📊 Top 10 Features ({best_name}):")
    print(fi.head(10).to_string(index=False))

    return results, best_name, preprocessor, feature_names


if __name__ == "__main__":
    run_pipeline()
