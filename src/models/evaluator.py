"""
src/models/evaluator.py
-----------------------
Model evaluation utilities: ROC curves, confusion matrices, SHAP values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_recall_curve
)


def plot_roc_curves(results: dict, y_test: pd.Series, figsize=(8, 5)):
    """Plot ROC curves for all trained models."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#E63946", "#2A9D8F", "#F4A261"]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Churn Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return fig


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray,
                           model_name: str = "", figsize=(5, 4)):
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["No Churn", "Churn"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "black")
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 15, figsize=(8, 6)):
    """Horizontal bar chart of feature importances."""
    df = fi_df.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    ax.barh(df["feature"], df["importance"], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    return fig


def plot_precision_recall(results: dict, y_test: pd.Series, figsize=(8, 5)):
    """Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#E63946", "#2A9D8F", "#F4A261"]

    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name} (AUC = {pr_auc:.3f})")

    baseline = y_test.mean()
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    return fig


def results_summary_table(results: dict) -> pd.DataFrame:
    """Return a DataFrame summarizing all model metrics."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  f"{res['accuracy']:.3f}",
            "AUC-ROC":   f"{res['auc_roc']:.3f}",
            "F1-Score":  f"{res['f1_score']:.3f}",
        })
    return pd.DataFrame(rows)
