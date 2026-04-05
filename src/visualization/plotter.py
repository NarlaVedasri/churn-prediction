"""
src/visualization/plotter.py
----------------------------
Reusable Plotly chart functions for the dashboard and notebooks.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "primary":  "#E63946",
    "success":  "#2A9D8F",
    "info":     "#457B9D",
    "warning":  "#F4A261",
    "muted":    "#6C757D",
}

SEGMENT_COLORS = {
    "Low-Risk Loyalists":  "#2A9D8F",
    "High-Value Engaged":  "#457B9D",
    "At-Risk Churners":    "#E63946",
    "New Uncertain":       "#F4A261",
}


def churn_donut(df: pd.DataFrame, target: str = "churn") -> go.Figure:
    counts = df[target].value_counts()
    fig = go.Figure(go.Pie(
        labels=["Retained", "Churned"],
        values=[counts.get(0, 0), counts.get(1, 0)],
        hole=0.5,
        marker_colors=[COLORS["success"], COLORS["primary"]],
    ))
    fig.update_layout(margin=dict(t=20, b=20), height=300,
                      showlegend=True, legend=dict(orientation="h"))
    return fig


def churn_by_feature(df: pd.DataFrame, feature: str, target: str = "churn") -> go.Figure:
    data = df.groupby(feature)[target].mean().sort_values(ascending=False).reset_index()
    data.columns = [feature, "churn_rate"]
    fig = px.bar(data, x=feature, y="churn_rate",
                 color="churn_rate", color_continuous_scale="RdYlGn_r")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=20), height=320)
    return fig


def scatter_2d_segments(df: pd.DataFrame, segment_col: str = "segment_name") -> go.Figure:
    fig = px.scatter(df, x="pca_x", y="pca_y",
                     color=segment_col,
                     color_discrete_map=SEGMENT_COLORS,
                     opacity=0.55,
                     labels={"pca_x": "PC1", "pca_y": "PC2"})
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(t=20), height=400)
    return fig


def probability_histogram(df: pd.DataFrame, prob_col: str = "churn_probability",
                           threshold: float = 0.7) -> go.Figure:
    fig = px.histogram(df, x=prob_col, nbins=50,
                       color_discrete_sequence=[COLORS["info"]])
    fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["primary"],
                  annotation_text=f"Threshold ({threshold:.0%})")
    fig.update_layout(margin=dict(t=20), height=280)
    return fig


def feature_importance_bar(fi_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    df = fi_df.head(top_n).sort_values("importance")
    fig = px.bar(df, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="RdYlGn_r")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=10),
                      height=max(300, top_n * 26))
    return fig


def elbow_plot(k_range: list, inertias: list, silhouettes: list, best_k: int) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Inertia (Elbow)", "Silhouette Score"])
    fig.add_trace(go.Scatter(x=k_range, y=inertias, mode="lines+markers",
                              line_color=COLORS["primary"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=k_range, y=silhouettes, mode="lines+markers",
                              line_color=COLORS["success"]), row=1, col=2)
    for col in [1, 2]:
        fig.add_vline(x=best_k, line_dash="dash", line_color=COLORS["warning"],
                       annotation_text=f"k={best_k}", row=1, col=col)
    fig.update_layout(showlegend=False, height=320, margin=dict(t=30))
    return fig
