# System Architecture

## Pipeline Overview

```
Raw Data (CSV / Synthetic)
        │
        ▼
┌─────────────────────┐
│   Data Loader       │  src/data/loader.py
│  • CSV ingestion    │
│  • Synthetic gen    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Preprocessor      │  src/data/preprocessor.py
│  • De-duplication   │
│  • Missing values   │
│  • Binary encoding  │
│  • One-hot encoding │
│  • Feature eng.     │
│  • Standard scaling │
└──────┬──────┬───────┘
       │      │
       ▼      ▼
┌──────────┐ ┌──────────────┐
│ Classif. │ │  Clustering  │
│ Pipeline │ │  Pipeline    │
├──────────┤ ├──────────────┤
│ RF       │ │ K-Means      │
│ XGBoost  │ │ DBSCAN       │
│ LogReg   │ │ PCA (2D viz) │
└────┬─────┘ └──────┬───────┘
     │               │
     └───────┬───────┘
             │
             ▼
    ┌─────────────────┐
    │  LLM Insights   │  src/models/llm_insights.py
    │  • Segment recs │
    │  • Retention    │
    │  • Exec summary │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Streamlit App  │  dashboard/app.py
    │  • Overview     │
    │  • Predictions  │
    │  • Segmentation │
    │  • Retention    │
    └─────────────────┘
```

## Data Schema

| Column | Type | Description |
|---|---|---|
| customer_id | str | Unique identifier |
| tenure | int | Months as customer |
| monthly_charges | float | Monthly bill ($) |
| total_charges | float | Cumulative spend ($) |
| contract | str | Month-to-month / 1yr / 2yr |
| payment_method | str | Payment channel |
| internet_service | str | DSL / Fiber / No |
| support_calls | int | Support contacts |
| num_products | int | Number of active products |
| churn | int | Target (0=retained, 1=churned) |

## Model Selection Rationale

- **XGBoost**: Best overall AUC-ROC; handles class imbalance via scale_pos_weight
- **Random Forest**: Strong baseline; robust to outliers; easy SHAP integration
- **Logistic Regression**: Interpretable baseline; good calibration

## Clustering Method

K-Means was chosen as the primary segmentation method due to:
1. Interpretable, spherical clusters suitable for business segments
2. Scalability to 5,000+ customers
3. Deterministic output (fixed random_state)

DBSCAN is included for outlier/anomaly detection — customers who don't fit any segment.
