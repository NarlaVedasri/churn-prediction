# 🔄 Customer Churn Prediction & Analytics Dashboard

> **Fall 2024** | ML Pipelines · Classification · Clustering · LLM · Python

An end-to-end machine learning system for predicting customer churn, segmenting customers, and delivering AI-driven business recommendations through an interactive dashboard.

---

## 📌 Project Highlights

- **~80% Classification Accuracy** using ensemble models (Random Forest, XGBoost, Logistic Regression)
- **Customer Segmentation** via K-Means and DBSCAN clustering
- **LLM-Augmented Dashboard** providing plain-English business recommendations via OpenAI API
- **Full ML Pipeline** from raw data → feature engineering → model training → evaluation → deployment

---

## 🗂️ Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned & feature-engineered data
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb  # Data Cleaning & Feature Engineering
│   ├── 03_Classification.ipynb # Churn Prediction Models
│   ├── 04_Clustering.ipynb     # Customer Segmentation
│   └── 05_LLM_Insights.ipynb  # LLM-Augmented Recommendations
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loading utilities
│   │   └── preprocessor.py     # Cleaning & feature engineering
│   ├── models/
│   │   ├── classifier.py       # Classification pipeline
│   │   ├── clustering.py       # Clustering pipeline
│   │   └── evaluator.py        # Model evaluation utilities
│   └── visualization/
│       └── plotter.py          # Reusable chart functions
├── dashboard/
│   └── app.py                  # Streamlit dashboard app
├── tests/
│   ├── test_preprocessor.py
│   └── test_classifier.py
├── docs/
│   └── architecture.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction
```

### 2. Set Up Environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 4. Generate Sample Data & Train Models
```bash
python src/data/loader.py          # Generate synthetic data
python src/models/classifier.py    # Train classification models
python src/models/clustering.py    # Run customer segmentation
```

### 5. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🧠 ML Pipeline Overview

### Classification (Churn Prediction)
| Model | Accuracy | AUC-ROC | F1-Score |
|---|---|---|---|
| Random Forest | ~80% | ~0.87 | ~0.79 |
| XGBoost | ~81% | ~0.88 | ~0.80 |
| Logistic Regression | ~76% | ~0.83 | ~0.74 |

### Clustering (Customer Segmentation)
- **K-Means**: 4 optimal clusters (Elbow Method + Silhouette Score)
- **DBSCAN**: Density-based outlier-aware segmentation

### LLM Integration
- GPT-3.5/4 API generates human-readable insights per customer segment
- Converts model metrics into actionable business recommendations

---

## 📊 Features

- **Data Pipeline**: Automated cleaning, encoding, scaling, and feature engineering
- **Interactive Dashboard**: Filter by segment, view churn probability scores, download reports
- **Explainability**: SHAP values for feature importance
- **Alerts**: Flag high-risk customers (churn probability > 0.7)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| ML/Data | `scikit-learn`, `xgboost`, `pandas`, `numpy` |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| LLM | `openai` |
| Explainability | `shap` |
| Testing | `pytest` |

---

👩‍💻 Created By

Vedasri Narla

Feel free to ⭐ star or fork the project if you found it interesting!

## 📄 License

MIT License — free to use, modify, and distribute.
