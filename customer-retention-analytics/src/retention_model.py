"""
retention_model.py

Trains and compares churn-prediction models for the Customer Retention
Analytics project, scores the full customer base, extracts feature
importance, builds a segment-level summary, and renders the chart set
used in the README / Power BI dashboard.

Usage:
    python src/retention_model.py
"""

import logging
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_retention.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges", "support_tickets",
    "avg_monthly_usage_gb", "num_products", "satisfaction_score",
]
CATEGORICAL_FEATURES = [
    "contract_type", "internet_service", "paperless_billing", "payment_method",
]

sns.set_theme(style="whitegrid", palette="deep")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw customer retention CSV and encode the churn target."""
    df = pd.read_csv(path)
    df["target"] = (df["churn"] == "Yes").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Assemble the shared numeric/categorical preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def get_candidate_models() -> dict:
    """Return the candidate models to compare."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=4, random_state=42
        ),
    }


def train_and_compare(X_train, X_test, y_train, y_test, preprocessor):
    """Fit each candidate model and return comparison metrics plus the best pipeline."""
    results, fitted_pipelines = [], {}

    for name, model in get_candidate_models().items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
        }
        results.append(metrics)
        fitted_pipelines[name] = pipeline
        logger.info("Trained %s | ROC-AUC: %.4f", name, metrics["roc_auc"])

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]

    logger.info("Best model: %s (ROC-AUC %.4f)", best_model_name, results_df.iloc[0]["roc_auc"])
    return results_df, best_model_name, best_pipeline


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Pull feature importance/coefficients out of a fitted pipeline, model-agnostic."""
    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = abs(model.coef_[0])

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def score_customers(df: pd.DataFrame, X: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Score every customer with the winning model and assign a risk band."""
    scored = df.copy()
    scored["retention_risk_probability"] = pipeline.predict_proba(X)[:, 1]
    scored["risk_band"] = pd.cut(
        scored["retention_risk_probability"],
        bins=[0, 0.35, 0.65, 1],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return scored


def build_segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate churn rate and charges by contract type and risk band."""
    summary = (
        scored.groupby(["contract_type", "risk_band"], observed=False)
        .agg(
            customers=("customer_id", "count"),
            avg_monthly_charges=("monthly_charges", "mean"),
            churn_rate=("target", "mean"),
        )
        .reset_index()
    )
    summary["avg_monthly_charges"] = summary["avg_monthly_charges"].round(2)
    summary["churn_rate"] = summary["churn_rate"].round(4)
    return summary


def render_visuals(results_df, best_pipeline, best_model_name, X_test, y_test, feature_importance, segment_summary):
    """Save the chart set used in the README and Power BI prep."""
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(best_pipeline, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve — {best_model_name.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix
    preds = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_model_name.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Feature importance
    top_features = feature_importance.head(10).copy()
    top_features["feature"] = top_features["feature"].str.replace("num__|cat__", "", regex=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=top_features, y="feature", x="importance", ax=ax, color="#2f6690")
    ax.set_title("Top 10 Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    # Churn rate by contract type / risk band
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=segment_summary, x="contract_type", y="churn_rate", hue="risk_band",
        ax=ax, palette="rocket",
    )
    ax.set_title("Churn Rate by Contract Type and Risk Band")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "churn_by_segment.png"), dpi=150)
    plt.close(fig)

    # Model comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    melted = results_df.melt(id_vars="model", var_name="metric", value_name="score")
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax, palette="mako")
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(title="", loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)

    logger.info("Saved 5 charts to %s", IMAGE_DIR)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading data from %s", DATA_PATH)
    df = load_data()
    X = df.drop(columns=["customer_id", "churn", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    results_df, best_model_name, best_pipeline = train_and_compare(
        X_train, X_test, y_train, y_test, preprocessor
    )
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    feature_importance = extract_feature_importance(best_pipeline)
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

    scored = score_customers(df, X, best_pipeline)
    scored.to_csv(os.path.join(OUTPUT_DIR, "customer_retention_predictions.csv"), index=False)

    segment_summary = build_segment_summary(scored)
    segment_summary.to_csv(os.path.join(OUTPUT_DIR, "churn_by_segment.csv"), index=False)

    render_visuals(results_df, best_pipeline, best_model_name, X_test, y_test, feature_importance, segment_summary)

    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Saved model pipeline to %s", MODEL_PATH)

    print("\nBest model:", best_model_name)
    print(results_df.to_string(index=False))
    print("\nAll outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
