import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_retention.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["target"] = (df["churn"] == "Yes").astype(int)

X = df.drop(columns=["customer_id", "churn", "target"])
y = df["target"]

numeric_features = [
    "tenure_months", "monthly_charges", "total_charges", "support_tickets",
    "avg_monthly_usage_gb", "num_products", "satisfaction_score"
]
categorical_features = [
    "contract_type", "internet_service", "paperless_billing", "payment_method"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=250, max_depth=8, min_samples_leaf=4, random_state=42
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

results = []

best_model_name = None
best_auc = -1
best_pipeline = None

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4)
    }
    results.append(metrics)

    if metrics["roc_auc"] > best_auc:
        best_auc = metrics["roc_auc"]
        best_model_name = name
        best_pipeline = pipeline

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

scored = df.copy()
scored["retention_risk_probability"] = best_pipeline.predict_proba(X)[:, 1]
scored["risk_band"] = pd.cut(
    scored["retention_risk_probability"],
    bins=[0, 0.35, 0.65, 1],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)
scored.to_csv(os.path.join(OUTPUT_DIR, "customer_retention_predictions.csv"), index=False)

rf = best_pipeline.named_steps["model"]
feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

if hasattr(rf, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
else:
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": best_pipeline.named_steps["model"].coef_[0]
    }).assign(importance=lambda d: d["importance"].abs()).sort_values("importance", ascending=False)

feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

segment_summary = (
    scored.groupby(["contract_type", "risk_band"], observed=False)
    .agg(customers=("customer_id", "count"),
         avg_monthly_charges=("monthly_charges", "mean"),
         churn_rate=("target", "mean"))
    .reset_index()
)
segment_summary["avg_monthly_charges"] = segment_summary["avg_monthly_charges"].round(2)
segment_summary["churn_rate"] = segment_summary["churn_rate"].round(4)
segment_summary.to_csv(os.path.join(OUTPUT_DIR, "churn_by_segment.csv"), index=False)

print("Best model:", best_model_name)
print(results_df)
print("Files written to:", OUTPUT_DIR)
