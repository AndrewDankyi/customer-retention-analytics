"""
Streamlit app for Customer Retention Analytics.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud by pointing it at this repo
and this file as the entrypoint.
"""

import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "outputs", "customer_retention_predictions.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "outputs", "feature_importance.csv")
MODEL_COMPARISON_PATH = os.path.join(BASE_DIR, "outputs", "model_comparison.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best_model.joblib")

st.set_page_config(page_title="Customer Retention Analytics", layout="wide")


@st.cache_data
def load_outputs():
    predictions = pd.read_csv(PREDICTIONS_PATH)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    model_comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    return predictions, feature_importance, model_comparison


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


predictions, feature_importance, model_comparison = load_outputs()
model = load_model()

st.title("📉 Customer Retention Analytics")
st.caption("Predictive churn model with risk scoring, segment breakdown, and a live scoring tool.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(predictions):,}")
col2.metric("Churn Rate", f"{predictions['target'].mean():.1%}")
col3.metric("High Risk", f"{(predictions['risk_band'] == 'High').sum():,}")
best_row = model_comparison.sort_values("roc_auc", ascending=False).iloc[0]
col4.metric("Best Model ROC-AUC", f"{best_row['roc_auc']:.3f}", best_row["model"].replace("_", " ").title())

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk Explorer", "Model Performance", "Score a Customer"])

with tab1:
    st.subheader("Churn Rate by Contract Type and Risk Band")
    segment = (
        predictions.groupby(["contract_type", "risk_band"], observed=False)["target"]
        .mean()
        .reset_index(name="churn_rate")
    )
    fig = px.bar(segment, x="contract_type", y="churn_rate", color="risk_band", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Drivers of Churn")
    top_features = feature_importance.head(10).copy()
    top_features["feature"] = top_features["feature"].str.replace("num__|cat__", "", regex=True)
    fig2 = px.bar(top_features, x="importance", y="feature", orientation="h")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Filter Customers by Risk")
    risk_filter = st.multiselect(
        "Risk band", options=predictions["risk_band"].unique().tolist(),
        default=predictions["risk_band"].unique().tolist(),
    )
    contract_filter = st.multiselect(
        "Contract type", options=predictions["contract_type"].unique().tolist(),
        default=predictions["contract_type"].unique().tolist(),
    )
    filtered = predictions[
        predictions["risk_band"].isin(risk_filter) & predictions["contract_type"].isin(contract_filter)
    ]
    st.dataframe(
        filtered[[
            "customer_id", "contract_type", "tenure_months", "monthly_charges",
            "satisfaction_score", "retention_risk_probability", "risk_band",
        ]].sort_values("retention_risk_probability", ascending=False),
        use_container_width=True,
        height=420,
    )
    st.download_button(
        "Download filtered results (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_retention_risk.csv",
    )

with tab3:
    st.subheader("Model Comparison")
    st.dataframe(model_comparison, use_container_width=True)
    fig3 = px.bar(
        model_comparison.melt(id_vars="model", var_name="metric", value_name="score"),
        x="metric", y="score", color="model", barmode="group",
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Score a Hypothetical Customer")
    if model is None:
        st.warning("No saved model found. Run `python src/retention_model.py` first to generate outputs/best_model.joblib.")
    else:
        c1, c2, c3 = st.columns(3)
        tenure_months = c1.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = c1.slider("Monthly charges ($)", 15.0, 150.0, 65.0)
        total_charges = c1.number_input("Total charges ($)", 0.0, 10000.0, float(tenure_months * monthly_charges))
        support_tickets = c2.slider("Support tickets", 0, 10, 1)
        avg_monthly_usage_gb = c2.slider("Avg. monthly usage (GB)", 0.0, 500.0, 150.0)
        num_products = c2.slider("Number of products", 1, 5, 2)
        satisfaction_score = c2.slider("Satisfaction score (1-5)", 1, 5, 3)
        contract_type = c3.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        internet_service = c3.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        paperless_billing = c3.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = c3.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        )

        input_row = pd.DataFrame([{
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "paperless_billing": paperless_billing,
            "support_tickets": support_tickets,
            "payment_method": payment_method,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "num_products": num_products,
            "satisfaction_score": satisfaction_score,
        }])

        if st.button("Predict churn risk"):
            prob = model.predict_proba(input_row)[0, 1]
            band = "Low" if prob < 0.35 else "Medium" if prob < 0.65 else "High"
            st.metric("Predicted churn probability", f"{prob:.1%}", band)

st.divider()
st.caption("Built by Andrew Dankyi Twum · [Portfolio](https://andrewdankyi.github.io/Portfolio/) · Data is synthetic/illustrative.")
