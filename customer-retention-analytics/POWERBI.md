# Power BI Dashboard Guide

A `.pbix` is a binary file Power BI Desktop owns — it can't be generated
from a script, so this is a build-ready spec: import the CSVs below,
paste in the DAX measures, and follow the page layout. Should take
15–20 minutes in Power BI Desktop.

## 1. Data sources

Import from `outputs/`:
- `customer_retention_predictions.csv` (main fact table — one row per customer)
- `feature_importance.csv`
- `model_comparison.csv`
- `churn_by_segment.csv`

Set `risk_band` as a categorical column with a fixed sort order
(Low → Medium → High) via **Column tool → Sort by column** against a
helper index column (Low=1, Medium=2, High=3).

## 2. DAX measures

```dax
Total Customers = COUNTROWS(customer_retention_predictions)

Churned Customers = CALCULATE([Total Customers], customer_retention_predictions[churn] = "Yes")

Churn Rate = DIVIDE([Churned Customers], [Total Customers])

High Risk Customers = CALCULATE([Total Customers], customer_retention_predictions[risk_band] = "High")

High Risk Rate = DIVIDE([High Risk Customers], [Total Customers])

Avg Monthly Charges = AVERAGE(customer_retention_predictions[monthly_charges])

Avg Satisfaction Score = AVERAGE(customer_retention_predictions[satisfaction_score])

Revenue at Risk = 
CALCULATE(
    SUM(customer_retention_predictions[monthly_charges]),
    customer_retention_predictions[risk_band] = "High"
)
```

## 3. Pages

**Page 1 — Executive KPI View**
- KPI cards: Total Customers, Churn Rate, High Risk Customers, Revenue at Risk
- Donut: customers by risk_band
- Line/area: churn rate trend if you add a date dimension later (current data is a snapshot, so treat this as a placeholder panel)

**Page 2 — Risk Driver View**
- Horizontal bar: `feature_importance.csv` sorted descending (top 10)
- Clustered bar: churn rate by `contract_type`
- Clustered bar: churn rate by `payment_method`
- Slicers: `internet_service`, `paperless_billing`

**Page 3 — Customer Segment View**
- Matrix: `contract_type` (rows) × `risk_band` (columns), values = customer count and churn rate, sourced from `churn_by_segment.csv`
- Scatter: `tenure_months` vs `retention_risk_probability`, color by `risk_band`
- Table: top 25 highest-risk customers, sorted by `retention_risk_probability` descending

**Page 4 — Model Performance** (optional, good for a technical audience)
- Table from `model_comparison.csv`
- KPI cards: best model's Accuracy / ROC-AUC (Logistic Regression, ROC-AUC 0.739)

## 4. Publish

Publish to Power BI Service, then embed the public link (or a screenshot)
in `README.md` under **Live Demo**.
