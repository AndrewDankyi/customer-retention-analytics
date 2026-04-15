# Customer Retention Analytics

## Overview
This project analyzes customer churn and builds predictive models to identify the strongest drivers of retention. The workflow combines Python-based machine learning with Tableau-ready output files so business teams can act on the results.

## Business Goal
Reduce churn by identifying:
- which customers are most at risk of leaving
- which product, billing, and service factors are associated with churn
- which retention actions are likely to have the highest impact

## Tech Stack
- Python
- pandas
- scikit-learn
- matplotlib
- Tableau (dashboard-ready exports)

## Project Structure
```text
customer-retention-analytics/
├── data/
│   └── customer_retention.csv
├── notebooks/
│   └── retention_analysis.ipynb
├── outputs/
│   ├── customer_retention_predictions.csv
│   ├── feature_importance.csv
│   └── churn_by_segment.csv
├── src/
│   └── retention_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Problem Statement
Subscription businesses often lose revenue when high-risk customers are not identified early. This project builds a churn prediction workflow that helps customer success and marketing teams target retention campaigns more effectively.

## Methods
- Data cleaning and preprocessing
- One-hot encoding for categorical variables
- Train/test split
- Logistic Regression baseline
- Random Forest classifier
- Model evaluation using accuracy, precision, recall, F1, and ROC-AUC
- Feature importance extraction
- Segment analysis for Tableau dashboards

## Tableau Dashboard Ideas
Use the exported CSVs to build:
1. **Executive KPI View**
   - churn rate
   - high-risk customer count
   - average monthly charges
2. **Risk Driver View**
   - churn by contract type
   - churn by payment method
   - feature importance
3. **Customer Segment View**
   - at-risk customers by tenure band
   - at-risk customers by product count

## How to Run
```bash
pip install -r requirements.txt
python src/retention_model.py
```

## Results
Running the script generates:
- scored customer-level predictions
- a feature importance table
- a segment summary for visualization

## Portfolio Value
This project demonstrates:
- predictive modeling in Python
- business translation of model outputs
- dashboard-ready analytics for retention strategy
