# Churn Predictor

An end-to-end machine learning project to predict customer churn in a telecommunications company. This project covers everything from data exploration and cleaning to model building, explainability with SHAP, and a live interactive dashboard using Streamlit.

## Live Demo

Try the interactive churn prediction dashboard live here:
[Customer Churn Predictor - Streamlit App](https://2n9wy5rztnl2upkvcsnj5k.streamlit.app/)

## Problem Statement

Customer churn is a critical problem in subscription-based businesses. This project aims to predict which customers are at risk of leaving, so that retention strategies can be applied proactively.

## Project Steps

### 1. Data Loading & Cleaning

- Used [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Handled missing values and incorrect data types
- Converted target to binary

### 2. EDA & Feature Engineering

- Visualized churn rates across services, contracts, and payment methods
- Engineered features like:
  - `NumServices` (number of active services)
  - `TenureGroup` (grouping tenure months)
- One-hot encoded all categorical variables

### 3. Modeling & Evaluation

- Trained 3 models: Logistic Regression, Random Forest, XGBoost
- Evaluated with accuracy, precision, recall, and ROC AUC
- **Logistic Regression outperformed the rest** (ROC AUC: 0.83)

### 4. Streamlit Dashboard

- Interactive churn prediction tool
- Input customer details → get churn probability
- SHAP-based explanation of top features driving prediction

## Results

| Model             | Accuracy | Recall (Churn) | ROC AUC |
|------------------|----------|----------------|---------|
| LogisticRegression | **0.80** | **0.52**         | **0.83**   |
| Random Forest     | 0.78     | 0.50           | 0.82    |
| XGBoost           | 0.77     | 0.52           | 0.81    |

## Business Insight

Churn is highest among customers who:

- Are on month-to-month contracts
- Have lower tenure
- Use fewer additional services
- Use electronic checks for billing

Retention strategies can be focused on these segments.
