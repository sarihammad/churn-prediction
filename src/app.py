import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import joblib

# XGBoost is usually more powerful, but Logistic Regression performed better here. It's also simpler, interpretable, and easier to explain to stakeholders.
model = joblib.load("logistic_model.pkl")
columns = joblib.load("model_columns.pkl")

explainer = shap.Explainer(model, feature_names=columns)

st.title("Customer Churn Predictor")

st.markdown("Enter customer information to predict churn risk.")

def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 0, 150, 50)
    total_charges = st.slider("Total Charges", 0, 10000, 500)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                               "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", [0, 1])
    
    data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaymentMethod': payment,
        'PaperlessBilling': paperless
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

df_encoded = pd.get_dummies(input_df)
df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

churn_prob = model.predict_proba(df_encoded)[0][1]
st.subheader(f"Predicted Churn Probability: `{churn_prob:.2%}`")

shap_values = explainer(df_encoded)
st.subheader("Top Factors Driving This Prediction")
st.pyplot(shap.plots.waterfall(shap_values[0], show=False))