import streamlit as st
import pandas as pd
import shap
import joblib

# load model and columns
MODEL_PATH = "src/logistic_model.pkl"
COLUMNS_PATH = "src/model_columns.pkl"
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

def get_user_input():
    st.sidebar.header("Customer Information")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 0, 150, 50)
    total_charges = st.sidebar.slider("Total Charges", 0, 10000, 500)
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless = st.sidebar.selectbox("Paperless Billing", [0, 1])
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
    return pd.DataFrame([data])

def preprocess_input(df, columns):
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)
    return df_encoded.astype(float)

def main():
    st.title("Customer Churn Predictor")
    st.markdown("Enter customer information to predict churn risk.")

    input_df = get_user_input()
    processed_df = preprocess_input(input_df, model_columns)

    churn_prob = model.predict_proba(processed_df)[0][1]
    st.subheader(f"Predicted Churn Probability: `{churn_prob:.2%}`")

    explainer = shap.Explainer(model.predict_proba, processed_df)
    shap_values = explainer(processed_df)
    st.subheader("Top Factors Driving This Prediction")
    st.pyplot(shap.plots.waterfall(shap_values[0], show=False))

if __name__ == "__main__":
    main()