import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# LOAD TRAINED PIPELINE
# -----------------------------

model = joblib.load(
    "churn_pipeline.pkl"
)

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

# -----------------------------
# USER INPUTS
# -----------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------

if st.button("Predict Churn"):

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------

    Charges_Per_Tenure = (
        TotalCharges / (tenure + 1)
    )

    HighValueCustomer = (
        1 if MonthlyCharges > 80 else 0
    )

    LongTermCustomer = (
        1 if tenure > 24 else 0
    )

    # -------------------------
    # CREATE INPUT DATAFRAME
    # -------------------------

    input_data = pd.DataFrame({

        'gender':[gender],

        'SeniorCitizen':[SeniorCitizen],

        'Partner':[Partner],

        'Dependents':[Dependents],

        'tenure':[tenure],

        'PhoneService':[PhoneService],

        'MultipleLines':[MultipleLines],

        'InternetService':[InternetService],

        'OnlineSecurity':[OnlineSecurity],

        'OnlineBackup':[OnlineBackup],

        'DeviceProtection':[DeviceProtection],

        'TechSupport':[TechSupport],

        'StreamingTV':[StreamingTV],

        'StreamingMovies':[StreamingMovies],

        'Contract':[Contract],

        'PaperlessBilling':[PaperlessBilling],

        'PaymentMethod':[PaymentMethod],

        'MonthlyCharges':[MonthlyCharges],

        'TotalCharges':[TotalCharges],

        'Charges_Per_Tenure':[
            Charges_Per_Tenure
        ],

        'HighValueCustomer':[
            HighValueCustomer
        ],

        'LongTermCustomer':[
            LongTermCustomer
        ]
    })

    # -------------------------
    # PREDICTION
    # -------------------------

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    # -------------------------
    # OUTPUT
    # -------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            f"⚠️ Customer likely to churn "
            f"({probability:.2%} probability)"
        )

    else:

        st.success(
            f"✅ Customer likely to stay "
            f"({1 - probability:.2%} probability)"
        )

    # -------------------------
    # PROBABILITY BAR
    # -------------------------

    st.progress(float(probability))

    st.write(
        f"Churn Probability: {probability:.2%}"
    )