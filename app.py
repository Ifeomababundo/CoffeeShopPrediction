# app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load Models, Scaler, and Model Scores
def load_resources():
    with open('models/linear_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler, feature_names = pickle.load(f)
    with open('models/model_scores.pkl', 'rb') as f:
        model_scores = pickle.load(f)
    return lr_model, rf_model, scaler, feature_names, model_scores

# Initialize resources
lr_model, rf_model, scaler, feature_names, model_scores = load_resources()

# Streamlit App Title
st.title("Coffee Shop Sales Prediction App")

# Header
st.header("Predict Revenue Using Machine Learning Models")

# Input Features
transaction_qty = st.number_input("Transaction Quantity", min_value=0, step=1, value=10)
unit_price = st.number_input("Unit Price", min_value=0.0, step=0.1, value=5.0)
hour = st.slider("Hour of the Day", min_value=0, max_value=23, value=12)
month = st.slider("Month", min_value=1, max_value=12, value=6)

# One-hot Encoding for Days of the Week
day_of_week = st.selectbox(
    "Day of the Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
day_columns = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_input = [1 if day == day_of_week else 0 for day in day_columns]

# Combine Inputs into DataFrame
input_data = pd.DataFrame(
    [[transaction_qty, unit_price, hour, month] + day_input],
    columns=feature_names
)

# Scale Input Data
scaled_data = scaler.transform(input_data)

# Model Selection
models_selected = st.multiselect(
    "Choose Models for Prediction:",
    ["Linear Regression", "Random Forest"],
    default=["Linear Regression", "Random Forest"]
)

# Predict Button
if st.button("Predict and Compare"):
    results = []
    if "Linear Regression" in models_selected:
        lr_prediction = lr_model.predict(scaled_data)[0]
        lr_score = model_scores.get("Linear Regression", "N/A")
        results.append(("Linear Regression", lr_prediction, lr_score))
    if "Random Forest" in models_selected:
        rf_prediction = rf_model.predict(scaled_data)[0]
        rf_score = model_scores.get("Random Forest", "N/A")
        results.append(("Random Forest", rf_prediction, rf_score))

    # Display Results
    st.write("### Model Comparisons")
    for model_name, prediction, score in results:
        st.write(f"**{model_name}:**")
        st.write(f"- Predicted Revenue: ${prediction:.2f}")
        st.write(f"- Validation R² Score: {score:.2f}")
