# app.py
import streamlit as st
import numpy as np
import pickle

# Load Models and Scaler
def load_models_and_scaler():
    with open('models/linear_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:  # Load the saved scaler
        scaler = pickle.load(f)
    return lr_model, rf_model, scaler

# Initialize models and scaler
lr_model, rf_model, scaler = load_models_and_scaler()

# Streamlit App Title
st.title("Coffee Shop Sales Prediction App")

# Header
st.header("Predict Revenue Using Machine Learning Models")

# Subheader
st.subheader("Enter the details below:")

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

# Combine Inputs
input_data = [transaction_qty, unit_price, hour, month] + day_input
input_data = np.array(input_data).reshape(1, -1)

# Scale Input Data Using the Saved Scaler
scaled_data = scaler.transform(input_data)  # Use the loaded scaler

# Predict Button
if st.button("Predict Revenue"):
    # Linear Regression Prediction
    lr_prediction = lr_model.predict(scaled_data)
    
    # Random Forest Prediction
    rf_prediction = rf_model.predict(scaled_data)
    
    # Display Predictions
    st.write(f"Linear Regression Predicted Revenue: ${lr_prediction[0]:.2f}")
    st.write(f"Random Forest Predicted Revenue: ${rf_prediction[0]:.2f}")
