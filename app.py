import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("house_price_model_lr.pkl")

# Title and description
st.title("🏠 Bengaluru House Price Predictor")
st.write("Enter the details of the house and get an estimated price in lakhs!")

# Sidebar inputs for features
st.sidebar.header("House Details")

# Example options from your dataset (replace with your own unique values if needed)
area_type_options = ["Super built-up Area", "Built-up Area", "Plot Area", "Carpet Area"]
location_options = ["Whitefield", "Indira Nagar", "Koramangala", "Jayanagar", "Hebbal"]  # shorten list
size_options = ["1 BHK", "2 BHK", "3 BHK", "4 BHK"]
availability_options = ["Ready To Move", "Immediate Possession", "Under Construction"]

area_type = st.sidebar.selectbox("Area Type", area_type_options)
location = st.sidebar.selectbox("Location", location_options)
size = st.sidebar.selectbox("Size", size_options)
size_sqft = st.sidebar.number_input("Total Sqft", min_value=200.0, max_value=10000.0, step=50.0)
bath = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, step=1)
balcony = st.sidebar.number_input("Balconies", min_value=0, max_value=5, step=1)
availability = st.sidebar.selectbox("Availability", availability_options)

# Button for prediction
if st.button("Predict Price"):
    # Prepare input dataframe similar to training
    input_df = pd.DataFrame({
        'Size_sqft': [size_sqft],
        'bath': [bath],
        'balcony': [balcony],
        'area_type': [area_type],
        'location': [location],
        'Size': [size],
        'availability': [availability]
    })

    # One-hot encode categorical variables (must match training)
    input_df = pd.get_dummies(input_df, columns=["area_type","location","Size","availability"], drop_first=True)

    # Add missing columns with 0 (important!)
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure the same column order
    input_df = input_df[model.feature_names_in_]

    # Predict
    price_pred = model.predict(input_df)[0]

    st.success(f"🏡 Estimated House Price: ₹ {price_pred:.2f} lakhs")
