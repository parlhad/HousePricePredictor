import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Load the trained model and model columns ---
MODEL_PATH = "model.joblib"
MODEL_COLUMNS_PATH = "model_columns.joblib"

if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_COLUMNS_PATH):
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(MODEL_COLUMNS_PATH)
else:
    st.error("⚠️ Model files not found. Please make sure model.joblib and model_columns.joblib are available.")
    st.stop()

# --- App title ---
st.title("🏡 Real Estate Price Predictor")
st.markdown("Provide the property details in the sidebar to predict the house price.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Enter Property Details")

    # Cities and streets (replace with actual from your data/model)
    CITIES_IN_MODEL = [
        'San Luis', 'Yorba Linda', 'Anaheim', 'Fullerton', 'Brea',
        'Newport Beach', 'Irvine', 'Santa Ana', 'Costa Mesa'
    ]

    STREETS_IN_MODEL = [
        'Isabella Way', 'Harbor Blvd', 'Main Street', 'Sunset Ave', 'Broadway'
    ]

    # --- Input Fields (increment/decrement buttons instead of sliders) ---
    area = st.number_input("Area (sqft)", min_value=500, max_value=30000, value=2500, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=4, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=3, step=1)

    st.markdown("---")

    mainroad = st.selectbox("Is it on a Main Road?", ("Yes", "No"), index=0)
    basement = st.selectbox("Does it have a Basement?", ("No", "Yes"), index=0)
    parking = st.number_input("Parking Spots", min_value=0, max_value=5, value=3, step=1)

    st.markdown("---")

    city = st.selectbox("City", options=CITIES_IN_MODEL)
    street = st.selectbox("Street", options=STREETS_IN_MODEL)  # Dropdown instead of text box

    predict_button = st.button("Predict Price", use_container_width=True)

# --- Prediction Logic ---
if predict_button:
    try:
        # Prepare input data
        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'mainroad': 1 if mainroad == 'Yes' else 0,
            'basement': 1 if basement == 'Yes' else 0,
            'parking': parking,
            'city': city
            # street is not used in model unless trained with it
        }

        # One-hot encode categorical variables to match model training
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)

        # Predict price
        prediction = model.predict(input_encoded)[0]

        # Display results
        st.success(f"🏠 Predicted House Price: *${prediction:,.2f}*")
        st.info(f"📍 Location: {street}, {city}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
