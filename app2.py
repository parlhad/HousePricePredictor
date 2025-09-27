import streamlit as st
import pandas as pd
import joblib

# --- LOAD THE SAVED FILES ---
try:
    model = joblib.load('model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Please ensure they are in the GitHub repository.")
    st.stop()

st.title("SoCal House Price Predictor 🏠")
st.write("Enter the details of the house to get a price prediction.")

# --- CREATE INPUT FORM ---
with st.form("prediction_form"):
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (sqft)", min_value=500, max_value=25000, value=3000, step=100)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)
        parking = st.slider("Parking Spaces", 0, 4, 1)

    with col2:
        mainroad = st.selectbox("On Main Road?", options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
        basement = st.selectbox("Has Basement?", options=[1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
        citi = st.text_input("City", "San Diego, CA")
        street = st.text_input("Street Address", "938 Opal St")
    
    submit_button = st.form_submit_button(label='Predict Price')

# --- PREDICTION LOGIC ---
if submit_button:
    # 1. Create a DataFrame from the user's input.
    # The column names MUST match the ones used in your training notebook before preprocessing.
    feature_columns = ['area', 'bedrooms', 'bathrooms', 'mainroad', 'basement', 'parking', 'citi', 'street']
    
    user_input = pd.DataFrame([[
        area, bedrooms, bathrooms, mainroad, basement, parking, citi, street
    ]], columns=feature_columns)
    
    # 2. Transform the user input using the loaded preprocessor.
    # This automatically handles the one-hot encoding correctly.
    user_input_processed = preprocessor.transform(user_input)
    
    # 3. Make the prediction using the loaded model.
    predicted_price = model.predict(user_input_processed)
    
    # 4. Display the result.
    st.success(f"Predicted House Price: ${predicted_price[0]:,.0f}")
