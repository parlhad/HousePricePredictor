import streamlit as st
import pandas as pd
import joblib

# --- LOAD THE SAVED FILES ---
# Make sure 'final_model.joblib' and 'preprocessor.joblib' are in the same folder as your app
try:
    model = joblib.load('model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please ensure they are in the correct directory.")
    st.stop()


st.title("House Price Predictor 🏠")

# --- GET USER INPUT ---
# Example using st.number_input and st.text_input
area = st.number_input("Area (sqft)", min_value=500, max_value=25000, value=1500)
bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 4, 2)
mainroad = st.selectbox("Is it on a Main Road?", ("Yes", "No"))
basement = st.selectbox("Does it have a Basement?", ("Yes", "No"))
parking = st.slider("Number of Parking Spaces", 0, 4, 1)
citi = st.text_input("City", "San Diego, CA")
street = st.text_input("Street Address", "938 Opal St")


# --- PREDICT BUTTON ---
if st.button("Predict Price"):

    # 1. Create a DataFrame from the user input
    # The column names MUST match the names used during training
    feature_columns = ['area', 'bedrooms', 'bathrooms', 'mainroad', 'basement', 'parking', 'citi', 'street']
    
    user_input = pd.DataFrame([[
        area,
        bedrooms,
        bathrooms,
        1 if mainroad == 'Yes' else 0, # Convert yes/no to 1/0
        1 if basement == 'Yes' else 0, # Convert yes/no to 1/0
        parking,
        citi,
        street
    ]], columns=feature_columns)

    # 2. Transform the user input using the LOADED preprocessor
    # This is the crucial step!
    user_input_processed = preprocessor.transform(user_input)

    # 3. Make the prediction
    predicted_price = model.predict(user_input_processed)

    # 4. Display the result
    st.success(f"Predicted House Price: ₹ {predicted_price[0]:,.0f}")
