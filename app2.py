
import streamlit as st
import pandas as pd
import joblib
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching the Model and Columns ---
# Using st.cache_resource to load the model and columns only once
@st.cache_resource
def load_model_assets():
    """Loads the trained model and the list of model columns from disk."""
    try:
        model = joblib.load('model.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'model.joblib' and 'model_columns.joblib' are in the same directory.")
        return None, None

model, model_columns = load_model_assets()

# --- UI Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #F5F5F5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
        }
        .stMetric {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 12px;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# --- Main Application ---
st.title("🏡 Real Estate Price Predictor")
st.markdown("Enter the details of a property to get an estimated market value. Our advanced model provides real-time predictions based on your inputs.")
st.markdown("---")


# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Enter Property Details")

    # IMPORTANT: Replace this list with the actual cities from your training data
    # The order does not matter, but the names must be identical.
    CITIES_IN_MODEL = [
        'San Luis', 'Yorba Linda', 'Anaheim', 'Fullerton', 'Brea',
        'Newport Beach', 'Irvine', 'Santa Ana', 'Costa Mesa'
    ]

    # --- Input Fields ---
    area = st.slider("Area (sqft)", min_value=500, max_value=30000, value=2500, step=100)
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=4)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=3)

    st.markdown("---") # Visual separator

    mainroad = st.selectbox("Is it on a Main Road?", ("Yes", "No"), index=0)
    basement = st.selectbox("Does it have a Basement?", ("No", "Yes"), index=0)
    parking = st.slider("Parking Spots", min_value=0, max_value=5, value=3)

    st.markdown("---")

    city = st.selectbox("City", options=CITIES_IN_MODEL)
    street = st.text_input("Street Address", placeholder="e.g., 921 Isabella Way") # Note: Street is often not used in models

    predict_button = st.button("Predict Price", use_container_width=True)


# --- Prediction Logic and Display ---
if predict_button and model is not None:
    # 1. Create a dictionary from user inputs
    input_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'mainroad': 1 if mainroad == 'Yes' else 0,
        'basement': 1 if basement == 'Yes' else 0,
        'parking': parking,
        'city': city,
        # 'street' is not included as it's typically not a feature in the model
        # If your model uses it, you must handle it similarly to 'city'.
    }

    # 2. Convert to a DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. One-Hot Encode the 'city' column
    input_df_encoded = pd.get_dummies(input_df, columns=['city'])

    # 4. Align columns with the model's training columns
    # This is the CRUCIAL step to avoid errors. It adds missing columns (with value 0)
    # and ensures the order is the same as the model expects.
    final_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # 5. Make the prediction
    with st.spinner('Calculating...'):
        time.sleep(1) # Simulate a small delay for better user experience
        prediction = model.predict(final_df)

    # 6. Display the result
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://i.imgur.com/J2g4Vha.png", width=200) # A simple house icon

    with col2:
        st.subheader("Predicted Property Value")
        # Format the prediction as currency with the Rupee symbol
        formatted_price = f"₹ {prediction[0]:,.0f}"
        st.metric(label="Estimated Price", value=formatted_price)
        st.success("The prediction is based on the features provided. Market conditions can influence the final price.")

    st.balloons()

elif not model:
    st.warning("Please make sure the model files are loaded correctly before trying to predict.")
