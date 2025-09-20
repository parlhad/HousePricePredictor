import streamlit as st
import pandas as pd
import joblib
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Luxe Estate AI Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Model & Asset Loading ---
@st.cache_resource
def load_model_assets():
    """Loads the trained model and model columns from disk using joblib."""
    try:
        # --- THIS IS THE LINE THAT WAS CHANGED ---
        model = joblib.load('final_model.joblib') 
        # -----------------------------------------
        model_columns = joblib.load('model_columns.joblib')
        return model, model_columns
    except FileNotFoundError:
        # This error is shown on the main page if files are missing
        return None, None

model, model_columns = load_model_assets()

# --- Advanced CSS Styling ---
st.markdown("""
    <style>
        /* --- General App Styling --- */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        body {
            font-family: 'Roboto', sans-serif;
        }

        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
        }

        /* --- Title Styling --- */
        h1 {
            color: #1E3A8A; /* Deep Blue */
            text-align: center;
            font-weight: 700;
        }
        .stMarkdown p {
             text-align: center;
             color: #4A5568; /* Gray */
        }

        /* --- Input Card Styling --- */
        .input-container {
            background: rgba(255, 255, 255, 0.6);
            border-radius: 20px;
            padding: 2rem 3rem;
            margin: 2rem auto;
            max-width: 1000px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        /* --- Button Styling --- */
        .stButton>button {
            background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
            color: white;
            border-radius: 50px;
            padding: 12px 30px;
            font-size: 18px;
            font-weight: 700;
            border: none;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 15px 0 rgba(71, 118, 230, 0.75);
            display: block;
            margin: 1.5rem auto 0 auto; /* Center the button */
        }
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px 0 rgba(142, 84, 233, 0.6);
        }

        /* --- Result Metric Styling --- */
        .stMetric {
            background-color: #FFFFFF;
            border-left: 10px solid #4776E6;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

    </style>
""", unsafe_allow_html=True)


# --- Main Application Header ---
st.title("💎 Luxe Estate AI Predictor")
st.markdown("Provide property details below to receive a real-time market valuation powered by our predictive model.")
st.markdown("---")

# --- Check if model files are loaded ---
if model is None or model_columns is None:
    # --- UPDATED ERROR MESSAGE ---
    st.error("🔴 Critical Error: Model files ('final_model.joblib', 'model_columns.joblib') not found. Please ensure they are in the app's root directory.")
else:
    # --- Input Section ---
    # We wrap the inputs in a div with a custom class for styling
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.subheader("Property Feature Inputs")

    # IMPORTANT: Replace this list with the actual cities from your training data
    CITIES_IN_MODEL = [
        'San Luis', 'Yorba Linda', 'Anaheim', 'Fullerton', 'Brea',
        'Newport Beach', 'Irvine', 'Santa Ana', 'Costa Mesa'
    ]

    # Row 1: Area, Bedrooms, Bathrooms, Parking
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        area = st.number_input("Area (sqft)", min_value=500, max_value=30000, value=2500, step=100)
    with col2:
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=4)
    with col3:
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=3)
    with col4:
        parking = st.slider("Parking Spots", min_value=0, max_value=5, value=3)

    # Row 2: Mainroad, Basement
    col5, col6 = st.columns(2)
    with col5:
        mainroad = st.selectbox("On Main Road?", ("Yes", "No"), index=0)
    with col6:
        basement = st.selectbox("Has Basement?", ("No", "Yes"), index=0)

    # Row 3: City and Street (Dropdown for city, text for street)
    col7, col8 = st.columns(2)
    with col7:
        city = st.selectbox("City", options=CITIES_IN_MODEL, help="Select the city from the list. This is a crucial feature for prediction.")
    with col8:
        street = st.text_input("Street Address", placeholder="e.g., 921 Isabella Way", help="Street address is for reference and not used in this model.")

    # A single, centered predict button
    predict_button = st.button("Calculate Estimated Value")

    st.markdown('</div>', unsafe_allow_html=True) # Close the input-container div


    # --- Prediction Logic and Display ---
    if predict_button:
        input_data = {
            'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
            'mainroad': 1 if mainroad == 'Yes' else 0,
            'basement': 1 if basement == 'Yes' else 0,
            'parking': parking, 'city': city,
        }
        input_df = pd.DataFrame([input_data])
        input_df_encoded = pd.get_dummies(input_df, columns=['city'])
        final_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)

        with st.spinner('AI is crunching the numbers...'):
            time.sleep(1)
            prediction = model.predict(final_df)

        st.markdown("---")
        st.subheader("Prediction Result")
        
        col_img, col_price = st.columns([1, 2])
        with col_img:
            st.image("https://i.imgur.com/J2g4Vha.png", width=250)
        with col_price:
            formatted_price = f"₹ {prediction[0]:,.0f}"
            st.metric(label="Estimated Property Value", value=formatted_price)
            st.success("This prediction is based on the provided features and historical market data.")
        
        st.balloons()

