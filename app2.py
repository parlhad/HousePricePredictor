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


# ------------------------------------------------------------------------------------
# --- CRITICAL STEP: HARD-CODED COLUMN LIST ---
# ------------------------------------------------------------------------------------
# You MUST replace the example list below with the actual list of columns
# from your training data AFTER one-hot encoding.
# To get this list, run `X_encoded.columns.tolist()` in your training notebook.
MODEL_TRAINING_COLUMNS = [
    'area',
    'bedrooms',
    'bathrooms',
    'mainroad',
    'basement',
    'parking',
    'city_Anaheim',
    'city_Brea',
    'city_Costa Mesa',
    'city_Fullerton',
    'city_Irvine',
    'city_Newport Beach',
    'city_San Luis',
    'city_Santa Ana',
    'city_Yorba Linda'
]
# ------------------------------------------------------------------------------------


# --- Model Loading (Updated to only load one file) ---
@st.cache_resource
def load_model():
    """Loads the trained model from disk using joblib."""
    try:
        model = joblib.load('final_model.joblib')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- Advanced CSS Styling ---
st.markdown("""
    <style>
        /* CSS from previous version - remains unchanged */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        body { font-family: 'Roboto', sans-serif; }
        .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; }
        h1 { color: #1E3A8A; text-align: center; font-weight: 700; }
        .stMarkdown p { text-align: center; color: #4A5568; }
        .input-container { background: rgba(255, 255, 255, 0.6); border-radius: 20px; padding: 2rem 3rem; margin: 2rem auto; max-width: 1000px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.18); }
        .stButton>button { background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%); color: white; border-radius: 50px; padding: 12px 30px; font-size: 18px; font-weight: 700; border: none; transition: all 0.3s ease-in-out; box-shadow: 0 4px 15px 0 rgba(71, 118, 230, 0.75); display: block; margin: 1.5rem auto 0 auto; }
        .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px 0 rgba(142, 84, 233, 0.6); }
        .stMetric { background-color: #FFFFFF; border-left: 10px solid #4776E6; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)


# --- Main Application Header ---
st.title("💎 Luxe Estate AI Predictor")
st.markdown("Provide property details below to receive a real-time market valuation powered by our predictive model.")
st.markdown("---")

# --- Check if model file is loaded ---
if model is None:
    st.error("🔴 Critical Error: Model file ('final_model.joblib') not found. Please ensure it is in the app's root directory.")
else:
    # --- Input Section ---
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.subheader("Property Feature Inputs")

    CITIES_IN_MODEL = [
        'San Luis', 'Yorba Linda', 'Anaheim', 'Fullerton', 'Brea',
        'Newport Beach', 'Irvine', 'Santa Ana', 'Costa Mesa'
    ]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        area = st.number_input("Area (sqft)", min_value=500, max_value=30000, value=2500, step=100)
    with col2:
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=4)
    with col3:
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=3)
    with col4:
        parking = st.slider("Parking Spots", min_value=0, max_value=5, value=3)

    col5, col6 = st.columns(2)
    with col5:
        mainroad = st.selectbox("On Main Road?", ("Yes", "No"), index=0)
    with col6:
        basement = st.selectbox("Has Basement?", ("No", "Yes"), index=0)

    col7, col8 = st.columns(2)
    with col7:
        city = st.selectbox("City", options=CITIES_IN_MODEL)
    with col8:
        street = st.text_input("Street Address", placeholder="e.g., 921 Isabella Way", help="Street address is not used in this model.")

    predict_button = st.button("Calculate Estimated Value")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Prediction Logic ---
    if predict_button:
        input_data = {
            'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
            'mainroad': 1 if mainroad == 'Yes' else 0,
            'basement': 1 if basement == 'Yes' else 0,
            'parking': parking, 'city': city,
        }
        input_df = pd.DataFrame([input_data])
        input_df_encoded = pd.get_dummies(input_df, columns=['city'])
        
        # --- THIS LINE NOW USES THE HARD-CODED LIST ---
        final_df = input_df_encoded.reindex(columns=MODEL_TRAINING_COLUMNS, fill_value=0)

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
            st.success("This prediction is based on historical market data.")
        
        st.balloons()
