import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("advanced_temperature_model.pkl")

# Page config
st.set_page_config(page_title="Smart Cold Chain Temperature Recommender", layout="centered")
st.title("🧊 Smart Cold Chain Temp Recommender")
st.markdown("AI-powered temperature optimization system for dairy product storage.")

# Input fields
product = st.selectbox("Select Product Type", ['milk', 'curd', 'butter', 'cheese', 'ice-cream', 'flavored_beverage'])
external_temp = st.slider("External Temperature (°C)", 20, 45, 30)
current_temp = st.slider("Current Room Temperature (°C)", -20, 25, 5)
humidity = st.slider("Humidity (%)", 30, 90, 60)
volume_kg = st.slider("Total Product Weight (kg)", 1, 100, 10)
packaging = st.selectbox("Packaging Type", ['plastic', 'glass', 'tetrapack', 'metal'])
storage_time = st.slider("Storage Duration (in hours)", 1, 72, 12)
airflow = st.selectbox("Airflow Rating", ['poor', 'moderate', 'good'])

# Predict button
if st.button("📡 Predict Ideal Room Temperature"):
    # Prepare input data
    input_df = pd.DataFrame([{
        'product_type': product,
        'external_temp': external_temp,
        'current_room_temp': current_temp,
        'humidity': humidity,
        'volume_kg': volume_kg,
        'packaging_type': packaging,
        'storage_time_hr': storage_time,
        'airflow_rating': airflow
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Recommended Ideal Room Temperature: **{round(prediction, 2)} °C**")
