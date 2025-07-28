import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("weather_model.pkl")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

        .title {
            font-family: 'Pacifico', cursive;
            font-size: 3.2rem;
            color: #77cef0;
            text-align: center;
            margin-bottom: 20px;
        }

        .stApp {
            background-image: url('https://clouds.jpg'); 
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }

        .stButton>button {
            background-color: #77cef0;
            color: white;
            font-family: 'Arial Rounded MT Bold', sans-serif;
            font-size: 16px;
            border-radius: 8px;
            padding: 12px;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #65b5d5;
        }

        .result {
            font-size: 24px;
            font-family: 'Arial Rounded MT Bold', sans-serif;
            color: #333;
            text-align: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Weather Prediction</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    wind_speed = st.number_input("Wind Speed (km/h)")
    precipitation = st.number_input("Precipitation (%)")

with col2:
    pressure = st.number_input("Atmospheric Pressure (hPa)")
    uv_index = st.number_input("UV Index")
    visibility = st.number_input("Visibility (km)")

    cloud_cover = st.selectbox("Cloud Cover", ["Clear", "Partly Cloudy", "Cloudy", "Overcast"])
    season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

input_dict = {
    "temperature": temperature,
    "humidity": humidity,
    "wind_speed": wind_speed,
    "precipitation": precipitation,
    "pressure": pressure,
    "uv_index": uv_index,
    "visibility": visibility,
    "cloud_cover": cloud_cover,
    "season": season
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df, columns=["cloud_cover", "season"])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predict Weather"):
    prediction = model.predict(input_df)[0]
    weather_labels = {
        0: "Sunny",
        1: "Rainy",
        2: "Snowy",
        3: "Cloudy"
    }
    result = weather_labels.get(prediction, str(prediction))
    st.markdown(f'<div class="result">Prediction: <strong>{result}</strong></div>', unsafe_allow_html=True)
