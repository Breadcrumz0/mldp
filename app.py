import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

model = joblib.load("weather_model.pkl")
le = joblib.load('label_encoder.pkl')

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Winky+Rough&display=swap');
        .title {
            font-family: 'Pacifico', cursive;
            font-size: 3.2rem;
            color: #ffffff !important;
            text-align: center;
            margin-bottom: 20px;
        }

        .stApp {
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .stButton>button {
            display: block;
            width: 100%;
            padding: 14px 0;
            margin-top: 20px;
            margin-left: 0;
            grid-column: 1 / -1;
            font-size: 1rem;
            font-weight: 150;
            font-family: 'Winky Rough', cursive; !important;
            border: none;
            border-radius: 8px; 
            background-color: #77cef0;
            color: white;
            cursor: pointer;
            box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #65b5d5;
        }

        .result {
            font-size: 2rem;
            font-family: 'Winky Rough', cursive;
            color: #333;
            text-align: center;
            margin-top: 20px;
        }
            
        label,        
        .stNumberInput label,
        .stSlider label,
        .stSelectbox label {
            font-family: 'Winky Rough', cursive !important;
            font-size: 1.1rem;
            color: #333333;
        }
            
        input, select, textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        .stNumberInput input {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 0 !important;
            border: none !important;
        }
            
        .stNumberInput button {
            background-color: #ffffff !important;
            color: #000000 !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            border: none !important;
        }
            
        .stNumberInput button:hover {
            background-color: #77cef0 !important; /* Same as your main hover color */
            color: white !important;
            border: 1px solid #77cef0 !important;
        }

        .stSelectbox > div > div {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button {
            background-color: #ffffff !important;
            border: none;
            color: black;
        }
            
    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">Weather Prediction</div>', unsafe_allow_html=True)
st.markdown('<style>.stApp {background-image: url("data:clouds.jpg;base64,' + base64.b64encode(open("clouds.jpg", "rb").read()).decode() + '");}</style>', unsafe_allow_html=True)
st.sidebar.markdown("""
<style>
    .css-1d391kg { background-color: #f0f4f8; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stSidebar"] .stSelectbox > div{
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


cloud_cover_options = {
    "Clear": "clear",
    "Cloudy": "cloudy",
    "Overcast": "overcast",
    "Partly Cloudy": "partly cloudy"
}

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    wind_speed = st.number_input("Wind Speed (km/h)")
    precipitation = st.number_input("Precipitation (%)")
    season = st.selectbox("Season", ["Autumn", "Spring", "Summer", "Winter"])

with col2:
    pressure = st.number_input("Atmospheric Pressure (hPa)")
    uv_index = st.number_input("UV Index")
    visibility = st.number_input("Visibility (km)")

    cloud_cover_pretty = st.selectbox("Cloud Cover", list(cloud_cover_options.keys()))
    cloud_cover = cloud_cover_options[cloud_cover_pretty]

input_dict = {
    "Temperature": temperature,
    "Humidity": humidity,
    "Wind Speed": wind_speed,
    "Precipitation (%)": precipitation,
    "Atmospheric Pressure": pressure,
    "UV Index": uv_index,
    "Visibility (km)": visibility,
    "Cloud Cover": cloud_cover,
    "Season": season
}


input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df, columns=["Cloud Cover", "Season"])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)


if st.button("Predict Weather"):

    prediction = model.predict(input_df)[0]
    decoded_prediction = le.inverse_transform([prediction])[0]
    probs = model.predict_proba(input_df)[0]
    confidence = probs[prediction] * 100

    st.markdown(f'<div class="result">Prediction: <strong>{decoded_prediction}</strong><br>'
            f'Confidence: <strong>{confidence:.2f}%</strong></div>', unsafe_allow_html=True)


