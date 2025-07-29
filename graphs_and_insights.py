import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Winky+Rough&display=swap');
    .title {
        font-family: Pacifico, cursive;
        font-size: 3.2rem;
        color: #77cef0;
        text-align: center;
        margin-bottom: 20px;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }
    .stSelectbox > div > div {
            background-color: #ffffff !important;
            color: #000000 !important;
    }
    .stButton>button {
            display: block;
            padding: 0.5rem 1rem;
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
    .sub {
            font-family: 'Winky Rough', cursive;!important;
            font-size: 1.3rem;
            text-align: center;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #77cef0 !important;
        color: white !important;
        border-radius: 10px;
        border: none;
        font-weight: 500;
    }

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1f4e79;
}


</style>
""", unsafe_allow_html=True)

df = pd.read_csv("weather_classification_data.csv")

# Set up the color palette
colors = {'Rainy': "#41A9FF", 'Cloudy': "#F8A1DE",'Sunny': "#FFE96C", 'Snowy': "#87E9EB"}
weather_order = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
color_palette = [colors[weather] for weather in weather_order]

cloud_map = {
    'clear': 0,
    'partly cloudy': 1,
    'cloudy': 2,
    'overcast': 3
}

df['Cloud Cover Numeric'] = df['Cloud Cover'].map(cloud_map)
# Sidebar for weather selection
with st.sidebar:
    
    st.markdown("🎯 Choose Your Weather")
    weather_choice = st.selectbox(
        "Select weather type",
        ["Sunny", "Rainy", "Cloudy", "Snowy"],
        help="Pick a weather type to explore specific insights!"
    )

    sunny_df = df[df['Weather Type'] == 'Sunny']
    rainy_df = df[df['Weather Type'] == 'Rainy']
    cloudy_df = df[df['Weather Type'] == 'Cloudy']
    snowy_df = df[df['Weather Type'] == 'Snowy']


    st.markdown("📊 Quick Stats")
    # Define weather_data mapping
    weather_data = {
        "Sunny": sunny_df,
        "Rainy": rainy_df,
        "Cloudy": cloudy_df,
        "Snowy": snowy_df
    }
    selected_data = weather_data[weather_choice]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📏 Avg UV", f"{selected_data['UV Index'].mean():.1f}")
        st.metric("🌡️ Avg Temp", f"{selected_data['Temperature'].mean():.1f}°C")
    with col2:
        st.metric("💧 Avg Humidity", f"{selected_data['Humidity'].mean():.1f}%")
        wind_speed_text = f"{selected_data['Wind Speed'].mean():.1f} km/h"
        st.metric("💨 Avg Wind", wind_speed_text)
st.markdown("<br>", unsafe_allow_html=True)  
st.markdown('<div class="title">Graphs & Insights 📊</div>', unsafe_allow_html=True)
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


st.markdown('<style>.stApp {background-image: url("data:clouds.jpg;base64,' + base64.b64encode(open("clouds.jpg", "rb").read()).decode() + '");'
'background-position: center;background-repeat: no-repeat;background-size: cover;}</style>', unsafe_allow_html=True)

st.markdown('<div class="sub">Explore cool patterns and weather facts from the dataset used to train this model!</div><br>', unsafe_allow_html=True)

sunny_facts = [
    f"☀️ Average UV index on sunny days is {sunny_df['UV Index'].mean():.1f}",
    f"☀️ Average temperature on sunny days is {sunny_df['Temperature'].mean():.1f}°C",
    f"☀️ Average humidity on sunny days is {sunny_df['Humidity'].mean():.1f}%",
    f"☀️ Sunny days have an average wind speed of {sunny_df['Wind Speed'].mean():.1f} km/h",
    f"☀️ Average Precipitation on sunny days is {sunny_df['Precipitation (%)'].mean():.1f}%",
]

rainy_facts = [
    f"🌧️ Average precipitation on rainy days is {rainy_df['Precipitation (%)'].mean():.1f}%",
    f"🌧️ Rainy days have an average temperature of {rainy_df['Temperature'].mean():.1f}°C",
    f"🌧️ Average wind speed during rain is {rainy_df['Wind Speed'].mean():.1f} km/h",
    f"🌧️ Rainy days have an average humidity of {rainy_df['Humidity'].mean():.1f}%",
    f"🌧️ Average atmospheric pressure during rain is {rainy_df['Atmospheric Pressure'].mean():.1f} hPa",
]

cloudy_facts = [
    f"☁️ Cloudy days have an average cloud cover of {cloudy_df['Cloud Cover Numeric'].mean():.1f}%",
    f"☁️ Average pressure on cloudy days is {cloudy_df['Atmospheric Pressure'].mean():.1f} hPa",
    f"☁️ Cloudy days are usually {cloudy_df['Temperature'].mean():.1f}°C on average",
    f"☁️ Average wind speed on cloudy days is {cloudy_df['Wind Speed'].mean():.1f} km/h",
    f"☁️ Average UV index on cloudy days is {cloudy_df['UV Index'].mean():.1f}",
]

snowy_facts = [
    f"❄️ Average temperature on snowy days is {snowy_df['Temperature'].mean():.1f}°C",
    f"❄️ Average wind speed during snow is {snowy_df['Wind Speed'].mean():.1f} km/h",
    f"❄️ UV index tends to be lower during snowy days: {snowy_df['UV Index'].mean():.1f}",
    f"❄️ Average humidity on snowy days is {snowy_df['Humidity'].mean():.1f}%",
    f"❄️ Average atmospheric pressure during snow is {snowy_df['Atmospheric Pressure'].mean():.1f} hPa",
]

weather_choice = st.selectbox("Choose a weather type", ["Sunny", "Rainy", "Cloudy", "Snowy"])


if st.button("🎁 Give me a random weather fact!"):
    if weather_choice == "Sunny":
        st.success(random.choice(sunny_facts))
    elif weather_choice == "Rainy":
        st.success(random.choice(rainy_facts))
    elif weather_choice == "Cloudy":
        st.success(random.choice(cloudy_facts))
    elif weather_choice == "Snowy":
        st.success(random.choice(snowy_facts))


st.subheader("🌡️ Average Temperature by Weather Type")
st.caption("See how hot or cold it is on Sunny, Rainy, Cloudy, or Snowy days.")
fig2, ax2 = plt.subplots(facecolor='white', figsize=(10, 6))
sns.barplot(data=df, x='Weather Type', y='Temperature', palette=color_palette,order=weather_order, ax=ax2)
ax2.set_title("Average Temperature per Weather Type")
st.pyplot(fig2)

st.subheader("💧 Average Humidity by Weather Type")
st.caption("How humid is it on different weather days? Let's find out!")
fig2, ax2 = plt.subplots(facecolor='white', figsize=(10, 6))
sns.barplot(data=df, x='Weather Type', y='Humidity', palette=color_palette,order=weather_order, ax=ax2)
ax2.set_title("Average Humidity per Weather Type")
st.pyplot(fig2)

st.subheader("☀️ Average UV Index by Weather Type")
st.caption("How does the weather affect UV levels? Let's find out!")
fig2, ax2 = plt.subplots(facecolor='white', figsize=(10, 6))
sns.barplot(data=df, x='Weather Type', y='UV Index', palette=color_palette,order=weather_order, ax=ax2)
ax2.set_title("Average UV Index per Weather Type")
st.pyplot(fig2)

st.subheader("⛅ Cloud Cover vs UV Index")
st.caption("How does cloud cover affect UV levels? Let's find out!")
avg_uv_by_cloud = df.groupby('Cloud Cover Numeric')['UV Index'].mean()
fig, ax = plt.subplots()
avg_uv_by_cloud.plot(kind='line', ax=ax, marker='o')
ax.set_xlabel("Cloud Cover Level (0=Clear to 3=Overcast)")
ax.set_ylabel("Average UV Index")
st.pyplot(fig)

st.subheader("🍃 Wind Speed by Weather Type")
st.caption("Ever noticed it's windier on some days? Here's the average wind speed for each weather type.")
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x='Weather Type', y='Wind Speed', palette=color_palette,order=weather_order, ax=ax3)
ax3.set_title("Average Wind Speed")
ax3.set_ylabel("Wind Speed (km/h)")
st.pyplot(fig3)

st.subheader("🌧️ Precipitation Levels by Weather Type")
st.caption("Not all rain is the same! Let’s see how much water each weather type brings.")
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='Weather Type', y='Precipitation (%)', palette=color_palette, order=weather_order, ax=ax4)
ax4.set_title("Rain & Snow Amounts")
ax4.set_ylabel("Precipitation (%)")
st.pyplot(fig4)










