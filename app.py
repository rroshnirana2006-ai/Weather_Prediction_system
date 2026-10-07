import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Weather Prediction",
    page_icon="🌤️",
    layout="centered"
)

# ==========================================
# LOAD TRAINEdef get_weather(city):D ML MODEL
# ==========================================

model = joblib.load("weather_model.pkl")

# ==========================================
# WEATHER API KEY
# ==========================================

API_KEY = st.secrets["WEATHER_API_KEY"]

# ==========================================
# PAGE TITLE
# ==========================================

st.title("🌤️ Weather Prediction System")
st.write("Get live weather data and predict temperature using Machine Learning.")

# ==========================================
# CITY INPUT
# ==========================================

city = st.text_input(
    "Enter City Name",
    placeholder="Example: Ludhiana"
)

# ==========================================
# GET WEATHER FUNCTION
# ==========================================
def get_weather(city):

    url = "https://api.weatherapi.com/v1/current.json"

    params = {
        "key": API_KEY,
        "q": city
    }

    response = requests.get(url, params=params)

    print("Status Code:", response.status_code)
    print("Response:", response.text)

    if response.status_code != 200:
        return None

    return response.json()
# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("🔍 Get Weather & Predict"):

    if city == "":
        st.warning("Please enter a city name.")

    else:

        with st.spinner("Getting live weather data..."):

            weather = get_weather(city)

        if weather is None:

            st.error("Unable to get weather data. Please check the city name.")

        else:

            # ==========================================
            # GET WEATHER INFORMATION
            # ==========================================

            current = weather["current"]
            location = weather["location"]

            apparent_temp = current["feelslike_c"]

            humidity = current["humidity"] / 100

            wind_speed = current["wind_kph"]

            wind_bearing = current["wind_degree"]

            visibility = current["vis_km"]

            pressure = current["pressure_mb"]

            year = int(location["localtime"][0:4])

            month = int(location["localtime"][5:7])

            day = int(location["localtime"][8:10])

            hour = int(location["localtime"][11:13])

            # ==========================================
            # CYCLICAL FEATURES
            # ==========================================

            hour_sin = np.sin(
                2 * np.pi * hour / 24
            )

            hour_cos = np.cos(
                2 * np.pi * hour / 24
            )

            month_sin = np.sin(
                2 * np.pi * month / 12
            )

            month_cos = np.cos(
                2 * np.pi * month / 12
            )

            # ==========================================
            # CREATE DATAFRAME
            # ==========================================

            new_data = pd.DataFrame({

                "Apparent Temperature (C)": [apparent_temp],

                "Humidity": [humidity],

                "Wind Speed (km/h)": [wind_speed],

                "Wind Bearing (degrees)": [wind_bearing],

                "Visibility (km)": [visibility],

                "Pressure (millibars)": [pressure],

                "Year": [year],

                "Month": [month],

                "Day": [day],

                "Hour": [hour],

                "Hour_sin": [hour_sin],

                "Hour_cos": [hour_cos],

                "Month_sin": [month_sin],

                "Month_cos": [month_cos]
            })

            # ==========================================
            # ML PREDICTION
            # ==========================================

            prediction = model.predict(new_data)

            predicted_temperature = prediction[0]

            # ==========================================
            # DISPLAY LOCATION
            # ==========================================

            st.success(
                f"Weather data found for {location['name']}, "
                f"{location['country']}"
            )

            # ==========================================
            # LIVE WEATHER
            # ==========================================

            st.subheader("🌍 Live Weather")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Actual Temperature",
                    f"{current['temp_c']} °C"
                )

                st.metric(
                    "Humidity",
                    f"{current['humidity']}%"
                )

                st.metric(
                    "Wind Speed",
                    f"{current['wind_kph']} km/h"
                )

            with col2:

                st.metric(
                    "Feels Like",
                    f"{current['feelslike_c']} °C"
                )

                st.metric(
                    "Pressure",
                    f"{current['pressure_mb']} mb"
                )

                st.metric(
                    "Visibility",
                    f"{current['vis_km']} km"
                )

            st.write(
                "🌦️ **Weather Condition:**",
                current["condition"]["text"]
            )

            # ==========================================
            # ML PREDICTION
            # ==========================================

            st.divider()

            st.subheader("🤖 Machine Learning Prediction")

            st.metric(
                "Predicted Temperature",
                f"{predicted_temperature:.2f} °C"
            )

            # ==========================================
            # COMPARISON
            # ==========================================

            difference = (
                predicted_temperature - current["temp_c"]
            )

            st.write(
                f"**Difference between actual and predicted:** "
                f"{difference:.2f} °C"
            )
            