# Lab5.py
import streamlit as st
import requests
import openai

# -----------------------------
# Load API keys from secrets
# -----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]

openai.api_key = OPENAI_API_KEY  # Set OpenAI key

# -----------------------------
# Weather function (Lab 5a)
# -----------------------------
def get_current_weather(location, API_key):
    if "," in location:
        location = location.split(",")[0].strip()
    
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_key}"
    response = requests.get(url)
    
    # Handle errors
    if response.status_code != 200:
        st.error(f"Error retrieving weather for {location}.")
        return None

    data = response.json()
    temp = data['main']['temp'] - 273.15
    feels_like = data['main']['feels_like'] - 273.15
    temp_min = data['main']['temp_min'] - 273.15
    temp_max = data['main']['temp_max'] - 273.15
    humidity = data['main']['humidity']

    return {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "humidity": round(humidity, 2)
    }

# -----------------------------
# OpenAI suggestion function (Lab 5b)
# -----------------------------
def get_clothing_suggestion(weather_info):
    prompt = f"""
    Given the following weather information:
    Temperature: {weather_info['temperature']}°C
    Feels like: {weather_info['feels_like']}°C
    Min: {weather_info['temp_min']}°C, Max: {weather_info['temp_max']}°C
    Humidity: {weather_info['humidity']}%
    
    Suggest appropriate clothing and tell if it's a good day for a picnic.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

# -----------------------------
# Streamlit app function
# -----------------------------
def run():
    st.title("🌤️ What to Wear Bot")

    # User input
    city = st.text_input("Enter a city (default: Syracuse, NY):", "Syracuse, NY")

    # Get weather info
    weather_info = get_current_weather(city, OPENWEATHER_API_KEY)

    if weather_info:
        st.subheader(f"Weather in {weather_info['location']}")
        st.write(f"Temperature: {weather_info['temperature']}°C")
        st.write(f"Feels like: {weather_info['feels_like']}°C")
        st.write(f"Humidity: {weather_info['humidity']}%")

        # Get clothing suggestion
        suggestion = get_clothing_suggestion(weather_info)
        st.subheader("🧥 Clothing & Picnic Suggestion")
        st.write(suggestion)
