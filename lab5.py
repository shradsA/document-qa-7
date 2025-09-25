# Lab5.py
import streamlit as st
import requests
import openai

# -----------------------------
# Load API keys from secrets
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", None)

# Check if keys exist
if not OPENAI_API_KEY:
    st.warning("⚠️ OpenAI API key is missing. Add it in secrets.toml or Streamlit Cloud secrets.")
if not OPENWEATHER_API_KEY:
    st.warning("⚠️ OpenWeatherMap API key is missing. Add it in secrets.toml or Streamlit Cloud secrets.")

openai.api_key = OPENAI_API_KEY

# -----------------------------
# Weather function
# -----------------------------
def get_current_weather(location, API_key):
    if not API_key:
        st.error("OpenWeatherMap API key is missing.")
        return None

    if "," in location:
        location = location.split(",")[0].strip()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_key}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
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

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None
    except KeyError:
        st.error("Unexpected response from OpenWeatherMap API.")
        return None

# -----------------------------
# OpenAI suggestion function
# -----------------------------
def get_clothing_suggestion(weather_info):
    if not OPENAI_API_KEY:
        return "OpenAI API key is missing. Cannot generate suggestions."

    prompt = f"""
    Given the following weather information:
    Temperature: {weather_info['temperature']}°C
    Feels like: {weather_info['feels_like']}°C
    Min: {weather_info['temp_min']}°C, Max: {weather_info['temp_max']}°C
    Humidity: {weather_info['humidity']}%
    
    Suggest appropriate clothing and tell if it's a good day for a picnic.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating clothing suggestion: {e}"

# -----------------------------
# Streamlit app function
# -----------------------------
def run():
    st.title("🌤️ What to Wear Bot")

    city = st.text_input("Enter a city (default: Syracuse, NY):", "Syracuse, NY")

    if not city:
        st.info("Please enter a city to get weather and clothing suggestions.")
        return

    weather_info = get_current_weather(city, OPENWEATHER_API_KEY)

    if weather_info:
        st.subheader(f"Weather in {weather_info['location']}")
        st.write(f"Temperature: {weather_info['temperature']}°C")
        st.write(f"Feels like: {weather_info['feels_like']}°C")
        st.write(f"Humidity: {weather_info['humidity']}%")

        suggestion = get_clothing_suggestion(weather_info)
        st.subheader("🧥 Clothing & Picnic Suggestion")
        st.write(suggestion)
    else:
        st.info("Unable to retrieve weather info. Please check the city name or API key.")
