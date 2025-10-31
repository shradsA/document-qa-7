import streamlit as st
import requests
import os
import json
from typing import Dict, Optional
from openai import OpenAI

# ============ PART A: Weather Function ============
def get_current_weather(location: str, API_key: str) -> Dict:
    """Fetch current weather data for a given location using OpenWeatherMap API."""
    try:
        if "," in location:
            location = location.split(",")[0].strip()
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_key}"
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": f"City '{location}' not found"}
        data = response.json()
        temp = data['main']['temp'] - 273.15
        feels_like = data['main']['feels_like'] - 273.15
        temp_min = data['main']['temp_min'] - 273.15
        temp_max = data['main']['temp_max'] - 273.15
        humidity = data['main']['humidity']
        weather_description = data['weather'][0]['description']
        wind_speed = data['wind']['speed']
        return {
            "location": location,
            "temperature": round(temp, 2),
            "feels_like": round(feels_like, 2),
            "temp_min": round(temp_min, 2),
            "temp_max": round(temp_max, 2),
            "humidity": round(humidity, 2),
            "description": weather_description,
            "wind_speed": round(wind_speed, 2)
        }
    except Exception as e:
        return {"error": f"Error fetching weather: {str(e)}"}

# ============ PART B: OpenAI Function Calling ============
def get_weather_for_openai(location: str = "Syracuse NY") -> str:
    api_key = st.session_state.get('weather_api_key')
    if not api_key:
        return json.dumps({"error": "Weather API key not configured"})
    if not location or location.strip() == "":
        location = "Syracuse NY"
    weather_data = get_current_weather(location, api_key)
    if "error" not in weather_data:
        formatted = {
            "location": weather_data["location"],
            "temperature_celsius": weather_data["temperature"],
            "feels_like_celsius": weather_data["feels_like"],
            "weather_condition": weather_data["description"],
            "humidity_percent": weather_data["humidity"],
            "wind_speed_ms": weather_data["wind_speed"]
        }
        return json.dumps(formatted)
    else:
        return json.dumps(weather_data)

WEATHER_FUNCTION_DEFINITION = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state/country, e.g. San Francisco, CA or Paris, France"
            }
        },
        "required": ["location"]
    }
}

def get_openai_response_with_function(client: OpenAI, user_input: str) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a helpful travel assistant that provides weather information and clothing suggestions."},
            {"role": "user", "content": user_input}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=[WEATHER_FUNCTION_DEFINITION],
            function_call="auto"
        )
        message = response.choices[0].message
        if message.function_call:
            function_args = json.loads(message.function_call.arguments)
            location = function_args.get("location", "Syracuse NY")
            weather_result = get_weather_for_openai(location)
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": message.function_call.model_dump()
            })
            messages.append({
                "role": "function",
                "name": "get_current_weather",
                "content": weather_result
            })
            messages.append({
                "role": "user",
                "content": "Based on this weather data, please provide detailed suggestions for what clothes to wear today. Include upper body, lower body, footwear, and accessories."
            })
            final_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return final_response.choices[0].message.content
        else:
            return message.content
    except Exception as e:
        return f"Error with OpenAI: {str(e)}"

# ============ MAIN APPLICATION ============
def run():
    st.set_page_config(page_title="Lab 5 - Travel Weather Bot", page_icon="🌤️", layout="wide")
    st.title("🌤️ Travel Weather & Clothing Suggestion Bot")
    st.markdown("Get weather information and personalized clothing recommendations using AI")

    # Load API keys
    weather_api_key = st.secrets.get("OPENWEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")
    if weather_api_key:
        st.session_state['weather_api_key'] = weather_api_key
    else:
        st.error("❌ OpenWeatherMap API key not found")
        return
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    # Sidebar for model selection (only OpenAI now)
    st.sidebar.header("🤖 AI Model Selection")
    selected_model = st.sidebar.selectbox("Choose AI Model:", ["OpenAI (GPT-3.5)", "OpenAI (GPT-4)"])

    # User input
    user_input = st.text_input(
        "Ask about weather or what to wear:",
        placeholder="e.g., 'What should I wear in Paris today?'"
    )

    if st.button("🔍 Get Suggestions", type="primary", disabled=not user_input):
        if user_input:
            with st.spinner("Getting weather and suggestions..."):
                client = OpenAI(api_key=openai_api_key)
                response = get_openai_response_with_function(client, user_input)
                st.subheader("🤖 AI Suggestions")
                st.markdown(response)
                st.caption(f"Generated by: {selected_model}")

if __name__ == "__main__":
    run()
