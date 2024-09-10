import requests
import concurrent.futures
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import keras
from dotenv import load_dotenv
import os
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()
# Retrieve the API key from the environment variables
api_key = os.getenv('api_key')
if api_key is None:
    raise ValueError("API key is not set in environment variables.")
# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})
# Load the trained model and scaler
with open('electricity_model/gru_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('electricity_model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
us_holidays_2024 = {
    "2024-01-01": 1,  # New Year's Day
    "2024-01-15": 1,  # Martin Luther King Jr. Day
    "2024-02-19": 1,  # Presidents' Day
    "2024-05-27": 1,  # Memorial Day
    "2024-06-19": 1,  # Juneteenth National Independence Day
    "2024-07-04": 1,  # Independence Day
    "2024-09-02": 1,  # Labor Day
    "2024-10-14": 1,  # Columbus Day
    "2024-11-11": 1,  # Veterans Day
    "2024-11-28": 1,  # Thanksgiving Day
    "2024-12-25": 1   # Christmas Day
}

# Function to fetch weather data from the API
def fetch_weather_data(location, start_date, end_date, api_key):
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{location}/{start_date}/{end_date}?"
        f"unitGroup=metric&key={api_key}&contentType=json"
    )
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": f"Failed to fetch data for {location}, status code: {response.status_code}"}
    return response.json()

# Function to extract relevant weather information
def extract_weather_info(data):
    weather_info = {
        "T2M_toc": {},
        "QV2M_toc": {},
        "TQL_toc": {},
        "W2M_toc": {},
        "T2M_san": {},
        "QV2M_san": {},
        "TQL_san": {},
        "W2M_san": {},
        "T2M_dav": {},
        "QV2M_dav": {},
        "TQL_dav": {},
        "W2M_dav": {}
    }

    address_mapping = {
        "Tocumen, Panama": "toc",
        "Santiago, Panama": "san",
        "David, Panama": "dav"
    }

    address = data.get("address")
    location_key = address_mapping.get(address)

    if location_key:
        for day in data.get("days", []):
            for hour in day.get("hours", []):
                hour_time = hour.get("datetime")
                hour_str = hour_time
                weather_info[f"T2M_{location_key}"][hour_str] = hour.get("temp", np.nan)
                weather_info[f"QV2M_{location_key}"][hour_str] = hour.get("humidity", np.nan)
                weather_info[f"TQL_{location_key}"][hour_str] = hour.get("precip", np.nan)
                weather_info[f"W2M_{location_key}"][hour_str] = hour.get("windspeed", np.nan)

    return weather_info

# Function to fetch and extract weather data for all locations
def fetch_and_extract_weather_for_all_locations(locations, start_date, end_date, api_key):
    total_weather_info = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_location = {
            executor.submit(fetch_weather_data, location, start_date, end_date, api_key): location
            for location in locations
        }

        for future in concurrent.futures.as_completed(future_to_location):
            location = future_to_location[future]
            try:
                data = future.result()
                weather_info = extract_weather_info(data)
                total_weather_info[location] = weather_info
            except Exception as e:
                print(f"An error occurred for {location}: {e}")

    return total_weather_info

# Function to flatten weather information from multiple locations
def flatten_weather_info(total_weather_info):
    flattened_weather_info = {}

    for location, weather_data in total_weather_info.items():
        for key, value in weather_data.items():
            if key in flattened_weather_info:
                flattened_weather_info[key].update(value)
            else:
                flattened_weather_info[key] = value

    return flattened_weather_info

@app.route('/')
def hello():
    return render_template('index.html')

from flask import jsonify

@app.route('/predict', methods=['POST'])
def predict_and_plot_day():
    date_str = request.form.get('date') or request.json.get('date')
    # return "searching for locations"
    # Fetch weather data
    locations = ["Santiago, Panama", "David, Panama", "Tocumen, Panama"]
    start_date = date_str
    end_date = date_str
    weather_data = fetch_and_extract_weather_for_all_locations(locations, start_date, end_date, api_key)
    # return jsonify({"message":"fetched weather data"})
    # Flatten the weather data
    flattened_weather_data = flatten_weather_info(weather_data)

    # Define the date and time range for the whole day (00:00 to 23:00)
    start_time = datetime.strptime(date_str, '%Y-%m-%d')
    end_time = start_time + timedelta(days=1)
    time_range = pd.date_range(start=start_time, end=end_time - timedelta(hours=1), freq='H')
    # return jsonify({"message":"combining information"})
    data = []
    for timestamp in time_range:
        hour_str = timestamp.strftime('%H:%M:%S')
        kmh_to_ms = 0.27778
        print("")
        # Add missing 'month', 'hour', 'day', 'hour_sin', 'hour_cos'
        row = {
            'month': timestamp.month,
            'hour': timestamp.hour,
            'day': timestamp.day,
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'T2M_toc': flattened_weather_data.get('T2M_toc', {}).get(hour_str, np.nan),
            'QV2M_toc': flattened_weather_data.get('QV2M_toc', {}).get(hour_str, np.nan),
            'TQL_toc': flattened_weather_data.get('TQL_toc', {}).get(hour_str, np.nan),
            'W2M_toc': flattened_weather_data.get('W2M_toc', {}).get(hour_str, np.nan) * kmh_to_ms if flattened_weather_data.get('W2M_toc', {}).get(hour_str, np.nan) is not np.nan else np.nan,
            'T2M_san': flattened_weather_data.get('T2M_san', {}).get(hour_str, np.nan),
            'QV2M_san': flattened_weather_data.get('QV2M_san', {}).get(hour_str, np.nan),
            'TQL_san': flattened_weather_data.get('TQL_san', {}).get(hour_str, np.nan),
            'W2M_san': flattened_weather_data.get('W2M_san', {}).get(hour_str, np.nan) * kmh_to_ms if flattened_weather_data.get('W2M_san', {}).get(hour_str, np.nan) is not np.nan else np.nan,
            'T2M_dav': flattened_weather_data.get('T2M_dav', {}).get(hour_str, np.nan),
            'QV2M_dav': flattened_weather_data.get('QV2M_dav', {}).get(hour_str, np.nan),
            'TQL_dav': flattened_weather_data.get('TQL_dav', {}).get(hour_str, np.nan),
            'W2M_dav': flattened_weather_data.get('W2M_dav', {}).get(hour_str, np.nan) * kmh_to_ms if flattened_weather_data.get('W2M_dav', {}).get(hour_str, np.nan) is not np.nan else np.nan,
            'holiday': us_holidays_2024.get(date_str, 0)
        }
        data.append(row)
   
    # return jsonify({"message":data})
    df = pd.DataFrame(data)
    print("Ensure all feature columns are present")
    # Ensure all feature columns are present
    feature_columns = ['month', 'hour', 'day', 'hour_sin', 'hour_cos', 'QV2M_toc', 'TQL_toc', 'W2M_toc', 
                       'QV2M_san', 'TQL_san', 'W2M_san', 'QV2M_dav', 'TQL_dav', 'W2M_dav', 
                       'T2M_toc', 'T2M_san', 'T2M_dav', 'holiday']
    df = df[feature_columns]
    # return jsonify({"message":"feature columns are added"})
    # Handle missing values if any
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Scale the features
    df_scaled = scaler.transform(df)
    # return jsonify({"df":df})
    # Reshape the data to fit the model's input shape
    df_reshaped = df_scaled.reshape(df_scaled.shape[0], 1, df_scaled.shape[1])
    # return jsonify({"message":"predicting for the given data"})
    print("predicting for the given data")
    # Predict using the loaded model
    predictions = loaded_model.predict(df_reshaped).flatten()

    # Convert predictions to a list of Python floats for JSON serialization
    predictions_list = predictions.tolist()
    print("sending response")
    # Return the predictions as a JSON object
    return jsonify({'predictions_list': predictions_list})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
