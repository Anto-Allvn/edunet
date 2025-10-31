import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import requests

# Google Drive model link
MODEL_URL = "https://drive.google.com/uc?export=download&id=1hggH670mypDdqUxSiq6nYkC8I7iqjsqi"
MODEL_PATH = "random_forest_model.pkl"

st.set_page_config(page_title="Energy Consumption Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Appliance Energy Consumption Predictor")

# === Download model if not present ===
if not os.path.exists(MODEL_PATH):
    st.info("Downloading trained model... Please wait ⏳")
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded successfully ✅")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")

# === Load model ===
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# === Load data to get unique values ===
try:
    df = pd.read_csv("data.csv")
    df.dropna(subset=['Energy Consumption (kWh)', 'Time', 'Date', 'Outdoor Temperature (°C)', 'Season', 'Household Size'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df['Hour'] = df['DateTime'].dt.hour
    df['Day of Week'] = df['DateTime'].dt.dayofweek
    df['Month'] = df['DateTime'].dt.month
    df = pd.get_dummies(df, columns=['Appliance Type', 'Season'], drop_first=True)

    desired_base_features = ['Hour', 'Day of Week', 'Month', 'Outdoor Temperature (°C)', 'Household Size']
    appliance_season_features = [col for col in df.columns if 'Appliance Type_' in col or 'Season_' in col]
    all_features = desired_base_features + appliance_season_features
    X = df[all_features]

    df_original = pd.read_csv('data.csv')
    appliance_types = df_original['Appliance Type'].unique().tolist()
    seasons = df_original['Season'].unique().tolist()

except Exception as e:
    st.warning(f"⚠️ Could not load data.csv ({e}). Using default options.")
    appliance_types = ['Fridge', 'Oven', 'Dishwasher', 'Heater', 'Microwave', 'Television', 'Washing Machine', 'Computer', 'Lights']
    seasons = ['Fall', 'Summer', 'Winter', 'Spring']
    all_features = ['Hour', 'Day of Week', 'Month', 'Outdoor Temperature (°C)', 'Household Size']

# === Prediction function ===
def predict_energy_consumption(model, hour, day_of_week, month, outdoor_temperature, household_size, appliance_type, season, all_features):
    user_input = {
        'Hour': hour,
        'Day of Week': day_of_week,
        'Month': month,
        'Outdoor Temperature (°C)': outdoor_temperature,
        'Household Size': household_size,
        'Appliance Type': appliance_type,
        'Season': season
    }
    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df, columns=['Appliance Type', 'Season'])
    for feature in all_features:
        if feature not in user_df.columns:
            user_df[feature] = 0
    user_df = user_df[all_features]
    prediction = model.predict(user_df)
    return prediction[0]

# === Streamlit UI ===
st.header("🔧 Enter the parameters below")

hour = st.slider("Hour of the Day", 0, 23, 10)
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_of_week_str = st.selectbox("Day of the Week", day_names)
day_of_week = day_names.index(day_of_week_str)
month = st.slider("Month", 1, 12, 7)
outdoor_temperature = st.number_input("Outdoor Temperature (°C)", value=25.0)
household_size = st.number_input("Household Size", min_value=1, value=3)
appliance_type = st.selectbox("Appliance Type", appliance_types)
season = st.selectbox("Season", seasons)

if st.button("🔮 Predict Energy Consumption"):
    try:
        predicted = predict_energy_consumption(model, hour, day_of_week, month, outdoor_temperature, household_size, appliance_type, season, all_features)
        st.success(f"⚡ Predicted Energy Consumption: {predicted:.2f} kWh")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
