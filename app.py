import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import joblib, requests, os

print("File exists:", os.path.exists("random_forest_model.pkl"))
print("File size (MB):", os.path.getsize("random_forest_model.pkl") / 1024 / 1024)


MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

# download once if not present
if not os.path.exists("random_forest_model.pkl"):
    st.write("Downloading model...")
    with open("random_forest_model.pkl", "wb") as f:
        f.write(requests.get(MODEL_URL).content)

model = joblib.load("random_forest_model.pkl")
st.write("Model loaded successfully!")



# Define the prediction function
def predict_energy_consumption(model, hour, day_of_week, month, outdoor_temperature, household_size, appliance_type, season, all_features):
    """
    Predicts energy consumption based on user inputs.

    Args:
        model: Trained machine learning model.
        hour: Hour of the day (0-23).
        day_of_week: Day of the week (0=Monday, 6=Sunday).
        month: Month of the year (1-12).
        outdoor_temperature: Outdoor temperature in Celsius.
        household_size: Number of people in the household.
        appliance_type: Type of appliance.
        season: Season of the year.
        all_features: A list of all feature names used during training.

    Returns:
        Predicted energy consumption (kWh).
    """
    # Create a dictionary with user input values
    user_input = {
        'Hour': hour,
        'Day of Week': day_of_week,
        'Month': month,
        'Outdoor Temperature (°C)': outdoor_temperature,
        'Household Size': household_size,
        'Appliance Type': appliance_type,
        'Season': season
    }

    # Convert the dictionary to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Apply one-hot encoding to 'Appliance Type' and 'Season'
    user_df = pd.get_dummies(user_df, columns=['Appliance Type', 'Season'])

    # Add missing columns with a value of 0 to match the training data features
    for feature in all_features:
        if feature not in user_df.columns:
            user_df[feature] = 0

    # Ensure the columns are in the same order as the training data
    user_df = user_df[all_features]

    # Make the prediction
    prediction = model.predict(user_df)

    return prediction[0]

# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Load the original data to get unique values for selectboxes and all_features
try:
    df = pd.read_csv('data.csv')
    # Preprocess the loaded data to get the same features as used in training
    df.dropna(subset=['Energy Consumption (kWh)', 'Time', 'Date', 'Outdoor Temperature (°C)', 'Season', 'Household Size'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    df['Hour'] = df['DateTime'].dt.hour
    df['Day of Week'] = df['DateTime'].dt.dayofweek
    df['Month'] = df['DateTime'].dt.month
    df = pd.get_dummies(df, columns=['Appliance Type', 'Season'], drop_first=True)

    desired_base_features = ['Hour', 'Day of Week', 'Month', 'Outdoor Temperature (°C)', 'Household Size']
    appliance_season_features = [col for col in df.columns if 'Appliance Type_' in col or 'Season_' in col]
    feature_columns = desired_base_features + appliance_season_features
    X = df[feature_columns]
    all_features = X.columns.tolist()

    # Get unique appliance types and seasons from the loaded data
    # Need to load the original data again before one-hot encoding to get original values
    df_original = pd.read_csv('data.csv')
    appliance_types = df_original['Appliance Type'].unique().tolist()
    seasons = df_original['Season'].unique().tolist()

except FileNotFoundError:
    st.error("Error: data.csv not found. Please make sure the data file is in the same directory.")
    # Fallback if data file is not found - list common types/seasons
    appliance_types = ['Fridge', 'Oven', 'Dishwasher', 'Heater', 'Microwave', 'Television', 'Washing Machine', 'Computer', 'Lights']
    seasons = ['Fall', 'Summer', 'Winter', 'Spring']
    all_features = [] # Cannot determine features without data


# Streamlit UI
st.title("Appliance Energy Consumption Predictor")

st.header("Enter the parameters to predict energy consumption:")

# Get user inputs
hour = st.slider("Hour of the Day", 0, 23, 10)

day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_of_week_str = st.selectbox("Day of the Week", day_names)
day_of_week = day_names.index(day_of_week_str) # Convert string to integer (0-6)

month = st.slider("Month of the Year", 1, 12, 7)

outdoor_temperature = st.number_input("Outdoor Temperature (°C)", value=25.0)

household_size = st.number_input("Household Size", min_value=1, value=3)

appliance_type = st.selectbox("Appliance Type", appliance_types)
season = st.selectbox("Season", seasons)


# Prediction button
if st.button("Predict Energy Consumption"):
    if all_features: # Only attempt prediction if features were successfully determined
        predicted_consumption = predict_energy_consumption(model, hour, day_of_week, month, outdoor_temperature, household_size, appliance_type, season, all_features)
        st.success(f"The predicted energy consumption is: {predicted_consumption:.2f} kWh")
    else:
        st.warning("Cannot make prediction due to missing data file.")



