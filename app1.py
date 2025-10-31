import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# === Streamlit Page Setup ===
st.set_page_config(page_title="Energy Consumption Predictor", page_icon="⚡", layout="centered")
st.title("⚡ Appliance Energy Consumption Predictor (Demo Model)")
st.write("This demo runs without any uploaded model file — it uses a lightweight Random Forest trained inside the app.")

# === Generate Sample Training Data ===
@st.cache_data
def create_demo_model():
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'Hour': np.random.randint(0, 24, n),
        'Day of Week': np.random.randint(0, 7, n),
        'Month': np.random.randint(1, 13, n),
        'Outdoor Temperature (°C)': np.random.uniform(10, 40, n),
        'Household Size': np.random.randint(1, 6, n),
        'Appliance Type': np.random.choice(['Fridge', 'Oven', 'Heater', 'Television', 'Computer'], n),
        'Season': np.random.choice(['Summer', 'Winter', 'Spring', 'Fall'], n),
    })
    # Simple rule-based simulated target
    data['Energy Consumption (kWh)'] = (
        data['Household Size'] * np.random.uniform(0.2, 0.5, n)
        + (40 - data['Outdoor Temperature (°C)']) * 0.05
        + data['Hour'] * 0.1
        + np.random.uniform(0, 1, n)
    )

    # One-hot encode categorical columns
    df = pd.get_dummies(data, columns=['Appliance Type', 'Season'], drop_first=True)
    X = df.drop(columns=['Energy Consumption (kWh)'])
    y = df['Energy Consumption (kWh)']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist(), data['Appliance Type'].unique().tolist(), data['Season'].unique().tolist()

model, all_features, appliance_types, seasons = create_demo_model()

# === Prediction Function ===
def predict_energy(model, hour, day_of_week, month, temp, size, appliance, season, all_features):
    user_input = {
        'Hour': hour,
        'Day of Week': day_of_week,
        'Month': month,
        'Outdoor Temperature (°C)': temp,
        'Household Size': size,
        'Appliance Type': appliance,
        'Season': season
    }

    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df, columns=['Appliance Type', 'Season'], drop_first=True)

    for feature in all_features:
        if feature not in user_df.columns:
            user_df[feature] = 0

    user_df = user_df[all_features]
    prediction = model.predict(user_df)
    return prediction[0]

# === Streamlit UI ===
st.header("🔧 Enter your parameters below")

hour = st.slider("Hour of the Day", 0, 23, 10)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_of_week_str = st.selectbox("Day of the Week", days)
day_of_week = days.index(day_of_week_str)
month = st.slider("Month", 1, 12, 7)
temp = st.number_input("Outdoor Temperature (°C)", value=25.0)
size = st.number_input("Household Size", min_value=1, value=3)
appliance = st.selectbox("Appliance Type", appliance_types)
season = st.selectbox("Season", seasons)

if st.button("🔮 Predict Energy Consumption"):
    predicted = predict_energy(model, hour, day_of_week, month, temp, size, appliance, season, all_features)
    st.success(f"⚡ Predicted Energy Consumption: {predicted:.2f} kWh")

st.caption("🧠 This demo uses a locally trained Random Forest model with synthetic data.")
