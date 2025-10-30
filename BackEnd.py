# Smart Home Energy Advisor Agent
# Author: ChatGPT (GPT-5)
# Description:
# This program loads smart meter data, analyzes energy consumption,
# and uses a Machine Learning model to predict future energy usage
# and provide saving recommendations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# 1️⃣ LOAD DATA
# -----------------------------
# Replace this with your dataset path
data = pd.read_csv("E:\New folder\Python\SmartHomeEnergyManagement\smart_home_energy_consumption_large.csv")

print("\n✅ Data Loaded Successfully!")
print("First few rows:\n", data.head())
print("\nDataset Info:")
print(data.info())

# -----------------------------
# 2️⃣ BASIC ANALYSIS
# -----------------------------
print("\n📊 Basic Energy Consumption Analysis:\n")

if 'energy_usage' in data.columns:
    print("Total Energy Consumed (kWh):", round(data['energy_usage'].sum(), 2))
    print("Average Daily Consumption (kWh):", round(data['energy_usage'].mean(), 2))
else:
    print("⚠️ Column 'energy_usage' not found. Please check your dataset headers.")

# Visualize consumption over time
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    plt.figure(figsize=(10,5))
    plt.plot(data['timestamp'], data['energy_usage'], color='teal')
    plt.title("Energy Consumption Over Time")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.show()

# -----------------------------
# 3️⃣ FEATURE ENGINEERING
# -----------------------------
# Example: extract time-based features
if 'timestamp' in data.columns:
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['weekday'] = data['timestamp'].dt.weekday

# Drop non-numeric/unnecessary columns
features = data.select_dtypes(include=[np.number]).dropna(axis=1)

# -----------------------------
# 4️⃣ TRAIN ML MODEL
# -----------------------------
if 'energy_usage' in features.columns:
    X = features.drop('energy_usage', axis=1)
    y = features['energy_usage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n🤖 Model Trained Successfully!")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

# -----------------------------
# 5️⃣ ENERGY SAVING RECOMMENDATIONS
# -----------------------------
print("\n💡 Smart Energy Recommendations:\n")

avg_usage = y.mean()
peak_usage = data.loc[data['energy_usage'].idxmax()] if 'energy_usage' in data.columns else None

if peak_usage is not None:
    print(f"- Your highest usage was on {peak_usage['timestamp']} with {peak_usage['energy_usage']:.2f} kWh.")
if avg_usage > 10:
    print("- Your average usage is quite high. Consider using appliances during off-peak hours.")
else:
    print("- Your energy usage is within normal range, great job!")

# Predict energy for next hour/day
if 'hour' in X.columns:
    next_hour = X.iloc[-1:].copy()
    next_hour['hour'] = (next_hour['hour'] + 1) % 24
    future_pred = model.predict(next_hour)
    print(f"\n🔮 Predicted energy usage for next hour: {future_pred[0]:.2f} kWh")

print("\n✅ Analysis complete. You can customize this model for cost prediction or appliance-level insights.")
