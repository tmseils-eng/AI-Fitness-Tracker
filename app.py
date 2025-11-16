# ===============================
# AI Fitness Tracker – Streamlit App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# 1. Page setup
# -------------------------------
st.set_page_config(page_title="AI Fitness Tracker", page_icon="💪", layout="centered")
st.title("💪 AI Fitness Tracker")
st.markdown("### Predict calories burned based on your daily activity!")

# -------------------------------
# 2. Load or train model
# -------------------------------
@st.cache_resource
def load_model():
    # Load dataset
    df = pd.read_csv("Data/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv")
    df.rename(columns={
        'TotalSteps': 'steps',
        'Calories': 'calories',
        'VeryActiveMinutes': 'very_active_min',
        'FairlyActiveMinutes': 'fairly_active_min',
        'LightlyActiveMinutes': 'light_active_min',
        'SedentaryMinutes': 'sedentary_min'
    }, inplace=True)

    # Feature engineering
    df['activity_index'] = (
        df['steps'] * (df['very_active_min'] + df['fairly_active_min']) /
        (df['sedentary_min'] + 1)
    )
    df['log_steps'] = np.log1p(df['steps'])

    # Train model
    features = [
        'log_steps', 'very_active_min', 'fairly_active_min',
        'light_active_min', 'sedentary_min', 'activity_index'
    ]
    X = df[features]
    y = df['calories']

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# -------------------------------
# 3. Sidebar for user input
# -------------------------------
st.sidebar.header("Enter Your Activity Data")

steps = st.sidebar.number_input("Steps", min_value=0, max_value=50000, value=8000)
very_active_min = st.sidebar.number_input("Very Active Minutes", min_value=0, max_value=300, value=30)
fairly_active_min = st.sidebar.number_input("Fairly Active Minutes", min_value=0, max_value=300, value=20)
light_active_min = st.sidebar.number_input("Lightly Active Minutes", min_value=0, max_value=600, value=120)
sedentary_min = st.sidebar.number_input("Sedentary Minutes", min_value=0, max_value=1500, value=600)

# -------------------------------
# 4. Prepare input for prediction
# -------------------------------
activity_index = steps * (very_active_min + fairly_active_min) / (sedentary_min + 1)
log_steps = np.log1p(steps)

input_data = pd.DataFrame({
    'log_steps': [log_steps],
    'very_active_min': [very_active_min],
    'fairly_active_min': [fairly_active_min],
    'light_active_min': [light_active_min],
    'sedentary_min': [sedentary_min],
    'activity_index': [activity_index]
})

# -------------------------------
# 5. Predict calories
# -------------------------------
if st.sidebar.button("Predict Calories Burned"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔥 Estimated Calories Burned: **{prediction:.0f} kcal**")

    # Insight based on activity index
    if activity_index < 5000:
        st.warning("💡 Tip: Try increasing your active minutes for a better burn!")
    elif activity_index < 15000:
        st.info("✅ You're moderately active — keep it up!")
    else:
        st.success("🏅 Excellent activity level! You're crushing it.")

# -------------------------------
# 6. Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + scikit-learn")
