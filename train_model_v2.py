# ===============================
# AI Fitness Tracker – Random Forest Model (v2)
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# 1. Load and clean data
# -------------------------------
df = pd.read_csv("Data/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv")

# Rename useful columns
df.rename(columns={
    'TotalSteps': 'steps',
    'Calories': 'calories',
    'VeryActiveMinutes': 'very_active_min',
    'FairlyActiveMinutes': 'fairly_active_min',
    'LightlyActiveMinutes': 'light_active_min',
    'SedentaryMinutes': 'sedentary_min'
}, inplace=True)

df = df.dropna()

# -------------------------------
# 2. Feature Engineering
# -------------------------------
# Create an "activity_index" that captures total effort
df['activity_index'] = (
    df['steps'] * (df['very_active_min'] + df['fairly_active_min']) /
    (df['sedentary_min'] + 1)
)

# Log-transform steps to stabilize scale differences
df['log_steps'] = np.log1p(df['steps'])

# -------------------------------
# 3. Select features and target
# -------------------------------
features = [
    'log_steps',
    'very_active_min',
    'fairly_active_min',
    'light_active_min',
    'sedentary_min',
    'activity_index'
]
target = 'calories'

X = df[features]
y = df[target]

# -------------------------------
# 4. Split into training/test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Train Random Forest Model
# -------------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate performance
# -------------------------------
preds = rf.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))

print("✅ Random Forest trained successfully!")
print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:.2f} calories")

# -------------------------------
# 7. Visualize Predictions
# -------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.6)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Calories Burned: Random Forest Prediction")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# -------------------------------
# 8. Feature Importance
# -------------------------------
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(8,5))
plt.barh(range(len(importances)), importances[sorted_idx], align="center")
plt.yticks(range(len(importances)), [features[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Which factors most affect calorie burn?")
plt.show()
