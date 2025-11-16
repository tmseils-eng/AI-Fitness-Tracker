# ===============================
# AI Fitness Tracker – Train Calorie Prediction Model
# ===============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import math

# -------------------------------
# 1. Load and clean data
# -------------------------------
df = pd.read_csv("Data/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv")

# Rename and keep useful columns
df.rename(columns={
    'TotalSteps': 'steps',
    'Calories': 'calories',
    'VeryActiveMinutes': 'very_active_min',
    'FairlyActiveMinutes': 'fairly_active_min',
    'LightlyActiveMinutes': 'light_active_min',
    'SedentaryMinutes': 'sedentary_min'
}, inplace=True)

# Drop rows with missing values
df = df.dropna()

# -------------------------------
# 2. Choose features and target
# -------------------------------
features = ['steps', 'very_active_min', 'fairly_active_min', 'light_active_min', 'sedentary_min']
target = 'calories'

X = df[features]
y = df[target]

# -------------------------------
# 3. Split into training/test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 4. Train linear regression model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluate model performance
# -------------------------------
preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))

print(f"✅ Model trained successfully!")
print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:.2f} calories")

# -------------------------------
# 6. Visualize predictions
# -------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.6)
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Calories Burned: Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # reference line
plt.show()
