# ===============================
# AI Fitness Tracker – Data Check
# ===============================

import pandas as pd

# Load your dataset
df = pd.read_csv("Data/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv")


# Confirm data loaded correctly
print("✅ Data loaded successfully!")
print(df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nPreview:")
print(df.head())
