"""
Description:
    This script computes and evaluates a baseline 7-day moving average model 
    to forecast daily motor vehicle collisions in New York City using 
    cleaned crash data.  

Inputs:
    - collisions_filtered.csv
        A cleaned dataset of NYC motor vehicle collisions containing a 
        'CRASH DATETIME' column with timestamps for each collision event.

Process:
    1. Reads and parses the collision data.
    2. Trims the dataset to the most recent five years for relevance.
    3. Aggregates crash counts at a daily frequency.
    4. Splits the time series chronologically into training (80%) and testing (20%) sets.
    5. Applies a 7-day rolling average as a baseline forecasting model.
    6. Evaluates model accuracy using RMSE, MAE, and MAPE metrics.

Outputs:
    - Printed model evaluation metrics:
        RMSE, MAE, and MAPE (baseline expected ~ RMSE: 25–35, MAE: 18–25, MAPE: ~10–20%)
    - A visualization comparing actual vs. predicted daily collisions, 
      with a red dashed line marking the train/test split.
"""


import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load and Clean Data
df = pd.read_csv(
    "collisions_filtered.csv",
    parse_dates=["CRASH DATETIME"],
    on_bad_lines="skip",
    engine="python"
)

print(f"Loaded {len(df):,} rows successfully.")
print("Columns:", df.columns.tolist())

# Trim dataset to last 5 years
df = df[df["CRASH DATETIME"] >= (df["CRASH DATETIME"].max() - pd.DateOffset(years=5))]

# Daily Aggregation
df = (
    df.set_index("CRASH DATETIME")
      .resample("d")
      .agg({
          "COLLISION_ID": "count",
          "NUMBER OF PERSONS INJURED": "sum",
          "NUMBER OF PERSONS KILLED": "sum"
      })
      .reset_index()
      .rename(columns={"COLLISION_ID": "TOTAL_COLLISIONS"})
)

print(f"Aggregated shape: {df.shape}")
print(df.head())

# Train/Test Split (80/20 chronological)
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

# Baseline Model: 7-Day Moving Average
window = 7
train.loc[:, "rolling_mean"] = train["TOTAL_COLLISIONS"].rolling(window=window).mean()

baseline_preds = []
history = list(train["TOTAL_COLLISIONS"].iloc[-window:])

for actual in test["TOTAL_COLLISIONS"]:
    pred = np.mean(history[-window:])
    baseline_preds.append(pred)
    history.append(actual)

test.loc[:, "baseline_pred"] = baseline_preds

# Evaluate
rmse = np.sqrt(mean_squared_error(test["TOTAL_COLLISIONS"], test["baseline_pred"]))
mae = mean_absolute_error(test["TOTAL_COLLISIONS"], test["baseline_pred"])
mape = np.mean(
    np.abs((test["TOTAL_COLLISIONS"] - test["baseline_pred"]) /
           test["TOTAL_COLLISIONS"].replace(0, np.nan))
) * 100

print("\nDaily Baseline Model (7-day Moving Average):")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# Visualization
plt.figure(figsize=(12,6))
plt.plot(train["CRASH DATETIME"], train["TOTAL_COLLISIONS"], color="gray", alpha=0.4, label="Train Data")
plt.plot(test["CRASH DATETIME"], test["TOTAL_COLLISIONS"], color="black", linewidth=1, label="Actual (Test)")
plt.plot(test["CRASH DATETIME"], test["baseline_pred"], color="blue", linewidth=1.2, label="Baseline Prediction")

plt.axvline(x=test["CRASH DATETIME"].iloc[0], color="red", linestyle="--", label="Train/Test Split")
plt.title("Daily Baseline Model (7-day Moving Average)")
plt.xlabel("Date")
plt.ylabel("Total Collisions")
plt.legend()
plt.tight_layout()
plt.show()
