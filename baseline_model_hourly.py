import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# --- Load and Clean Data ---
df = pd.read_csv(
    "collisions_filtered.csv",
    parse_dates=["CRASH DATETIME"],
    on_bad_lines="skip",
    engine="python"
)

print(f"Loaded {len(df):,} rows successfully.")
print("Columns:", df.columns.tolist())

# --- Trim dataset to last 5 years ---
df = df[df["CRASH DATETIME"] >= (df["CRASH DATETIME"].max() - pd.DateOffset(years=5))]

# --- Hourly Aggregation ---
df = (
    df.set_index("CRASH DATETIME")
      .resample("h")
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

# --- Train/Test Split (80/20 chronological) ---
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

# --- Baseline Model: 24-Hour Moving Average ---
window = 24
train.loc[:, "rolling_mean"] = train["TOTAL_COLLISIONS"].rolling(window=window).mean()

baseline_preds = []
history = list(train["TOTAL_COLLISIONS"].iloc[-window:])

for actual in test["TOTAL_COLLISIONS"]:
    pred = np.mean(history[-window:])
    baseline_preds.append(pred)
    history.append(actual)

test.loc[:, "baseline_pred"] = baseline_preds

# --- Evaluate ---
rmse = np.sqrt(mean_squared_error(test["TOTAL_COLLISIONS"], test["baseline_pred"]))
mae = mean_absolute_error(test["TOTAL_COLLISIONS"], test["baseline_pred"])
mape = np.mean(
    np.abs((test["TOTAL_COLLISIONS"] - test["baseline_pred"]) /
           test["TOTAL_COLLISIONS"].replace(0, np.nan))
) * 100

print("\nHourly Baseline Model (24-hour Moving Average):")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# --- Visualization ---
plt.figure(figsize=(12,6))
plt.plot(train["CRASH DATETIME"], train["TOTAL_COLLISIONS"], color="gray", alpha=0.4, label="Train Data")
plt.plot(test["CRASH DATETIME"], test["TOTAL_COLLISIONS"], color="black", linewidth=1, label="Actual (Test)")
plt.plot(test["CRASH DATETIME"], test["baseline_pred"], color="blue", linewidth=1.2, label="Baseline Prediction")

plt.axvline(x=test["CRASH DATETIME"].iloc[0], color="red", linestyle="--", label="Train/Test Split")
plt.title("Hourly Baseline Model (24-hour Moving Average)")
plt.xlabel("Datetime")
plt.ylabel("Total Collisions")
plt.legend()
plt.tight_layout()
plt.show()
