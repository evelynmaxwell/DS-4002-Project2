"""
Description:
    This script computes and evaluates a baseline 24-hour moving average model 
    to forecast hourly motor vehicle collisions in New York City using 
    cleaned crash data.

Inputs:
    - ../DATA/collisions_cleaned.csv
        A cleaned dataset of NYC motor vehicle collisions containing a 
        'CRASH DATETIME' column with timestamps for each collision event.

Process:
    1. Reads and parses the collision data.
    2. Trims the dataset to the most recent five years for relevance.
    3. Aggregates crash counts at an hourly frequency.
    4. Splits the time series chronologically into training (80%) and testing (20%) sets.
    5. Applies a 24-hour rolling average as a baseline forecasting model.
    6. Evaluates performance using RMSE, MAE, and MAPE metrics.

Outputs:
    - Printed model evaluation metrics:
        RMSE, MAE, and MAPE (baseline expected ~ RMSE: 4–6, MAE: 2–4, MAPE varies)
    - A visualization comparing actual vs. predicted hourly collisions, 
      with a red line marking the train/test split.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Read and preprocess data
df = pd.read_csv('../DATA/collisions_cleaned.csv',
                 low_memory=False,
                 dtype={'BOROUGH': 'string'})

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Keep only last 2 years of data (hourly dataset is large)
df = df[df['timestamp'] >= (df['timestamp'].max() - pd.DateOffset(years=2))]

# 2. Aggregate by hour
hourly = (
    df.set_index('timestamp')
      .resample('h')
      .size()
      .reset_index(name='TOTAL_COLLISIONS')
)

# 3. Train-test split (80/20)
split_idx = int(len(hourly) * 0.8)
train = hourly.iloc[:split_idx].copy()
test = hourly.iloc[split_idx:].copy()

# 4. Baseline (24-hour moving average)
window = 24
train['rolling_mean'] = train['TOTAL_COLLISIONS'].rolling(window=window).mean()
last_train_mean = train['rolling_mean'].iloc[-1]
baseline_preds = np.concatenate([train['rolling_mean'].iloc[-window:], np.full(len(test), last_train_mean)])
test['baseline_pred'] = baseline_preds[-len(test):]

# 5. Evaluate metrics
rmse = np.sqrt(mean_squared_error(test['TOTAL_COLLISIONS'], test['baseline_pred']))
mae = mean_absolute_error(test['TOTAL_COLLISIONS'], test['baseline_pred'])
mape = np.mean(np.abs((test['TOTAL_COLLISIONS'] - test['baseline_pred']) / (test['TOTAL_COLLISIONS'] + 1e-5))) * 100

print(f"\nBaseline Model (24-Hour Moving Average):")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# 6. Plot results
plt.figure(figsize=(12, 6))
plt.plot(hourly['timestamp'], hourly['TOTAL_COLLISIONS'], label='Actual', color='black', linewidth=0.8)
plt.plot(test['timestamp'], test['baseline_pred'], label='24-Hour Baseline Prediction', color='blue', linestyle='--')
plt.axvline(x=hourly['timestamp'].iloc[split_idx], color='red', linestyle='--', label='Train/Test Split')
plt.xlabel('Timestamp')
plt.ylabel('Total Collisions per Hour')
plt.title('Baseline 24-Hour Moving Average Forecast (Hourly)')
plt.legend()
plt.tight_layout()
plt.show()