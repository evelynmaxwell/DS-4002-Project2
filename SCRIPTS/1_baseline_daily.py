"""
Description:
    This script computes and evaluates a baseline 7-day moving average model 
    to forecast daily motor vehicle collisions in New York City using cleaned crash data.

Inputs:
    - ../DATA/collisions_cleaned.csv
        A cleaned dataset of NYC motor vehicle collisions containing a 
        'timestamp' column representing crash date and time.

Process:
    1. Reads and parses the collision data.
    2. Trims the dataset to the most recent five years for relevance.
    3. Aggregates crash counts at a daily frequency.
    4. Splits the time series chronologically into training (80%) and testing (20%) sets.
    5. Applies a 7-day moving average as a baseline forecasting model.
    6. Evaluates performance using RMSE, MAE, and MAPE metrics.

Outputs:
    - Printed model evaluation metrics:
        RMSE, MAE, and MAPE (baseline expected ~ RMSE: 25–35, MAE: 18–25, MAPE: 10–20%)
    - A visualization comparing actual vs. predicted daily collisions, 
      with a red dashed line marking the train/test split.
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

# Keep only last 5 years of data for relevance
df = df[df['timestamp'] >= (df['timestamp'].max() - pd.DateOffset(years=5))]

# 2. Aggregate by day
daily = (
    df.set_index('timestamp')
      .resample('d')
      .size()
      .reset_index(name='TOTAL_COLLISIONS')
)

# 3. Train-test split (80/20)
split_idx = int(len(daily) * 0.8)
train = daily.iloc[:split_idx].copy()
test = daily.iloc[split_idx:].copy()

# 4. Baseline (7-day moving average)
window = 7
train['rolling_mean'] = train['TOTAL_COLLISIONS'].rolling(window=window).mean()
last_train_mean = train['rolling_mean'].iloc[-1]
baseline_preds = np.concatenate([train['rolling_mean'].iloc[-window:], np.full(len(test), last_train_mean)])
test['baseline_pred'] = baseline_preds[-len(test):]

# 5. Evaluate metrics
rmse = np.sqrt(mean_squared_error(test['TOTAL_COLLISIONS'], test['baseline_pred']))
mae = mean_absolute_error(test['TOTAL_COLLISIONS'], test['baseline_pred'])
mape = np.mean(np.abs((test['TOTAL_COLLISIONS'] - test['baseline_pred']) / (test['TOTAL_COLLISIONS'] + 1e-5))) * 100

print(f"\nBaseline Model (7-Day Moving Average):")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# 6. Plot results
plt.figure(figsize=(12, 6))
plt.plot(daily['timestamp'], daily['TOTAL_COLLISIONS'], label='Actual', color='black', linewidth=1)
plt.plot(test['timestamp'], test['baseline_pred'], label='7-Day Baseline Prediction', color='blue', linestyle='--')
plt.axvline(x=daily['timestamp'].iloc[split_idx], color='red', linestyle='--', label='Train/Test Split')
plt.xlabel('Date')
plt.ylabel('Total Collisions per Day')
plt.title('Baseline 7-Day Moving Average Forecast (Daily)')
plt.legend()
plt.tight_layout()
plt.show()
