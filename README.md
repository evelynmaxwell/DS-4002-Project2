# Forecasting NYC Motor Vehicle Collisions


## 1. Software and Platform
- **Platform Used:** Mac was used to write and run scripts. The scripts can be run across platforms.
- **Software Used**: Python 3
 
- **Add-on Packages:**  
  - `pandas` - data loading and manipulation
  - `numpy` - numerical operations
  - `matplotlib` – plotting and visualization
  - `os`- file and directory management
  - `pathlib` - filesystem paths
  - `prophet` - time-series forecasting framework
  - `scikit-learn` (`sklearn`) - machine learning models and metrics
    - Subpackages, Classes, and Functions used:
      - `metrics` → (`mean_absolute_error`, `mean_squared_error`)
      - `preprocessing` → (`StandardScaler`)
  - `statsmodels` - statistic models
    - Subpackage and Class used:
      - `tsa.statespace.sarimax` → (SARIMAX)

## 2. Documentation Map
The hierarchy of folders and files contained in this project are as follows:

```text
DS-4002-Project2
├── DATA
│   ├── README.md
│   ├── aggregate_hourly_distributions.png
│   ├── collisions_cleaned.csv.zip
│   ├── collisions_raw.csv.zip
│   └── hourly_daily_heatmap.png
├── OUTPUT
│   ├── daily_baseline_model_figure.png
│   ├── daily_prophet_model.png
│   ├── hourly_baseline_model.png
│   └── hourly_prophet_model.png
├── SCRIPTS
│   ├── 1_BASELINE_daily.py
│   ├── 1_BASELINE_hourly.py
│   ├── 2_SARIMA_daily.py
|   ├── 2_SARIMA_hourly.py
|   ├── 3_Prophet_daily.py
|   ├── 3_Prophet_hourly.py
|   ├── 4_Prophet_Borough_Daily_Forecast.py
|   ├── 5_Prophet_Borough_Hourly_Forecast.py
│   └── preprocessing.py
├── LICENSE.md
└── README.md

```

## 3. Reproducing Our Results
  1. **Set up Python and install required add-on packages**
     - Clone this repository: https://github.com/evelynmaxwell/DS-4002-Project2/
     - Ensure you have Python 3 installed on your system.
     - See section 1 for packages needed.
  2. **Prepare the data**
     - Option A: Preprocess Raw Data Yourself
         - Download and unzip `collisions_raw.csv.zip` from `DATA` folder. Manually add `collisions_raw.csv` file to `DATA` folder.
         - Run the `preprocessing.py` file from the `SCRIPTS` folder, which will save preprocessed data in a new `collisions_cleaned.csv` file in the `DATA` folder.
     - Option B: Use Preprocessed Data
         - Download and unzip `collisions_cleaned.csv.zip` from `DATA` folder.
         - Manually add `collisions_cleaned.csv` file to the `DATA` folder in your IDE or local environment.
  3. **Run model scripts**
     - Navigate to the `SCRIPTS` folder.
     - Scripts 1-3 train and test on historical data only to evaluate model accuracy. Each pair corresponds to a specific model (baseline, SARIMA, Prophet) and should be executed in numerical order to reproduce results. Each script:
         - Prints evaluation metrics: RMSE, MAE, and MAPE.
         - Generates plots of observed vs. predicted collisions.
  4. **Forecast Future Collisions**
     - Scripts 4 and 5 provide templates for forecasting the number of collisions in any NYC borough at daily or hourly scales.
     - You can specify:
         - The borough name (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
         - A future date to generate the forecast
     
 

     
