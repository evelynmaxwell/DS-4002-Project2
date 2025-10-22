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
│   ├── collisions_cleaned.csv.zip
│   └── collisions_raw.csv.zip
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
     - You can either preprocess the raw data yourself or use the pre-cleaned dataset.
     - Option A: Preprocess Raw Data Yourself
         - Navigate to the `DATA` folder
         - Download and unzip `collisions_raw.csv.zip` and move the extracted       `collisions_raw.csv` file back into the `DATA` folder.
         - Navigate to the `SCRIPTS` folder and run the `preprocessing.py` file, which will save the preprocessed data in a new `collisions_cleaned.csv` file within the `DATA` folder.
     - Option B: Use Preprocessed Data
         - Download the `collisions_cleaned.csv.zip` file from `DATA` folder.
         - Unzip and manually add the `collisions_cleaned.csv` file to the DATA folder in your IDE or local environment.
  3. **Run model scripts**
     - Navigate to the `SCRIPTS` folder.
     - Scripts 1-3 train and test on historical data only to evaluate model accuracy. Each pair corresponds to a specific model (baseline, SARIMA, Prophet) and should be executed in numerical order to reproduce results.
         - Step 1: Baseline models for daily/hourly forecasts
         - Step 2: SARIMA models for daily/hourly forecasts
         - Step 3: Prophet models for daily/hourly forecasts
     - Each modeling script (1-3) automatically:
         - Prints evaluation metrics: RMSE, MAE, and MAPE.
         - Generates plots of observed vs. predicted collisions and saves to `OUTPUT` folder.
  4. **Forecast Future Collisions**
     - Scripts 4 and 5 provide templates for forecasting the number of collisions in any NYC borough at daily or hourly scales.
     - You can specify:
         - The borough name (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
         - A future date to generate the forecast
     
 

     
