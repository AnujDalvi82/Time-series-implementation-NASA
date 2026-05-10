# ## 1. Introduction & Objectives
# 
# This notebook demonstrates a comprehensive machine learning pipeline for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.
# 
# ### Project Overview
# The CMAPSS dataset simulates aircraft engine turbofan degradation over time. Each engine starts with some initial wear and gradually degrades during operation. By analyzing sensor telemetry data, we can predict when an engine will fail, enabling predictive maintenance strategies.
# 
# ### Objectives
# 1. **Data Exploration**: Understand the structure and characteristics of the CMAPSS dataset
# 2. **Feature Engineering**: Create meaningful features from raw sensor readings
# 3. **Model Development**: Build and train a Random Forest regression model
# 4. **Evaluation**: Assess model performance using industry-standard metrics
# 
# ### Why This Matters
# Accurate RUL prediction helps reduce unplanned downtime, optimize maintenance scheduling, lower operational costs, and improve flight safety.
# 
# ### Environment Setup
# 
# This section imports all required libraries for data manipulation, mathematical operations, visualizations, and machine learning components.
# 
# This notebook demonstrates a comprehensive machine learning pipeline for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.
# 
# ### Project Overview
# The CMAPSS dataset simulates aircraft engine turbofan degradation over time. Each engine starts with some initial wear and gradually degrades during operation. By analyzing sensor telemetry data, we can predict when an engine will fail, enabling predictive maintenance strategies.
# 
# ### Objectives
# 1. **Data Exploration**: Understand the structure and characteristics of the CMAPSS dataset
# 2. **Feature Engineering**: Create meaningful features from raw sensor readings
# 3. **Model Development**: Build and train a Random Forest regression model
# 4. **Evaluation**: Assess model performance using industry-standard metrics
# 
# ### Why This Matters
# Accurate RUL prediction helps:
# - Reduce unplanned maintenance downtime
# - Optimize maintenance scheduling
# - Lower operational costs
# - Improve flight safety
# 
# ### Environment Setup
# 
# This section imports all required libraries for data manipulation, mathematical operations, visualizations, and machine learning components.
# 
# This notebook demonstrates a comprehensive machine learning pipeline for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. We encompass exploratory data analysis, feature engineering, model training, and robust evaluation using Random Forest Regressors.
# 
# ### Environment Setup
# 
# In this section, we import all required libraries for data manipulation, mathematical operations, visualizations, and machine learning components.
# This notebook demonstrates how to predict the Remaining Useful Life (RUL) of turbofan engines using the CMAPSS dataset. We encompass exploratory data analysis, feature engineering, and robust evaluation using Random Forest Regressors.
# 
# ### Environment Setup
# Importing required libraries for data manipulation, mathematical operations, visualizations, and machine learning components.

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.ensemble import RandomForestRegressor

import warnings 
warnings.filterwarnings("ignore")

# ### Feature Definitions
# 
# The raw CMAPSS data files contain numerical values without headers. We need to define meaningful column names to structure our dataset properly.
# 
# **Column Categories:**
# 1. **Engine Identifier**: Unique ID for each engine unit
# 2. **Cycle Number**: Operational cycle/time step
# 3. **Operational Settings (op_1, op_2, op_3)**: Engine operating conditions
# 4. **Sensor Measurements (sensor_1 - sensor_21)**: Telemetry data from various sensors
# 
# Understanding these features is crucial for proper data preprocessing and feature engineering.
# 
# The raw CMAPSS data files contain numerical values without headers. We need to define meaningful column names to structure our dataset properly.
# 
# **Column Categories:**
# 1. **Engine Identifier**: Unique ID for each engine unit
# 2. **Cycle Number**: Operational cycle/time step
# 3. **Operational Settings (op_1, op_2, op_3)**: Engine operating conditions
# 4. **Sensor Measurements (sensor_1 - sensor_21)**: Telemetry data from various sensors
# 
# Understanding these features is crucial for proper data preprocessing and feature engineering.
# Defining logical column names separated into distinct operational settings and sensor measurements to successfully structure the incoming raw dataset.

op_cols = ["op_1", "op_2", "op_3"]
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

columns = ["engine_id", "cycle"] + op_cols + sensor_cols

# ### Data Loading
# 
# We load the training data from `train_FD001.txt`, which contains the complete operational history of multiple turbofan engines until failure. Each row represents one operational cycle for one engine.
# 
# **Data Characteristics:**
# - Multiple engine units tracked over time
# - Each engine runs until failure (different lifespans)
# - Contains operational settings and 21 sensor readings per cycle
# 
# The data is loaded using pandas with whitespace as the delimiter since the file has no headers.
# 
# We load the training data from `train_FD001.txt`, which contains the complete operational history of multiple turbofan engines until failure. Each row represents one operational cycle for one engine.
# 
# **Data Characteristics:**
# - Multiple engine units tracked over time
# - Each engine runs until failure (different lifespans)
# - Contains operational settings and 21 sensor readings per cycle
# 
# The data is loaded using pandas with whitespace as the delimiter since the file has no headers.
# Loading the training data (`train_FD001.txt`) into a structured pandas DataFrame and appending the mapped columns.

train_df = pd.read_csv("train_FD001.txt", sep="\s+", header=None)

train_df.columns = columns
train_df.head()

train_df.shape

# ### Initial Data Inspection
# 
# Before building our predictive model, we need to understand the dataset structure. This step provides:
# - **Data Types**: Verify numerical values are properly typed
# - **Missing Values**: Check for any null entries
# - **Dataset Size**: Understand the scale of data
# 
# This inspection ensures our data is ready for analysis and helps identify any preprocessing needs.
# 
# Before building our predictive model, we need to understand the dataset structure. This step provides:
# - **Data Types**: Verify numerical values are properly typed
# - **Missing Values**: Check for any null entries
# - **Dataset Size**: Understand the scale of data
# 
# This inspection ensures our data is ready for analysis and helps identify any preprocessing needs.
# Running a succinct summary of the overall training dataset to interpret data types and verify the presence of missing values.

train_df.info()

# ### Computing Max Cycles
# 
# To calculate the Remaining Useful Life (RUL), we first need to determine when each engine failed. The maximum cycle number for each engine represents its total operational lifespan before failure.
# 
# This information is essential because:
# - Each engine has a different lifespan
# - RUL is calculated relative to the failure point
# - We need to know when each engine reached end-of-life
# 
# To calculate the Remaining Useful Life (RUL), we first need to determine when each engine failed. The maximum cycle number for each engine represents its total operational lifespan before failure.
# 
# This information is essential because:
# - Each engine has a different lifespan
# - RUL is calculated relative to the failure point
# - We need to know when each engine reached end-of-life
# To establish our target labels for RUL, we first resolve the maximum number of life cycles reached by each distinct turbofan engine prior to failure.

max_cycle = train_df.groupby("engine_id")["cycle"].max()
max_cycle.head()

train_df["max_cycle"] = train_df["engine_id"].map(max_cycle)
train_df.head()

# ### Deriving RUL (Remaining Useful Life)
# 
# The **Remaining Useful Life (RUL)** is our target variable - the number of cycles remaining before engine failure. We calculate it as:
# 
# ```
# RUL = Maximum Cycle (at failure) - Current Cycle
# ```
# 
# **Key Points:**
# - RUL decreases as the engine accumulates more cycles
# - At the time of failure, RUL = 0
# - Early cycles have higher RUL values
# - This creates a linear degradation assumption
# 
# The **Remaining Useful Life (RUL)** is our target variable - the number of cycles remaining before engine failure. We calculate it as:
# 
# ```
# RUL = Maximum Cycle (at failure) - Current Cycle
# ```
# 
# **Key Points:**
# - RUL decreases as the engine accumulates more cycles
# - At the time of failure, RUL = 0
# - Early cycles have higher RUL values
# - This creates a linear degradation assumption
# The Remaining Useful Life (RUL) is deduced linearly by subtracting the present cycle step from the engine's absolute overall failure cycle.

train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
train_df.head()

train_df[train_df["engine_id"] == 1].tail()

engine1 = train_df[train_df["engine_id"] == 1]
engine1.head()

plt.figure(figsize=(8, 5))

plt.plot(engine1["cycle"], engine1["sensor_11"])
plt.xlabel("Cycle")
plt.ylabel("Sensor 11")
plt.title("Sensor 11 degradation pattern (Engine 1)")


plt.show()

sensors_to_plot = ["sensor_2", "sensor_3", "sensor_4", "sensor_11"]

plt.figure(figsize=(10, 6))

for sensor in sensors_to_plot:
    plt.plot(engine1["cycle"], engine1[sensor], label=sensor)

plt.xlabel("Cycle")
plt.ylabel("Sensor Value")
plt.title("Sensor behavior over engine life (Engine 1)")
plt.legend()

plt.show()


# Dropping columns with little to no variance.
drop_sensors = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
train_df = train_df.drop(columns=drop_sensors)

train_df.columns

train_df.shape

y = train_df["RUL"]
X = train_df.drop(columns=["RUL", "engine_id", "max_cycle"])

X.shape, y.shape

X.columns

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ### Machine Learning Pipeline Initialization
# 
# We construct a scikit-learn Pipeline that chains together:
# 1. **StandardScaler**: Normalizes feature values to have zero mean and unit variance
# 2. **RandomForestRegressor**: Ensemble learning method for regression
# 
# **Why this pipeline?**
# - **Scaling**: Ensures all features contribute equally to the model
# - **Random Forest**: Handles non-linear relationships, resistant to overfitting
# - **Pipeline**: Ensures consistent preprocessing during training and prediction
# 
# We construct a scikit-learn Pipeline that chains together:
# 1. **StandardScaler**: Normalizes feature values to have zero mean and unit variance
# 2. **RandomForestRegressor**: Ensemble learning method for regression
# 
# **Why this pipeline?**
# - **Scaling**: Ensures all features contribute equally to the model
# - **Random Forest**: Handles non-linear relationships, resistant to overfitting
# - **Pipeline**: Ensures consistent preprocessing during training and prediction
# Constructing a Scikit-Learn `Pipeline`. It sequentially chains a `StandardScaler` to appropriately normalize the feature space, followed by a robust `RandomForestRegressor` as our predictive foundation.

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# ### Model Training
# 
# We fit the complete pipeline (scaler + Random Forest) on our training data. The model learns the relationship between:
# - Input features: Sensor readings, operational settings
# - Target: Remaining Useful Life
# 
# **Training Process:**
# - Data is split into training (80%) and validation (20%) sets
# - The scaler transforms features to normalized values
# - The Random Forest learns from the training data
# 
# We fit the complete pipeline (scaler + Random Forest) on our training data. The model learns the relationship between:
# - Input features: Sensor readings, operational settings
# - Target: Remaining Useful Life
# 
# **Training Process:**
# - Data is split into training (80%) and validation (20%) sets
# - The scaler transforms features to normalized values
# - The Random Forest learns from the training data
# Fitting our entire Random Forest scaling-and-regression pipeline securely onto the partitioned training observations.

pipe.fit(X_train, y_train)

# ### Validation Prediction
# 
# We use the trained pipeline to predict RUL values for the held-out validation set. This gives us immediate predictions to evaluate how well the model generalizes to unseen data.
# 
# We use the trained pipeline to predict RUL values for the held-out validation set. This gives us immediate predictions to evaluate how well the model generalizes to unseen data.
# Deploying the trained pipeline onto the held-out validation segment to retrieve immediate RUL estimations.

y_pred = pipe.predict(X_valid)

# ### Validation Internal Metrics
# 
# We evaluate model performance using standard regression metrics:
# 
# 1. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual RUL
# 2. **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
# 
# These metrics help us understand how accurate our predictions are on average and whether the model generalizes well.
# 
# We evaluate model performance using standard regression metrics:
# 
# 1. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual RUL
# 2. **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
# 
# These metrics help us understand:
# - How accurate our predictions are on average
# - The spread of prediction errors
# - Whether the model generalizes well
# Calculating universally standard evaluation metrics including Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to gauge early generalizability.

mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# ### Extracting Feature Importances
# 
# Random Forest provides built-in feature importance scores based on how much each feature contributes to reducing prediction error. We extract these importance values to understand which sensors and features are most predictive of engine degradation.
# 
# Random Forest provides built-in feature importance scores based on how much each feature contributes to reducing prediction error. We extract these importance values to understand which sensors and features are most predictive of engine degradation.
# Extracting built-in permutation weights to understand which parameters structurally contribute the strongest towards estimating failure timelines.

model = pipe.named_steps["model"]
importances = model.feature_importances_

# ### Feature Importance Tabulation
# 
# We organize the feature importances into a sorted DataFrame to identify:
# - Which sensors provide the most degradation information
# - The relative importance of different features
# - Potential candidates for feature selection
# 
# We organize the feature importances into a sorted DataFrame to identify:
# - Which sensors provide the most degradation information
# - The relative importance of different features
# - Potential candidates for feature selection
# Structuring, compiling, and ordering the evaluated feature importances into a descending DataFrame chart.

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

feature_importance.head(10)

# ### Visualizing Feature Importances
# 
# We create a horizontal bar chart showing the top 15 most important features. This visualization helps identify key sensors for RUL prediction and understand which features drive the model's decisions.
# 
# We create a horizontal bar chart showing the top 15 most important features. This visualization helps:
# - Identify key sensors for RUL prediction
# - Understand which features drive the model's decisions
# - Guide feature engineering efforts
# Generating an easy-to-read seaborn bar chart demonstrating the top 15 most instrumental sensors for modeling the RUL objective.

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(15),
    x="importance",
    y="feature"
)

plt.title("Top 15 Feature Importances (Random Forest)")
plt.show()

# ### Time-Series Re-sorting
# 
# We sort the DataFrame by engine_id and cycle to ensure time-series operations work correctly. This ordering is critical for rolling window calculations, computing cycle-to-cycle changes (deltas), and maintaining temporal sequence within each engine.
# 
# We sort the DataFrame by engine_id and cycle to ensure time-series operations work correctly. This ordering is critical for:
# - Rolling window calculations
# - Computing cycle-to-cycle changes (deltas)
# - Maintaining temporal sequence within each engine
# Rigidly aligning and sorting the DataFrame explicitly by unique engines and strictly successive cycles. This ensures moving operations act sequentially.

train_df = train_df.sort_values(["engine_id", "cycle"])

# ### Selecting Highly Informative Sensors
# 
# Based on our EDA and feature importance analysis, we select the top sensors that show clear degradation patterns. These sensors have high feature importance and show consistent trends as engines degrade.
# 
# Using only informative sensors reduces noise and improves model performance.
# 
# Based on our EDA and feature importance analysis, we select the top sensors that show clear degradation patterns. These sensors:
# - Have high feature importance
# - Show consistent trends as engines degrade
# - Provide meaningful signal for prediction
# 
# Using only informative sensors reduces noise and improves model performance.
# Isolating a hand-picked subset of top-tier expressive sensors (derived via EDA & Random Forest Importances) to hone in on signal extraction and minimize ambient dataset noise.

top_sensors = ["sensor_11", "sensor_9", "sensor_4", "sensor_12", "sensor_14", "sensor_7", "sensor_15", "sensor_21", "sensor_2"]

# ### Rolling Statistical Mean Engineering
# 
# We create rolling average features (window size = 5 cycles) to smooth out sensor noise, capture underlying degradation trends, and provide stable representations of engine health. The rolling mean represents the local average behavior of each sensor over recent cycles.
# 
# We create rolling average features (window size = 5 cycles) to:
# - Smooth out sensor noise
# - Capture underlying degradation trends
# - Provide stable representations of engine health
# 
# The rolling mean represents the local average behavior of each sensor over recent cycles.
# Synthesizing simple 5-cycle wide rolling averages continuously over the premier sensors to smooth fluctuating sensor noise and unveil fundamental linear breakdown trends.

for sensor in top_sensors:
    train_df[f"{sensor}_rollmean"] = (
        train_df.groupby("engine_id")[sensor]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# ### Rolling Standard Deviation Engineering
# 
# We compute rolling standard deviation features to detect changes in sensor variability, identify periods of unusual engine behavior, and capture stress-related patterns that indicate approaching failure. High variance in certain sensors often signals imminent mechanical issues.
# 
# We compute rolling standard deviation features to:
# - Detect changes in sensor variability
# - Identify periods of unusual engine behavior
# - Capture stress-related patterns that indicate approaching failure
# 
# High variance in certain sensors often signals imminent mechanical issues.
# Creating subsequent 5-cycle rolling standard deviations on paramount sensors to empirically capture rapidly shifting local variance commonly symptomatic of imminent machine stress.

for sensor in top_sensors:
    train_df[f"{sensor}_rollstd"] = (
        train_df.groupby("engine_id")[sensor]
        .rolling(window=5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

# ### Calculating Trajectory Deltas
# 
# Delta features represent cycle-to-cycle changes in sensor values. These features capture the rate of degradation, short-term sensor fluctuations, and local trends in engine health.
# 
# Delta features represent cycle-to-cycle changes in sensor values:
# - **Delta = Current Value - Previous Value**
# 
# These features capture:
# - Rate of degradation
# - Short-term sensor fluctuations
# - Local trends in engine health
# 
# Positive or negative deltas can indicate accelerating or stabilizing degradation.
# Extracting precise cycle-over-cycle momentum changes (deltas) iteratively within each unit allowing the model to intrinsically assess the pace of functional decay.

for sensor in top_sensors:
    train_df[f"{sensor}_delta"] = (
        train_df.groupby("engine_id")[sensor]
        .diff()
    )

train_df = train_df.fillna(0)

train_df.shape

y = train_df["RUL"]
X = train_df.drop(columns=["RUL", "engine_id", "max_cycle"])

X.shape

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ### Model Training
# 
# We fit the complete pipeline (scaler + Random Forest) on our training data. The model learns the relationship between:
# - Input features: Sensor readings, operational settings
# - Target: Remaining Useful Life
# 
# **Training Process:**
# - Data is split into training (80%) and validation (20%) sets
# - The scaler transforms features to normalized values
# - The Random Forest learns from the training data
# 
# We fit the complete pipeline (scaler + Random Forest) on our training data. The model learns the relationship between:
# - Input features: Sensor readings, operational settings
# - Target: Remaining Useful Life
# 
# **Training Process:**
# - Data is split into training (80%) and validation (20%) sets
# - The scaler transforms features to normalized values
# - The Random Forest learns from the training data
# Fitting our entire Random Forest scaling-and-regression pipeline securely onto the partitioned training observations.

pipe.fit(X_train, y_train)

# ### Validation Prediction
# 
# We use the trained pipeline to predict RUL values for the held-out validation set. This gives us immediate predictions to evaluate how well the model generalizes to unseen data.
# 
# We use the trained pipeline to predict RUL values for the held-out validation set. This gives us immediate predictions to evaluate how well the model generalizes to unseen data.
# Deploying the trained pipeline onto the held-out validation segment to retrieve immediate RUL estimations.

y_pred = pipe.predict(X_valid)

# ### Validation Internal Metrics
# 
# We evaluate model performance using standard regression metrics:
# 
# 1. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual RUL
# 2. **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
# 
# These metrics help us understand how accurate our predictions are on average and whether the model generalizes well.
# 
# We evaluate model performance using standard regression metrics:
# 
# 1. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual RUL
# 2. **Root Mean Squared Error (RMSE)**: Penalizes larger errors more heavily
# 
# These metrics help us understand:
# - How accurate our predictions are on average
# - The spread of prediction errors
# - Whether the model generalizes well
# Calculating universally standard evaluation metrics including Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to gauge early generalizability.

mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# ### Test Cohort Data Ingestion
# 
# We load the test data from `test_FD001.txt`. Unlike training data, test engines have not yet failed and stop at various points during their operational life, requiring prediction of remaining cycles until failure.
# 
# We load the test data from `test_FD001.txt`. Unlike training data, test engines:
# - Have not yet failed
# - Stop at various points during their operational life
# - Require prediction of remaining cycles until failure
# 
# This is the "real-world" scenario where we predict RUL for operating engines.
# Loading forward unseen ground-truth machine trajectories (`test_FD001.txt`) intended for purely predictive and official scoring.

test_df = pd.read_csv("test_FD001.txt", sep="\s+", header=None)

test_df.columns = columns
test_df.head()

test_df.shape

# ### Extracting Out-Of-Sample Test Labels
# 
# We load the ground truth RUL values from `RUL_FD001.txt`. These represent the actual remaining useful life for each test engine, which we'll use to evaluate our predictions.
# 
# We load the ground truth RUL values from `RUL_FD001.txt`. These represent the actual remaining useful life for each test engine, which we'll use to evaluate our predictions.
# 
# **Note**: These are the final RUL values at the last recorded cycle for each engine.
# Accessing the strict `RUL_FD001.txt` repository to secure definitive final time-to-failure cycle targets accompanying each specific test turbine.

rul_test = pd.read_csv(
    "RUL_FD001.txt",
    header=None
)
rul_test.columns = ["RUL"]
rul_test.info()

test_df = test_df.drop(columns=drop_sensors)

test_df = test_df.sort_values(["engine_id", "cycle"])

# ### Rolling Statistical Mean Engineering
# 
# We create rolling average features (window size = 5 cycles) to smooth out sensor noise, capture underlying degradation trends, and provide stable representations of engine health. The rolling mean represents the local average behavior of each sensor over recent cycles.
# 
# We create rolling average features (window size = 5 cycles) to:
# - Smooth out sensor noise
# - Capture underlying degradation trends
# - Provide stable representations of engine health
# 
# The rolling mean represents the local average behavior of each sensor over recent cycles.
# Synthesizing simple 5-cycle wide rolling averages continuously over the premier sensors to smooth fluctuating sensor noise and unveil fundamental linear breakdown trends.

for sensor in top_sensors:
    test_df[f"{sensor}_rollmean"] = (
        test_df.groupby("engine_id")[sensor]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# ### Rolling Standard Deviation Engineering
# 
# We compute rolling standard deviation features to detect changes in sensor variability, identify periods of unusual engine behavior, and capture stress-related patterns that indicate approaching failure. High variance in certain sensors often signals imminent mechanical issues.
# 
# We compute rolling standard deviation features to:
# - Detect changes in sensor variability
# - Identify periods of unusual engine behavior
# - Capture stress-related patterns that indicate approaching failure
# 
# High variance in certain sensors often signals imminent mechanical issues.
# Creating subsequent 5-cycle rolling standard deviations on paramount sensors to empirically capture rapidly shifting local variance commonly symptomatic of imminent machine stress.

for sensor in top_sensors:
    test_df[f"{sensor}_rollstd"] = (
        test_df.groupby("engine_id")[sensor]
        .rolling(window=5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

# ### Calculating Trajectory Deltas
# 
# Delta features represent cycle-to-cycle changes in sensor values. These features capture the rate of degradation, short-term sensor fluctuations, and local trends in engine health.
# 
# Delta features represent cycle-to-cycle changes in sensor values:
# - **Delta = Current Value - Previous Value**
# 
# These features capture:
# - Rate of degradation
# - Short-term sensor fluctuations
# - Local trends in engine health
# 
# Positive or negative deltas can indicate accelerating or stabilizing degradation.
# Extracting precise cycle-over-cycle momentum changes (deltas) iteratively within each unit allowing the model to intrinsically assess the pace of functional decay.

for sensor in top_sensors:
    test_df[f"{sensor}_delta"] = (
        test_df.groupby("engine_id")[sensor]
        .diff()
    )

test_df = test_df.fillna(0)

test_last = test_df.groupby("engine_id").last().reset_index()

test_last.shape

X_test = test_last.drop(columns=["engine_id"])

X_test.shape

y_test_pred = pipe.predict(X_test)

# ### Instantiating the Corrective NASA Scoring Formula
# 
# NASA uses a specialized scoring function for prognostics competitions that penalizes late predictions heavily (predicting RUL too low) and penalizes early predictions lightly (predicting RUL too high). This asymmetric penalty reflects real-world concerns where late predictions can lead to unexpected failures while early predictions just mean earlier maintenance.
# 
# NASA uses a specialized scoring function for prognostics competitions that:
# - **Penalizes late predictions heavily** (predicting RUL too low): Exp(-d/13) - 1
# - **Penalizes early predictions lightly** (predicting RUL too high): Exp(d/10) - 1
# 
# This asymmetric penalty reflects real-world concerns:
# - Late predictions can lead to unexpected failures (dangerous)
# - Early predictions just mean earlier maintenance (safe)
# 
# A lower (closer to zero) score is better.
# Coding the distinctive asymmetric NASA metric. It mathematically incorporates an exponential punishment for late predictions (which actively endanger flights) against far smaller penalties for relatively preemptive early approximations.

def nasa_score(y_true, y_pred):
    score=0
    for true, pred in zip(y_true, y_pred):
        d = pred - true
        if d < 0:
            score += np.exp(-d/13) - 1
        else:
            score += np.exp(d/10) - 1
    return score

# ### Test Output Production & Benchmarking
# 
# We generate final predictions for the test set and evaluate using MAE, RMSE, and the NASA Score. These metrics provide a comprehensive assessment of model performance in real-world conditions.
# 
# We generate final predictions for the test set and evaluate using:
# 1. **MAE**: Average absolute prediction error
# 2. **RMSE**: Root mean squared error
# 3. **NASA Score**: The asymmetric scoring function
# 
# These metrics provide a comprehensive assessment of model performance in real-world conditions.
# Synthesizing our predictions externally and comparing our exact outcome arrays to true values under MAE, RMSE, and significantly, the NASA specialized scoring protocol.

mae_test = mean_absolute_error(rul_test["RUL"], y_test_pred)
rmse_test = np.sqrt(mean_squared_error(rul_test["RUL"], y_test_pred))
score_test = nasa_score(rul_test["RUL"].values, y_test_pred)

print("Test MAE:", mae_test)
print("Test RMSE:", rmse_test)
print("Test NASA Score: ", score_test)

errors = y_test_pred - rul_test["RUL"].values

# ### Evaluating Crucial Prediction Asymmetry
# 
# We analyze our prediction errors to understand early predictions (Predicted RUL > Actual RUL, safe errors) versus late predictions (Predicted RUL < Actual RUL, dangerous errors). Understanding this distribution helps assess the practical risk of our predictions in real maintenance scenarios.
# 
# We analyze our prediction errors to understand:
# - **Early predictions**: Predicted RUL > Actual RUL (safe errors)
# - **Late predictions**: Predicted RUL < Actual RUL (dangerous errors)
# 
# Understanding this distribution helps assess the practical risk of our predictions in real maintenance scenarios.
# Examining our error distribution split to empirically measure the comparative quantity of over-cautious "early" forecasts versus dangerous "late" predictions.

early_predcitions = (errors < 0).sum()
late_predictions = (errors > 0).sum()

print("Early predictions: ", early_predcitions)
print("Late predictions: ", late_predictions)

plt.figure(figsize=(8, 5))

sns.histplot(errors, bins=30)
plt.axvline(0, color="red", linestyle='--')

plt.xlabel("Prediction Error (Predicted RUL - True RUL)")
plt.title("Error Distribution")

plt.show()

