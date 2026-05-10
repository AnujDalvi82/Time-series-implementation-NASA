# ✈️ NASA Turbofan Engine Remaining Useful Life Prediction

<p align="center">
🔧 Predictive Maintenance | 📊 ML & Data Science | 🤖 Random Forest
</p>

---

## 📌 Project Overview

### Background
Aircraft engines produce large volumes of sensor data during operation. These measurements contain valuable information about engine health and degradation patterns over time. As engines operate, they accumulate wear and tear, leading to gradual performance degradation.

### Problem Statement
This project focuses on predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA CMAPSS dataset. The goal is to estimate how many operational cycles remain before an engine reaches failure.

### Why This Matters
Accurate prediction of RUL enables:
- ✅ **Predictive Maintenance** - Schedule maintenance before failures occur
- ✅ **Cost Reduction** - Optimize maintenance intervals and reduce unplanned downtime  
- ✅ **Safety Improvement** - Prevent in-flight failures through early warning systems
- ✅ **Operational Efficiency** - Better planning for engine replacements and resource allocation

---

## 📂 Dataset

### About the Dataset
The dataset used in this project is the **NASA CMAPSS Turbofan Engine Degradation Simulation Dataset**. This is a benchmark dataset widely used in prognostics and health management research.

Each engine unit begins with different levels of wear and gradually degrades over time. Sensor readings and operational settings are recorded at every operational cycle until the engine fails.

### Dataset Files

| File | Description |
|:-----|:------------|
| `train_FD001.txt` | Training engine trajectories until failure |
| `test_FD001.txt` | Test engine trajectories that stop before failure |
| `RUL_FD001.txt` | True remaining useful life values for the test engines |

### Dataset Characteristics

| Feature | Description |
|:--------|:------------|
| 🛠 **Engine Units** | 100+ engines in training set |
| ⏱ **Data Type** | Cycle-based time series |
| ⚙ **Settings** | 3 operational conditions |
| 📡 **Sensors** | 21 comprehensive measurements |
| 📉 **Degradation** | Progressive until failure |
| 📊 **Lifespans** | Variable across engines |

---

## 🔬 Project Workflow

### Step 1: Data Preparation
- Load CMAPSS dataset files
- Assign descriptive column names to raw data
- Merge cycle information for each engine
- Compute Remaining Useful Life target values

### Step 2: Exploratory Data Analysis  
- Visualize sensor behavior across operational cycles
- Identify sensors that show degradation patterns
- Analyze trends related to engine wear
- Examine correlation between sensors and RUL
- Understand the distribution of engine lifespans

### Step 3: Data Preprocessing
- Remove sensors with low variance (no information content)
- Handle missing values
- Normalize/scale features

### Step 4: Feature Engineering
- **Rolling Mean Features**: Smooth sensor data over 5-cycle windows
- **Rolling Standard Deviation**: Capture local variability
- **Delta Features**: Calculate cycle-to-cycle changes
- **Temporal Degradation Indicators**: Track trends over time

### Step 5: Model Development
- **Algorithm**: Random Forest Regression
- **Why Random Forest?**:
  - Handles non-linear relationships
  - Provides feature importance rankings
  - Resistant to overfitting
  - Works well with multiple features

### Step 6: Model Evaluation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- NASA Scoring Function (asymmetric evaluation)
- Prediction asymmetry analysis

---

## 🗂 Repository Structure

```
Time-Series-Implementation-NASA/
│
├── CMAPSS_TimeSeries_Evaluation.ipynb   # Main analysis notebook
├── train_FD001.txt                       # Training data
├── test_FD001.txt                         # Test data
├── RUL_FD001.txt                          # Ground truth RUL values
├── Damage Propagation Modeling.pdf       # Reference documentation
└── README.md                              # This file
```

---

## 🛠 Technologies Used

- 🐍 **Python** - Programming language
- 📊 **Pandas** - Data manipulation and analysis
- 🔢 **NumPy** - Numerical computations
- 📉 **Matplotlib** - Static visualization plotting
- 📈 **Seaborn** - Statistical data visualization
- 🤖 **Scikit Learn** - Machine learning algorithms

---

## 📊 Results

### Model Performance
The machine learning model learns degradation signals from multiple sensors and predicts the remaining operational cycles of turbofan engines.

### Key Findings
- **Top Predictive Sensors**: Identified sensors with highest feature importance
- **Feature Engineering Impact**: Rolling statistics improved model accuracy  
- **Error Analysis**: Examined prediction bias (early vs late predictions)

### Business Impact
This demonstrates how machine learning can transform raw sensor telemetry into actionable maintenance insights for aerospace systems.

---

## 🚀 Applications

- ✈️ **Predictive Maintenance in Aviation** - Airline engine health monitoring
- 🏭 **Industrial Equipment Monitoring** - Factory machinery lifecycle management
- 📊 **Reliability Engineering** - System reliability analysis
- ⏳ **Time Series Machine Learning** - General time-based prediction problems
- 🔧 **Condition-Based Maintenance** - Real-time equipment health assessment

---

## 🔭 Future Improvements

### Short-term Enhancements
- ☐ Hyperparameter tuning for Random Forest
- ☐ Feature selection based on importance
- ☐ Cross-validation for robust evaluation

### Long-term Improvements
- ☐ Advanced survival analysis approaches
- ☐ Gradient boosting models (XGBoost, LightGBM)
- ☐ Deep learning-based temporal models (LSTM, GRU)
- ☐ Sensor importance analysis
- ☐ Ensemble methods
- ☐ Real-time prediction deployment

---

## 📚 References

- **NASA Prognostics Center of Excellence** - CMAPSS Turbofan Engine Degradation Simulation Dataset
- **PHM Society Conference Papers** - Prognostics and health management research
- **IEEE** - "Prognostics and Health Management: A Review"

---

## 👨‍💻 Authors

**Arjun Vinod Patil & Anuj Dalvi**

*Machine Learning and Data Science Enthusiasts*

---

## 📝 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

Thanks to NASA for providing the CMAPSS dataset and to the open-source community for the tools used in this project.

