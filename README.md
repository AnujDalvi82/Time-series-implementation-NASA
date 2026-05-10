# ✈️ NASA Turbofan Engine Remaining Useful Life Prediction

<p align="center">
🔧 Predictive Maintenance | 📊 Deep Learning (LSTM) | 🤖 FastAPI Dashboard
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

## 🚀 Live Production Deployment

This project has been fully containerized and deployed as an interactive predictive maintenance dashboard.

- **Hugging Face Space**: [NASA CMAPSS RUL Dashboard](https://huggingface.co/spaces/anujjj321/CMAPSS_Time_Series_evalutation)

---

## 📂 Dataset

### About the Dataset
The dataset used in this project is the **NASA CMAPSS Turbofan Engine Degradation Simulation Dataset**. This is a benchmark dataset widely used in prognostics and health management research.

Each engine unit begins with different levels of wear and gradually degrades over time. Sensor readings and operational settings are recorded at every operational cycle until the engine fails.

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

## 🔬 Project Architecture & Workflow

### 1. Data Processing
- Extracted and normalized raw sensor telemetry.
- Engineered sequential rolling windows to capture long-term temporal degradation.
- Handled massive datasets using standard scaling and min-max normalization.

### 2. Deep Learning (LSTM)
- Transitioned from baseline Random Forest models to a state-of-the-art **Long Short-Term Memory (LSTM)** neural network.
- Why LSTM? LSTMs natively retain memory of previous operational cycles, making them the industry standard for time-series degradation tracking.

### 3. Production Backend (FastAPI)
- Developed a high-performance REST API using **FastAPI**.
- The API dynamically serves real-time inference data and operational fleet status to the front end.

### 4. Interactive Dashboard
- Created a beautiful, responsive, glassmorphism-inspired web UI to act as the primary operational dashboard.
- Features real-time animated alerts and clear visual indicators for engine health status.

---

## 🗂 Repository Structure

```text
Time-Series-Implementation-NASA/
│
├── CMAPSS_TimeSeries_Evaluation.py   # Core analytics and pipeline
├── cmapss_lstm.py                    # LSTM model definition and training script
├── app.py                            # FastAPI production backend
├── index.html                        # Front-end predictive maintenance dashboard
├── lstm_model.pth                    # Saved PyTorch weights
├── scaler.pkl                        # Data normalizer
├── requirements.txt                  # Deployment dependencies
├── Dockerfile                        # Containerization setup
└── README.md                         # This file
```

---

## 🛠 Technologies Used

- 🐍 **Python** - Core language
- 🔥 **PyTorch** - Deep Learning framework (LSTM)
- ⚡ **FastAPI** - High-speed backend API
- 💻 **HTML/CSS/JS** - Frontend dashboard (Vanilla, Glassmorphism design)
- 🐳 **Docker** - Containerization
- 🤗 **Hugging Face Spaces** - Cloud deployment

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
