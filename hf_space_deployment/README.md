---
title: AeroPulse Predictive Maintenance
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AeroPulse RUL Prediction Dashboard

This is a FastAPI application that serves a PyTorch LSTM model predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset.

## How it works

1. **Backend**: A FastAPI server running via Uvicorn.
2. **Model**: A trained PyTorch LSTM model (`lstm_model.pth`) and a Scikit-Learn scaler (`scaler.pkl`).
3. **Data**: The application uses `test_FD001.txt` as a simulated data stream to provide predictions.
4. **Frontend**: An interactive web dashboard (`index.html`) using Chart.js to visualize the sensor readings and RUL predictions.

## Local Deployment

To run this locally:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then visit `http://127.0.0.1:8000` in your browser.
