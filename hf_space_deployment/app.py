from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="AeroPulse RUL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = None
scaler = None
test_df_processed = None
feature_cols = None
SEQ_LENGTH = 30

@app.on_event("startup")
def load_artifacts():
    global model, scaler, test_df_processed, feature_cols
    try:
        print("Loading model and scaler...")
        scaler = joblib.load("scaler.pkl")
        model = LSTMModel(input_size=45, hidden_size=64, num_layers=2, output_size=1)
        model.load_state_dict(torch.load("lstm_model.pth"))
        model.eval()
        
        print("Loading and preprocessing test_FD001.txt...")
        op_cols = ["op_1", "op_2", "op_3"]
        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
        columns = ["engine_id", "cycle"] + op_cols + sensor_cols
        
        test_df = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None)
        test_df.columns = columns
        
        drop_sensors = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
        test_df = test_df.drop(columns=drop_sensors)
        test_df = test_df.sort_values(["engine_id", "cycle"])
        
        top_sensors = ["sensor_11", "sensor_9", "sensor_4", "sensor_12", "sensor_14", "sensor_7", "sensor_15", "sensor_21", "sensor_2"]
        
        for sensor in top_sensors:
            test_df[f"{sensor}_rollmean"] = test_df.groupby("engine_id")[sensor].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
            test_df[f"{sensor}_rollstd"] = test_df.groupby("engine_id")[sensor].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True)
            test_df[f"{sensor}_delta"] = test_df.groupby("engine_id")[sensor].diff()
            
        test_df = test_df.fillna(0)
        feature_cols = [c for c in test_df.columns if c not in ["RUL", "engine_id", "max_cycle"]]
        
        # Scale test set
        test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)
        
        test_df_processed = test_df
        print("Successfully loaded artifacts and preprocessed test dataset!")
    except Exception as e:
        print("Startup Error:", str(e))

@app.get("/engine/{engine_id}/history")
def get_engine_history(engine_id: int):
    if test_df_processed is None:
        raise HTTPException(status_code=500, detail="Data not loaded properly on server startup.")
        
    engine_data = test_df_processed[(test_df_processed["engine_id"] == engine_id)]
    if engine_data.empty:
        raise HTTPException(status_code=404, detail="Engine not found in dataset.")
        
    features = engine_data[feature_cols].values
    
    if len(features) >= SEQ_LENGTH:
        X_seq = features[-SEQ_LENGTH:]
    else:
        pad = np.zeros((SEQ_LENGTH - len(features), len(feature_cols)))
        X_seq = np.vstack([pad, features])
        
    X_test_t = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0) # shape (1, 30, 45)
    
    with torch.no_grad():
        pred = model(X_test_t)
        
    # Get last 20 cycles for the charts (unscaled versions preferably, but we scaled them inplace)
    # Let's get them from unscaled inverse transform or just return the scaled for now since 
    # the frontend draws the shape. For accuracy, let's reverse transform.
    history = engine_data.tail(20)
    unscaled_hist = scaler.inverse_transform(history[feature_cols].values)
    history_df = pd.DataFrame(unscaled_hist, columns=feature_cols)
    history_df["cycle"] = history["cycle"].values
    
    cycles = history_df["cycle"].tolist()
    sensor_11 = history_df["sensor_11"].tolist()
    sensor_14 = history_df["sensor_14"].tolist()
    
    return {
        "engine_id": engine_id,
        "current_cycle": int(cycles[-1]),
        "predicted_rul": float(pred.item()),
        "history": {
            "cycles": cycles,
            "sensor_11": sensor_11,
            "sensor_14": sensor_14
        }
    }

@app.get("/fleet")
def get_fleet_status():
    if test_df_processed is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
    
    X_test_seq = []
    engine_ids = []
    cycles = []
    
    for engine_id, group in test_df_processed.groupby("engine_id"):
        engine_ids.append(engine_id)
        cycles.append(group["cycle"].iloc[-1])
        features = group[feature_cols].values
        if len(features) >= SEQ_LENGTH:
            X_test_seq.append(features[-SEQ_LENGTH:])
        else:
            pad = np.zeros((SEQ_LENGTH - len(features), len(feature_cols)))
            X_test_seq.append(np.vstack([pad, features]))
            
    X_test_t = torch.tensor(np.array(X_test_seq), dtype=torch.float32)
    
    with torch.no_grad():
        preds = model(X_test_t).flatten().numpy()
        
    fleet = []
    for i in range(len(engine_ids)):
        fleet.append({
            "id": int(engine_ids[i]),
            "cycles": int(cycles[i]),
            "rul": float(preds[i])
        })
        
    # Sort fleet by RUL ascending
    fleet.sort(key=lambda x: x["rul"])
    return fleet

@app.get("/health")
def health_check():
    return {"status": "Healthy"}

@app.get("/engine_bg.png")
def serve_bg():
    return FileResponse("engine_bg.png")

@app.get("/")
def serve_ui():
    return FileResponse("index.html")
