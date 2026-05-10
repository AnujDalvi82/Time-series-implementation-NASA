import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings 
warnings.filterwarnings("ignore")

# Define columns
op_cols = ["op_1", "op_2", "op_3"]
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
columns = ["engine_id", "cycle"] + op_cols + sensor_cols

# Load train data
train_df = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None)
train_df.columns = columns

max_cycle = train_df.groupby("engine_id")["cycle"].max()
train_df["max_cycle"] = train_df["engine_id"].map(max_cycle)
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]

# Dropping columns with little to no variance.
drop_sensors = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
train_df = train_df.drop(columns=drop_sensors)
train_df = train_df.sort_values(["engine_id", "cycle"])

# Feature engineering
top_sensors = ["sensor_11", "sensor_9", "sensor_4", "sensor_12", "sensor_14", "sensor_7", "sensor_15", "sensor_21", "sensor_2"]
for sensor in top_sensors:
    train_df[f"{sensor}_rollmean"] = train_df.groupby("engine_id")[sensor].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    train_df[f"{sensor}_rollstd"] = train_df.groupby("engine_id")[sensor].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True)
    train_df[f"{sensor}_delta"] = train_df.groupby("engine_id")[sensor].diff()

train_df = train_df.fillna(0)

# Define feature columns
feature_cols = [c for c in train_df.columns if c not in ["RUL", "engine_id", "max_cycle"]]

# Create train/valid split by engine ID to prevent data leakage
train_engines, valid_engines = train_test_split(train_df["engine_id"].unique(), test_size=0.2, random_state=42)
train_split_df = train_df[train_df["engine_id"].isin(train_engines)].copy()
valid_split_df = train_df[train_df["engine_id"].isin(valid_engines)].copy()

# Scale features
scaler = StandardScaler()
train_split_df[feature_cols] = scaler.fit_transform(train_split_df[feature_cols])
valid_split_df[feature_cols] = scaler.transform(valid_split_df[feature_cols])

# Sequence generator function
def create_sequences(df, feature_cols, target_col, seq_length=30):
    X_seq = []
    y_seq = [] 
    for engine_id, group in df.groupby("engine_id"):
        features = group[feature_cols].values
        targets = group[target_col].values
        if len(features) >= seq_length:
            for i in range(len(features) - seq_length + 1):
                X_seq.append(features[i : i + seq_length])
                y_seq.append(targets[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 30
print(f"Creating sequences of length {SEQ_LENGTH}...")
X_train_seq, y_train_seq = create_sequences(train_split_df, feature_cols, "RUL", SEQ_LENGTH)
X_valid_seq, y_valid_seq = create_sequences(valid_split_df, feature_cols, "RUL", SEQ_LENGTH)

print("X_train shape:", X_train_seq.shape)

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

X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
X_valid_t = torch.tensor(X_valid_seq, dtype=torch.float32)
y_valid_t = torch.tensor(y_valid_seq, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

model = LSTMModel(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
print("Training True Sequential LSTM Model...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid_t)
        val_loss = criterion(val_outputs, y_valid_t).item()
        val_mae = mean_absolute_error(y_valid_t.numpy(), val_outputs.numpy())
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.2f} - Val Loss: {val_loss:.2f} - Val MAE: {val_mae:.2f}")

print("Evaluating on Test Set...")
test_df = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None)
test_df.columns = columns
test_df = test_df.drop(columns=drop_sensors)
test_df = test_df.sort_values(["engine_id", "cycle"])

for sensor in top_sensors:
    test_df[f"{sensor}_rollmean"] = test_df.groupby("engine_id")[sensor].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    test_df[f"{sensor}_rollstd"] = test_df.groupby("engine_id")[sensor].rolling(window=5, min_periods=1).std().reset_index(level=0, drop=True)
    test_df[f"{sensor}_delta"] = test_df.groupby("engine_id")[sensor].diff()

test_df = test_df.fillna(0)

# Scale test set
test_df[feature_cols] = scaler.transform(test_df[feature_cols].values)

# Prepare test sequences
X_test_seq = []
for engine_id, group in test_df.groupby("engine_id"):
    features = group[feature_cols].values
    if len(features) >= SEQ_LENGTH:
        X_test_seq.append(features[-SEQ_LENGTH:])
    else:
        pad = np.zeros((SEQ_LENGTH - len(features), len(feature_cols)))
        padded = np.vstack([pad, features])
        X_test_seq.append(padded)
        
X_test_seq = np.array(X_test_seq)
X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_test_pred_t = model(X_test_t)
y_test_pred = y_test_pred_t.numpy().flatten()

rul_test = pd.read_csv("RUL_FD001.txt", header=None)
rul_test.columns = ["RUL"]

def nasa_score(y_true, y_pred):
    score=0
    for true, pred in zip(y_true, y_pred):
        d = pred - true
        if d < 0:
            score += np.exp(-d/13) - 1
        else:
            score += np.exp(d/10) - 1
    return score

mae_test = mean_absolute_error(rul_test["RUL"], y_test_pred)
rmse_test = np.sqrt(mean_squared_error(rul_test["RUL"], y_test_pred))
score_test = nasa_score(rul_test["RUL"].values, y_test_pred)

print(f"Test MAE: {mae_test:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test NASA Score: {score_test:.2f}")

print("Saving model and scaler...")
torch.save(model.state_dict(), "lstm_model.pth")
joblib.dump(scaler, "scaler.pkl")
print("Saved successfully!")
