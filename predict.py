import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# CONFIG

DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "model", "lstm_model.pth")
INPUT_CSV = os.path.join(DATA_DIR, "Indian_Weather_Data.csv")  
OUTPUT_CSV = os.path.join(DATA_DIR, "predictions.csv")

FEATURES = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
HIDDEN_SIZE = 64
SEQ_LEN = 1  


# MODEL

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL

model = SimpleLSTM(len(FEATURES), HIDDEN_SIZE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# LOAD AND PREPROCESS DATA

df = pd.read_csv(INPUT_CSV)


date_col = None
for col in df.columns:
    if "date" in col.lower():
        date_col = col
        break

if date_col:
    df['date'] = pd.to_datetime(df[date_col], utc=True)
else:
    df['date'] = pd.date_range(start="2000-01-01", periods=len(df), freq='D')


available_features = [f for f in FEATURES if f in df.columns]
if len(available_features) == 0:
    raise ValueError(f"No matching feature columns found. Available columns: {df.columns.tolist()}")

X = df[available_features].values.astype(np.float32)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)

# MAKE PREDICTIONS

with torch.no_grad():
    preds = model(X_tensor)
    preds = preds.cpu().numpy().flatten()


# SAVE PREDICTIONS
df['Predicted_Temperature'] = preds
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to {OUTPUT_CSV}")
