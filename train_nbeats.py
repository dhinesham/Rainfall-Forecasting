import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("blr_rain.csv", parse_dates=["date"])
df = df.drop(columns=["_id"])  # drop ID column

target_col = "Rain"
data = df[[target_col]].values  # shape (N,1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, window=30, horizon=7):
    X, y = [], []
    for i in range(len(data) - window - horizon):
        X.append(data[i:i+window])
        y.append(data[i+window:i+window+horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, 30, 7)

X_tensor = torch.tensor(X, dtype=torch.float32).squeeze(-1)  # (samples, 30)
y_tensor = torch.tensor(y, dtype=torch.float32).squeeze(-1)  # (samples, 7)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size=512, n_layers=4):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)

        
        self.backcast = nn.Linear(hidden_size, input_size)   
        self.forecast = nn.Linear(hidden_size, theta_size)   

    def forward(self, x):
        x = self.fc(x)
        backcast = self.backcast(x)
        forecast = self.forecast(x)
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, input_size=30, horizon=7, n_blocks=3, hidden_size=512):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, horizon, hidden_size) for _ in range(n_blocks)
        ])

    def forward(self, x):
        residual = x
        forecast_final = torch.zeros(x.size(0), 7, device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_final = forecast_final + forecast
        return forecast_final


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NBeats(input_size=30, horizon=7).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 60
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")


torch.save(model.state_dict(), "nbeats_blr_rain.pth")
joblib.dump(scaler, "scaler_blr.save")
print("✅ Model + scaler saved for blr_rain.csv")

