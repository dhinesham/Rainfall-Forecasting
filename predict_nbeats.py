import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size=512, n_layers=4):
        super().__init__()
        layers = []
        # First layer maps input_size -> hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        # Remaining layers are hidden_size -> hidden_size
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
model.load_state_dict(torch.load("Dnbeats_blr_rain.pth", map_location=device))
model.eval()

scaler = joblib.load("scaler_blr.save")

df = pd.read_csv("blr_rain.csv", parse_dates=["date"])
data = df[["Rain"]].values
data_scaled = scaler.transform(data)

last_seq = torch.tensor(data_scaled[-30:], dtype=torch.float32).unsqueeze(0).squeeze(-1).to(device)

with torch.no_grad():
    forecast_scaled = model(last_seq).cpu().numpy()

forecast = scaler.inverse_transform(forecast_scaled.reshape(-1,1)).flatten()

future_days = pd.date_range(df["date"].iloc[-1] + pd.Timedelta(days=1), periods=7)

plt.figure(figsize=(10,5))
plt.plot(future_days, forecast, marker="o", linestyle="-", color="orange", label="Forecast")

plt.title("7-Day Rainfall Forecast (Bengaluru)", fontsize=16, weight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Rainfall (mm)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

print("🌧 Next 7 days rainfall forecast (mm):")
for d, r in zip(future_days, forecast):
    print(f"{d.date()}: {r:.2f} mm")
