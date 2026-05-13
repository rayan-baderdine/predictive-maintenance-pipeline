import numpy as np
import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Predictive Maintenance API")

FEATURE_COLS = [
    "ch1_rms", "ch1_std", "ch1_kurtosis", "ch1_crest",
    "ch2_rms", "ch2_std", "ch2_kurtosis", "ch2_crest",
    "ch3_rms", "ch3_std", "ch3_kurtosis", "ch3_crest",
    "ch4_rms", "ch4_std", "ch4_kurtosis", "ch4_crest",
] + [f"fft_bin_{i}" for i in range(1, 11)]

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Load models at startup
with open("models/scaler_if.pkl", "rb") as f:
    scaler_if = pickle.load(f)
with open("models/scaler_ae.pkl", "rb") as f:
    scaler_ae = pickle.load(f)

with open("models/isolation_forest.pkl", "rb") as f:
    if_model = pickle.load(f)

ae_model = Autoencoder(len(FEATURE_COLS))
ae_model.load_state_dict(torch.load("models/autoencoder.pt", map_location="cpu"))
ae_model.eval()

AE_THRESHOLD = 0.01  # adjust after seeing your ae_results.csv

class SensorReading(BaseModel):
    features: List[float]  # must be len(FEATURE_COLS) = 26 values

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(reading: SensorReading):
    x = np.array(reading.features).reshape(1, -1)

    # Isolation Forest
    x_if = scaler_if.transform(x)
    if_score = float(if_model.decision_function(x_if)[0])
    if_label = "anomaly" if if_model.predict(x_if)[0] == -1 else "normal"

    # Autoencoder
    x_ae = scaler_ae.transform(x).astype(np.float32)
    with torch.no_grad():
        recon = ae_model(torch.tensor(x_ae))
        ae_error = float(((torch.tensor(x_ae) - recon) ** 2).mean())
    ae_label = "anomaly" if ae_error > AE_THRESHOLD else "normal"

    return {
        "isolation_forest": {"score": if_score, "label": if_label},
        "autoencoder": {"reconstruction_error": ae_error, "label": ae_label},
        "consensus": "anomaly" if if_label == "anomaly" or ae_label == "anomaly" else "normal"
    }