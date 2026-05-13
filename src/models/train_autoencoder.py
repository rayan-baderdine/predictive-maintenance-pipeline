import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import pickle, os

DB_URL = "postgresql://postgres:0000@localhost:5432/bearing_db"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "ch1_rms", "ch1_std", "ch1_kurtosis", "ch1_crest",
    "ch2_rms", "ch2_std", "ch2_kurtosis", "ch2_crest",
    "ch3_rms", "ch3_std", "ch3_kurtosis", "ch3_crest",
    "ch4_rms", "ch4_std", "ch4_kurtosis", "ch4_crest",
] + [f"fft_bin_{i}" for i in range(1, 11)]

INPUT_DIM = len(FEATURE_COLS)
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_features():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM features ORDER BY window_start", engine)
    print(f"Loaded {len(df)} feature windows.")
    return df

def train():
    df = load_features()
    X = df[FEATURE_COLS].values.astype(np.float32)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    tensor = torch.tensor(X_scaled)
    loader = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True)
    
    model = Autoencoder(INPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    mlflow.set_experiment("predictive-maintenance")
    
    with mlflow.start_run(run_name="autoencoder"):
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("latent_dim", 4)
        mlflow.log_param("input_dim", INPUT_DIM)
        
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.6f}")
        
        # Reconstruction errors = anomaly scores
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled)
            recon = model(X_tensor)
            errors = ((X_tensor - recon) ** 2).mean(dim=1).numpy()
        
        threshold = np.percentile(errors, 90)
        anomalies = (errors > threshold).sum()
        
        mlflow.log_metric("recon_error_mean", float(errors.mean()))
        mlflow.log_metric("recon_error_max", float(errors.max()))
        mlflow.log_metric("threshold_p90", float(threshold))
        mlflow.log_metric("n_anomalies_detected", int(anomalies))
        mlflow.pytorch.log_model(model, "autoencoder_model")
        
        torch.save(model.state_dict(), f"{MODEL_DIR}/autoencoder.pt")
        with open(f"{MODEL_DIR}/scaler_ae.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(f"{MODEL_DIR}/autoencoder.pt")
        mlflow.log_artifact(f"{MODEL_DIR}/scaler_ae.pkl")
        
        df["ae_error"] = errors
        df["ae_label"] = (errors > threshold).astype(int)
        df[["window_start", "window_end", "ae_error", "ae_label"]].to_csv(
            f"{MODEL_DIR}/ae_results.csv", index=False
        )
        mlflow.log_artifact(f"{MODEL_DIR}/ae_results.csv")
        print(f"Anomalies detected: {anomalies}/{len(errors)} (threshold={threshold:.6f})")
        print("Run logged to MLflow.")

if __name__ == "__main__":
    train()