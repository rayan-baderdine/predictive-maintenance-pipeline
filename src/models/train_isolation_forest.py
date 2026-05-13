import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

def load_features():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM features ORDER BY window_start", engine)
    print(f"Loaded {len(df)} feature windows.")
    return df

def train():
    df = load_features()
    X = df[FEATURE_COLS].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = 0.1  # assume ~10% of windows are anomalous
    
    mlflow.set_experiment("predictive-maintenance")
    
    with mlflow.start_run(run_name="isolation-forest"):
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        model.fit(X_scaled)
        
        scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)  # 1=normal, -1=anomaly
        n_anomalies = (predictions == -1).sum()
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metric("n_anomalies_detected", int(n_anomalies))
        mlflow.log_metric("anomaly_rate", float(n_anomalies / len(predictions)))
        mlflow.sklearn.log_model(model, "isolation_forest_model")
        
        with open(f"{MODEL_DIR}/isolation_forest.pkl", "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(f"{MODEL_DIR}/isolation_forest.pkl")

        with open(f"{MODEL_DIR}/scaler_if.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(f"{MODEL_DIR}/scaler_if.pkl")

        with open(f"{MODEL_DIR}/scaler_if.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(f"{MODEL_DIR}/scaler_if.pkl")
        
        print(f"Anomalies detected: {n_anomalies}/{len(predictions)}")
        print(f"Anomaly scores range: {scores.min():.4f} to {scores.max():.4f}")
        
        df["if_score"] = scores
        df["if_label"] = predictions
        df[["window_start", "window_end", "if_score", "if_label"]].to_csv(
            f"{MODEL_DIR}/if_results.csv", index=False
        )
        mlflow.log_artifact(f"{MODEL_DIR}/if_results.csv")
        print("Run logged to MLflow.")

if __name__ == "__main__":
    train()