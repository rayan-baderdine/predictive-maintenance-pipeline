import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

DB_URL = "postgresql://postgres:0000@localhost:5432/bearing_db"

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance Dashboard")
st.caption("NASA Bearing Dataset — Anomaly Detection")

@st.cache_data
def load_data():
    features = pd.read_csv("models/if_results.csv")
    ae = pd.read_csv("models/ae_results.csv")
    df = features.merge(ae, on=["window_start", "window_end"])
    return df

df = load_data()

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("Controls")
channel = st.sidebar.selectbox("Channel", ["ch1", "ch2", "ch3", "ch4"])
ae_threshold = st.sidebar.slider(
    "Autoencoder threshold",
    float(df["ae_error"].min()),
    float(df["ae_error"].max()),
    float(df["ae_error"].quantile(0.85))
)
df["ae_label_live"] = (df["ae_error"] > ae_threshold).astype(int)

# ── KPI row ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total windows", len(df))
col2.metric("IF anomalies", int((df["if_label"] == -1).sum()))
col3.metric("AE anomalies", int(df["ae_label_live"].sum()))
consensus = int(((df["if_label"] == -1) | (df["ae_label_live"] == 1)).sum())
col4.metric("Consensus anomalies", consensus)

st.divider()

# ── RMS over time ─────────────────────────────────────────
st.subheader(f"RMS — {channel}")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(df["window_start"], df[f"{channel}_rms"], label="RMS")
anomaly_idx = df[df["ae_label_live"] == 1]["window_start"]
ax.vlines(anomaly_idx, ymin=df[f"{channel}_rms"].min(),
          ymax=df[f"{channel}_rms"].max(),
          color="red", alpha=0.4, label="AE anomaly")
ax.legend()
ax.set_xlabel("Window")
ax.set_ylabel("RMS")
st.pyplot(fig)

# ── Kurtosis over time ────────────────────────────────────
st.subheader(f"Kurtosis — {channel}")
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(df["window_start"], df[f"{channel}_kurtosis"], color="darkorange")
ax2.axhline(3.0, color="gray", linestyle="--", label="Normal (3.0)")
ax2.legend()
ax2.set_xlabel("Window")
ax2.set_ylabel("Kurtosis")
st.pyplot(fig2)

# ── Autoencoder reconstruction error ─────────────────────
st.subheader("Autoencoder reconstruction error")
fig3, ax3 = plt.subplots(figsize=(12, 3))
ax3.plot(df["window_start"], df["ae_error"], color="purple")
ax3.axhline(ae_threshold, color="red", linestyle="--",
            label=f"Threshold ({ae_threshold:.4f})")
ax3.legend()
ax3.set_xlabel("Window")
ax3.set_ylabel("MSE")
st.pyplot(fig3)

# ── Raw feature table ─────────────────────────────────────
st.subheader("Feature table")
st.dataframe(df[["window_start", "window_end",
                  f"{channel}_rms", f"{channel}_kurtosis",
                  "ae_error", "ae_label_live", "if_label"]],
             use_container_width=True)
