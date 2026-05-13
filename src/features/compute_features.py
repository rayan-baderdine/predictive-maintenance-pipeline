import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.fft import fft
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:0000@localhost:5432/bearing_db"
WINDOW_SIZE = 50
N_FFT_BINS = 10

def get_engine():
    return create_engine(DB_URL)

def create_features_table(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS features (
                id SERIAL PRIMARY KEY,
                window_start INTEGER,
                window_end INTEGER,
                ch1_rms FLOAT, ch1_mean FLOAT, ch1_std FLOAT,
                ch1_kurtosis FLOAT, ch1_crest FLOAT,
                ch2_rms FLOAT, ch2_mean FLOAT, ch2_std FLOAT,
                ch2_kurtosis FLOAT, ch2_crest FLOAT,
                ch3_rms FLOAT, ch3_mean FLOAT, ch3_std FLOAT,
                ch3_kurtosis FLOAT, ch3_crest FLOAT,
                ch4_rms FLOAT, ch4_mean FLOAT, ch4_std FLOAT,
                ch4_kurtosis FLOAT, ch4_crest FLOAT,
                fft_bin_1 FLOAT, fft_bin_2 FLOAT, fft_bin_3 FLOAT,
                fft_bin_4 FLOAT, fft_bin_5 FLOAT, fft_bin_6 FLOAT,
                fft_bin_7 FLOAT, fft_bin_8 FLOAT, fft_bin_9 FLOAT,
                fft_bin_10 FLOAT
            )
        """))
        conn.commit()
    print("Features table ready.")

def compute_time_features(series):
    rms = np.sqrt(np.mean(series**2))
    mean = np.mean(series)
    std = np.std(series)
    kurt = kurtosis(series)
    peak = np.max(np.abs(series))
    crest = peak / rms if rms > 0 else 0
    return rms, mean, std, kurt, crest

def compute_fft_features(series, n_bins=10):
    spectrum = np.abs(fft(series))
    half = spectrum[:len(spectrum)//2]
    # Top n_bins frequency magnitudes (sorted by magnitude descending)
    top_indices = np.argsort(half)[-n_bins:][::-1]
    top_mags = half[top_indices]
    return top_mags

def load_and_compute():
    engine = get_engine()
    create_features_table(engine)

    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM raw_readings ORDER BY file_index", conn)

    print(f"Loaded {len(df)} rows from raw_readings.")
    print(f"Processing in windows of {WINDOW_SIZE}...")

    records = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]
        record = {
            "window_start": int(window["file_index"].iloc[0]),
            "window_end": int(window["file_index"].iloc[-1]),
        }
        for ch in ["ch1", "ch2", "ch3", "ch4"]:
            series = window[ch].values.astype(float)
            rms, mean, std, kurt, crest = compute_time_features(series)
            record[f"{ch}_rms"] = rms
            record[f"{ch}_mean"] = mean
            record[f"{ch}_std"] = std
            record[f"{ch}_kurtosis"] = kurt
            record[f"{ch}_crest"] = crest

        # FFT on channel 1 (primary bearing channel)
        fft_bins = compute_fft_features(window["ch1"].values.astype(float), N_FFT_BINS)
        for i, mag in enumerate(fft_bins):
            record[f"fft_bin_{i+1}"] = float(mag)

        records.append(record)

    print(f"Computed {len(records)} feature windows. Inserting into database...")

    with engine.connect() as conn:
        for i, rec in enumerate(records):
            conn.execute(text("""
                INSERT INTO features (
                    window_start, window_end,
                    ch1_rms, ch1_mean, ch1_std, ch1_kurtosis, ch1_crest,
                    ch2_rms, ch2_mean, ch2_std, ch2_kurtosis, ch2_crest,
                    ch3_rms, ch3_mean, ch3_std, ch3_kurtosis, ch3_crest,
                    ch4_rms, ch4_mean, ch4_std, ch4_kurtosis, ch4_crest,
                    fft_bin_1, fft_bin_2, fft_bin_3, fft_bin_4, fft_bin_5,
                    fft_bin_6, fft_bin_7, fft_bin_8, fft_bin_9, fft_bin_10
                ) VALUES (
                    :window_start, :window_end,
                    :ch1_rms, :ch1_mean, :ch1_std, :ch1_kurtosis, :ch1_crest,
                    :ch2_rms, :ch2_mean, :ch2_std, :ch2_kurtosis, :ch2_crest,
                    :ch3_rms, :ch3_mean, :ch3_std, :ch3_kurtosis, :ch3_crest,
                    :ch4_rms, :ch4_mean, :ch4_std, :ch4_kurtosis, :ch4_crest,
                    :fft_bin_1, :fft_bin_2, :fft_bin_3, :fft_bin_4, :fft_bin_5,
                    :fft_bin_6, :fft_bin_7, :fft_bin_8, :fft_bin_9, :fft_bin_10
                )
            """), rec)
        conn.commit()

    print(f"Done. {len(records)} feature rows stored in 'features' table.")

if __name__ == "__main__":
    load_and_compute()