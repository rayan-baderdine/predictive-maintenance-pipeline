import os
import json
import time
import pandas as pd

DATA_PATH = "data/raw/bearing"

def load_bearing_files(path):
    files = sorted(os.listdir(path))
    return files

def read_file(filepath):
    df = pd.read_csv(filepath, sep="\t", header=None)
    n_cols = df.shape[1]
    df.columns = [f"ch{i+1}" for i in range(n_cols)]
    return df
def simulate_stream(delay=0.05):
    files = load_bearing_files(DATA_PATH)
    print(f"Found {len(files)} files. Starting stream...")
    for i, fname in enumerate(files):
        fpath = os.path.join(DATA_PATH, fname)
        df = read_file(fpath)
        row = df.mean().to_dict()
        row["timestamp"] = fname
        row["file_index"] = i
        message = json.dumps(row)
        print(f"[{i+1}/{len(files)}] {message}")
        time.sleep(delay)

if __name__ == "__main__":
    simulate_stream()
