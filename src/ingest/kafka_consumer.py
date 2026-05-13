import os
import json
import time
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:0000@localhost:5432/bearing_db"
DATA_PATH = "data/raw/bearing"

def get_engine():
    engine = create_engine(DB_URL)
    return engine

def create_table(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_readings (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(50),
                file_index INTEGER,
                ch1 FLOAT, ch2 FLOAT, ch3 FLOAT, ch4 FLOAT
            )
        """))
        conn.commit()
    print("Table ready.")

def load_and_store():
    engine = get_engine()
    create_table(engine)
    files = sorted(os.listdir(DATA_PATH))
    print(f"Found {len(files)} files. Storing to database...")
    for i, fname in enumerate(files):
        fpath = os.path.join(DATA_PATH, fname)
        df = pd.read_csv(fpath, sep="\t", header=None)
        n_cols = df.shape[1]
        df.columns = [f"ch{j+1}" for j in range(n_cols)]
        row = df.mean().to_dict()
        row["timestamp"] = fname
        row["file_index"] = i
        record = {
            "timestamp": row["timestamp"],
            "file_index": row["file_index"],
            "ch1": row.get("ch1", 0),
            "ch2": row.get("ch2", 0),
            "ch3": row.get("ch3", 0),
            "ch4": row.get("ch4", 0)
        }
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO raw_readings (timestamp, file_index, ch1, ch2, ch3, ch4)
                VALUES (:timestamp, :file_index, :ch1, :ch2, :ch3, :ch4)
            """), record)
            conn.commit()
        if i % 100 == 0:
            print(f"[{i+1}/{len(files)}] Stored {fname}")
    print("All done. Database populated.")

if __name__ == "__main__":
    load_and_store()