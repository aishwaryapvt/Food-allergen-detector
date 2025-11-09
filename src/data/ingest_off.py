# src/data/ingest_off.py
import os
import pandas as pd

OUT_DIR = "data_raw"
CSV_PATH = os.path.join(OUT_DIR, "off_subset.csv")

def verify_data():
    """Check if offline dataset exists and show a preview."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at {CSV_PATH}. Please make sure off_subset.csv exists.")
    
    print(f"✅ Found offline dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"\n📊 Dataset loaded successfully! Rows: {len(df)}, Columns: {list(df.columns)}")
    print("\n🧾 Sample data:")
    print(df.head())

if __name__ == "__main__":
    verify_data()
