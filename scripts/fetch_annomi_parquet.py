import urllib.request
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
parquet_path = RAW_DIR / "annomi.parquet"

url = "https://huggingface.co/datasets/to-be/annomi-motivational-interviewing-therapy-conversations/resolve/main/annomi.parquet"
print(f"Downloading real AnnoMI dataset from {url}...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = resp.read()
    with open(parquet_path, "wb") as f:
        f.write(data)

df = pd.read_parquet(parquet_path)
print(f"[SUCCESS] Loaded AnnoMI parquet! Total Rows: {len(df)}, Transcripts: {df['transcript_id'].nunique()}")

csv_path = RAW_DIR / "AnnoMI-simple.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"[SUCCESS] Saved to {csv_path}!")
