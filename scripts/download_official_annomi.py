import urllib.request
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ANNOMI_URL = "https://raw.githubusercontent.com/uccollab/AnnoMI/master/AnnoMI-simple.csv"
annomi_path = RAW_DIR / "AnnoMI-simple.csv"

print(f"Downloading official AnnoMI dataset from {ANNOMI_URL}...")
req = urllib.request.Request(ANNOMI_URL, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    df = pd.read_csv(resp)
    print(f"[SUCCESS] Downloaded official AnnoMI dataset!")
    print(f"  Rows: {len(df)}")
    print(f"  Transcript IDs: {df['transcript_id'].nunique()}")

df.to_csv(annomi_path, index=False, encoding="utf-8")
