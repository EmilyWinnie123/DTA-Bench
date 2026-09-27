import urllib.request
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

splits = ["train.csv", "validation.csv", "test.csv"]
dfs = []

for split in splits:
    url = f"https://huggingface.co/datasets/BurgerTruck/AnnoMI/raw/main/{split}"
    print(f"Fetching AnnoMI split {split} from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            df = pd.read_csv(resp)
            print(f"Loaded {split}: {len(df)} rows")
            dfs.append(df)
    except Exception as e:
        print(f"Error fetching {split}: {e}")

if dfs:
    combined_annomi = pd.concat(dfs, ignore_index=True)
    combined_annomi.to_csv(RAW_DIR / "AnnoMI-simple.csv", index=False, encoding="utf-8")
    print(f"[SUCCESS] Saved REAL AnnoMI dataset! Total Rows: {len(combined_annomi)}, Transcripts: {combined_annomi['transcript_id'].nunique()}")
