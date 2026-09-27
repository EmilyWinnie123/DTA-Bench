import urllib.request
import json
import pandas as pd

hf_repos = [
    "hanaearg/AnnoMI_TypeUser",
    "TobiasPitters/annomi",
    "to-be/annomi-motivational-interviewing-therapy-conversations",
    "dexhrestha/annomi-sample",
    "Beijuka/AnnoMI-simple_speaker_diarisation"
]

for repo in hf_repos:
    api_url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            info = json.loads(resp.read().decode('utf-8'))
            siblings = [s.get("rfilename") for s in info.get("siblings", [])]
            print(f"\nRepo: {repo} | Files: {siblings}")
            for sf in siblings:
                if sf.endswith(".csv") or sf.endswith(".parquet") or sf.endswith(".json"):
                    raw_url = f"https://huggingface.co/datasets/{repo}/raw/main/{sf}"
                    try:
                        r2 = urllib.request.Request(raw_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(r2) as resp2:
                            if sf.endswith(".csv"):
                                df = pd.read_csv(resp2)
                                print(f"  -> File: {sf} | Rows: {len(df)} | Columns: {list(df.columns[:4])}")
                                if "transcript_id" in df.columns:
                                    print(f"     transcript_ids: {df['transcript_id'].nunique()}")
                    except Exception as e:
                        print(f"  -> Error fetching {sf}: {e}")
    except Exception as e:
        print(f"API Error {repo}: {e}")
