import urllib.request
import pandas as pd

users = ["zixiuwu", "zixiu-wu", "zixiu_wu", "WuZixiu", "wuzixiu", "Wu-Zixiu", "zixiu"]
repos = ["AnnoMI", "annomi", "AnnoMI-simple"]
filenames = ["AnnoMI-simple.csv", "AnnoMI_simple.csv", "AnnoMI-full.csv", "AnnoMI_full.csv", "annomi.csv"]
branches = ["main", "master"]

found = False
for u in users:
    for r in repos:
        for b in branches:
            for f in filenames:
                url = f"https://raw.githubusercontent.com/{u}/{r}/{b}/{f}"
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req) as resp:
                        df = pd.read_csv(resp)
                        print(f"FOUND REAL ANNOMI! URL: {url} | Rows: {len(df)}")
                        if "transcript_id" in df.columns:
                            print(f"  transcript_ids: {df['transcript_id'].nunique()}")
                        found = True
                        df.to_csv("data/raw/AnnoMI-simple.csv", index=False, encoding="utf-8")
                        break
                except Exception:
                    pass
            if found: break
        if found: break
    if found: break

if not found:
    print("GitHub raw scan did not find direct URL.")
