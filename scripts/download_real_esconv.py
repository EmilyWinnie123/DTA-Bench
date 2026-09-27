import urllib.request
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

ESCONV_URL = "https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json"
esconv_path = RAW_DIR / "ESConv.json"

print(f"Downloading official ESConv dataset from {ESCONV_URL}...")
req = urllib.request.Request(ESCONV_URL, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    print(f"[SUCCESS] Downloaded official ESConv dataset! Dialogues: {len(data)}")

with open(esconv_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
