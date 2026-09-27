import urllib.request
import json

url = "https://huggingface.co/api/datasets?search=AnnoMI"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))

for d in data:
    repo_id = d.get("id")
    api_files_url = f"https://huggingface.co/api/datasets/{repo_id}"
    try:
        r2 = urllib.request.Request(api_files_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(r2) as resp2:
            d_info = json.loads(resp2.read().decode('utf-8'))
            siblings = [s.get("rfilename") for s in d_info.get("siblings", [])]
            for sf in siblings:
                if "AnnoMI" in sf or "simple" in sf or "full" in sf or sf.endswith(".csv"):
                    print(f"Repo: {repo_id} | File: {sf}")
    except Exception as e:
        pass
