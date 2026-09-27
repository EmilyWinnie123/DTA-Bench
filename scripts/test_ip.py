import os
import requests
import urllib3
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings()

api_key = os.getenv("CLAUDE_API_KEY", "")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Host": "www.xiaoyaoapi.com"
}
payload = {
    "model": "[G]claude-sonnet-5",
    "messages": [
        {"role": "system", "content": "You are an evaluator of Digital Therapeutic Alliance (DTA). Return valid JSON only."},
        {"role": "user", "content": "Return valid JSON: {\"status\": \"ok\"}"}
    ],
    "temperature": 0.0
}

try:
    r = requests.post("https://202.47.135.211/v1/chat/completions", headers=headers, json=payload, verify=False, timeout=20)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print("Response:", r.json()["choices"][0]["message"]["content"])
    else:
        print("Error response:", r.text)
except Exception as e:
    print(f"Exception: {e}")
