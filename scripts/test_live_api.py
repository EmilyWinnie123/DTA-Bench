import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CLAUDE_API_KEY", "")
BASE_URL = "https://www.xiaoyaoapi.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_api():
    print("Testing live API call with model [G]claude-sonnet-5 ...")
    payload = {
        "model": "[G]claude-sonnet-5",
        "messages": [
            {"role": "system", "content": "You are an evaluator of Digital Therapeutic Alliance (DTA). Return valid JSON only."},
            {"role": "user", "content": "Return valid JSON: {\"status\": \"ok\", \"test\": 123}"}
        ],
        "temperature": 0.0
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=25)
        print(f"HTTP Status: {r.status_code}")
        if r.status_code == 200:
            res = r.json()
            content = res["choices"][0]["message"]["content"]
            print(f"[SUCCESS] API call successful! Content: {content}")
        else:
            print(f"[FAIL] HTTP {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[ERROR] Request exception: {e}")

if __name__ == "__main__":
    test_api()
