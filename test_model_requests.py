import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("CLAUDE_API_KEY", "")
BASE_URL = os.getenv("CLAUDE_API_BASE_URL", "https://www.xiaoyaoapi.com/v1")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

models_to_test = ["claude-sonnet-5", "[G]claude-sonnet-5", "[G]claude-sonnet-4-6"]

for m in models_to_test:
    print(f"Testing model: {m} ...")
    payload = {
        "model": m,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Respond with JSON: {\"status\": \"ok\"}"}
        ],
        "temperature": 0.0
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            res = r.json()
            content = res["choices"][0]["message"]["content"]
            print(f"[SUCCESS] Model {m} worked! Response: {content[:100]}")
            sys.exit(0)
        else:
            print(f"[FAIL] Model {m} HTTP {r.status_code}: {r.text[:150]}")
    except Exception as e:
        print(f"[ERR] Model {m} error: {e}")
