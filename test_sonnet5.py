import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("CLAUDE_API_KEY", ""),
    base_url=os.getenv("CLAUDE_API_BASE_URL", "https://www.xiaoyaoapi.com/v1")
)

for m in ['claude-sonnet-5', '[G]claude-sonnet-5', '[G]claude-sonnet-4-6']:
    try:
        res = client.chat.completions.create(
            model=m,
            messages=[
                {'role': 'system', 'content': 'You are a psychological evaluator. Return valid JSON only.'},
                {'role': 'user', 'content': 'Return JSON: {"status": "ok"}'}
            ],
            temperature=0.0
        )
        print(f'[OK] Model {m} response:', res.choices[0].message.content)
        break
    except Exception as e:
        print(f'[ERR] Model {m} failed:', e)
