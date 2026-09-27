import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\1rog\dta_project")
SHEETS_DIR = BASE_DIR / "annotation" / "sheets"
ANNOTATION_DIR = BASE_DIR / "annotation"
CACHE_DIR = BASE_DIR / "cache"
CACHE_FILE = CACHE_DIR / "llm_responses.jsonl"
LIVE_PROGRESS_FILE = CACHE_DIR / "live_stage1_progress.json"
RUBRIC_FILE = BASE_DIR / "rubric" / "DTA_Rubric.md"

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("CLAUDE_API_KEY", "")
BASE_URL = os.getenv("CLAUDE_API_BASE_URL", "https://www.xiaoyaoapi.com/v1")

# Model strictly locked to Claude Sonnet 5 as requested by user
MODEL_CANDIDATES = ["[G]claude-sonnet-5"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def load_rubric_text():
    if RUBRIC_FILE.exists():
        with open(RUBRIC_FILE, "r", encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError(f"Rubric file not found at {RUBRIC_FILE}")

def call_sonnet(sys_prompt: str, user_prompt: str, max_retries: int = 8) -> str:
    for model in MODEL_CANDIDATES:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }
        for attempt in range(max_retries):
            try:
                r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    res = r.json()
                    content = res["choices"][0]["message"]["content"]
                    return content
                else:
                    print(f"  [Model {model} Attempt {attempt+1}] HTTP {r.status_code}: {r.text[:80]}", flush=True)
                    time.sleep(2 * (attempt + 1))
            except Exception as e:
                print(f"  [Model {model} Attempt {attempt+1}] Exception: {e}", flush=True)
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Failed to call API after retrying model {MODEL_CANDIDATES[0]}.")

def parse_json_from_response(text: str) -> dict:
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
        
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
        
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {"bond": "NA", "goal": "NA", "task": "NA", "raw": text}

def load_live_progress():
    if LIVE_PROGRESS_FILE.exists():
        try:
            with open(LIVE_PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_live_progress(progress):
    with open(LIVE_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def format_dialogue_window(sample: dict) -> str:
    ctx = str(sample.get("dialogue_context", "")).strip()
    su = str(sample.get("seeker_utterance", "")).strip()
    sr = str(sample.get("supporter_response", "")).strip()

    parts = []
    if ctx:
        parts.append(f"--- Dialogue Context ---\n{ctx}")
    if su and su.lower() not in ["nan", "none", "null", ""]:
        parts.append(f"Seeker: {su}")
    if sr:
        parts.append(f"--- Target Supporter Response to Evaluate ---\nSupporter: {sr}")
    
    return "\n\n".join(parts)

def run_live_stage1_sonnet5():
    print("=" * 80)
    print(" RUNNING LIVE STAGE 1 PROMPT ABLATION WITH FULL DTA-RUBRIC v1.0 ")
    print("=" * 80)

    rubric_text = load_rubric_text()
    print(f"[OK] Successfully loaded DTA_Rubric.md ({len(rubric_text)} chars).")

    gold_dev_file = SHEETS_DIR / "gold_standard_dev_20.xlsx"
    if not gold_dev_file.exists():
        gold_dev_file = SHEETS_DIR / "gold_standard.xlsx"
        
    df_dev = pd.read_excel(gold_dev_file, keep_default_na=False)
    if "pool_type" in df_dev.columns:
        df_dev = df_dev[df_dev["pool_type"] == "dev"].copy()
        
    samples = []
    for idx, r in df_dev.iterrows():
        samples.append({
            "sample_id": str(r["sample_id"]).strip(),
            "dialogue_context": str(r["dialogue_context"]).strip(),
            "seeker_utterance": str(r["seeker_utterance"]).strip(),
            "supporter_response": str(r["supporter_response"]).strip()
        })
        
    print(f"[OK] Loaded {len(samples)} development samples.")
    
    # SYSTEM PROMPT WITH FULL FROZEN DTA-RUBRIC v1.0 INJECTED
    sys_prompt = (
        "You are an expert evaluator of Digital Therapeutic Alliance (DTA) in text-based supportive dialogue.\n"
        "You MUST strictly apply the frozen DTA-Rubric v1.0 provided below.\n"
        "Follow all General Principles (P1-P6), score anchors (1-5), Task N/A decision rules (Section 6), and adjudicated edge cases (Section 7).\n"
        "The unit of evaluation is the dialogue window. Rate only observable alliance-relevant behaviour displayed by the supporter.\n"
        "Bond, Goal, and Task must be evaluated independently.\n"
        "Valid ratings are integer 1, 2, 3, 4, 5, or \"NA\". Do not use half-points.\n"
        "If genuinely uncertain between two adjacent scores, select the lower score (P6).\n"
        "Do not use information outside the dialogue window.\n\n"
        f"=== FROZEN DTA-RUBRIC V1.0 FULL TEXT ===\n{rubric_text}\n=== END OF RUBRIC ===\n"
    )

    p3_examples_text = """## Example 1 — Low scores across dimensions
Dialogue context:
Supporter: It seems like an exhausting time for a lot of people right now
Supporter: I'm also a bit tired but doing OK
Supporter: What's been on your mind recently?
Seeker: just trying to finish this semester
Supporter: It seems like a stressful time to be in college.
Response to evaluate:
I'm sure you are busy, but realize, your emotional health is important too and we always need to make time to connect with people.
Gold rating:
{
"bond": 1,
"goal": 1,
"task": 1
}

---

## Example 2 — Dimensions can differ substantially
Dialogue context:
Therapist: Yeah. Well, I'm so glad that you reached out for help, um, and just know that you're not alone. And-and so, yeah, is that okay if I share a little bit about, um, how I tend to work with clients on that topic?
Client: Yes, please.
Therapist: Yeah. Well, um, first of all, I really am passionate about looking at the whole picture, um, physical health and emotional wellbeing...
Response to evaluate:
Um, and also look at activities that enhance your mood... What do you make of all that?
Gold rating:
{
"bond": 2,
"goal": 4,
"task": 4
}

---

## Example 3 — High Bond with Task not applicable (N/A)
Dialogue context:
Seeker: Yea i suppose your right. Btw yes i love to soak in the tub...
Supporter: Well there you go...
Response to evaluate:
Oh that's exciting! You get to start dating again! What do you look for in a man?
Gold rating:
{
"bond": 5,
"goal": 4,
"task": "NA"
}
"""

    conditions_meta = [
        ("P1 Zero-shot", "zero_shot", CACHE_DIR / "stage1_p1_zero_shot.json"),
        ("P2 Structured", "cot_then_score", CACHE_DIR / "stage1_p2_structured.json"),
        ("P3 Few-shot", "few_shot", CACHE_DIR / "stage1_p3_few_shot.json"),
        ("P4 Dimension-wise", "dimension_wise", CACHE_DIR / "stage1_p4_dimensionwise.json")
    ]

    progress = load_live_progress()

    for p_label, cond_key, out_json_path in conditions_meta:
        print(f"\n--- Running Condition: {p_label} ({cond_key}) ---", flush=True)
        if cond_key not in progress:
            progress[cond_key] = {}

        p_records = []

        for s_idx, s in enumerate(samples, 1):
            sid = s["sample_id"]
            
            # Check if already in progress cache
            if sid in progress[cond_key]:
                cached_res = progress[cond_key][sid]
                b = cached_res.get("bond", "NA")
                g = cached_res.get("goal", "NA")
                t = cached_res.get("task", "NA")
                p_records.append({"sample_id": sid, "bond": b, "goal": g, "task": t})
                print(f"  [{s_idx:02d}/20] Sample {sid} (Cached) -> Bond={b}, Goal={g}, Task={t}", flush=True)
                continue

            dialogue_text = format_dialogue_window(s)

            if cond_key == "zero_shot":
                user_prompt = f"Using the frozen DTA-Rubric v1.0, independently rate the following dialogue window on Bond, Goal, and Task.\n\n{dialogue_text}\n\nReturn ONLY valid JSON:\n{{\n\"bond\": 1-5 or \"NA\",\n\"goal\": 1-5 or \"NA\",\n\"task\": 1-5 or \"NA\"\n}}"
            
            elif cond_key == "cot_then_score":
                user_prompt = f"Using the frozen DTA-Rubric v1.0, evaluate the following dialogue window systematically.\nFor each dimension:\n1. Determine whether scorable.\n2. Identify observable supporter behaviour.\n3. Match closest score anchor.\n4. Apply Task N/A decision rules (Section 6) and Edge Cases (Section 7).\n5. Assign final rating (1-5 or NA).\n\n{dialogue_text}\n\nReturn ONLY valid JSON:\n{{\n\"reasoning\": \"Step-by-step evaluation...\",\n\"bond\": 1-5 or \"NA\",\n\"goal\": 1-5 or \"NA\",\n\"task\": 1-5 or \"NA\"\n}}"
                
            elif cond_key == "few_shot":
                user_prompt = f"You are evaluating Digital Therapeutic Alliance (DTA) using the frozen DTA-Rubric v1.0.\nHere are calibration examples:\n{p3_examples_text}\n\nNow evaluate the target dialogue window independently.\nTarget dialogue:\n{dialogue_text}\n\nReturn ONLY valid JSON:\n{{\n\"bond\": 1-5 or \"NA\",\n\"goal\": 1-5 or \"NA\",\n\"task\": 1-5 or \"NA\"\n}}"
                
            elif cond_key == "dimension_wise":
                user_prompt = f"Using the frozen DTA-Rubric v1.0, evaluate each dimension independently (Bond first, then Goal, then Task following Section 1 recommendations).\nMake sure to apply the Section 6 Task N/A 3-step decision rule specifically for Task.\n\n{dialogue_text}\n\nReturn ONLY valid JSON:\n{{\n\"bond_eval\": \"...\",\n\"bond\": 1-5 or \"NA\",\n\"goal_eval\": \"...\",\n\"goal\": 1-5 or \"NA\",\n\"task_eval\": \"...\",\n\"task\": 1-5 or \"NA\"\n}}"

            print(f"  [{s_idx:02d}/20] Sample {sid} ...", end=" ", flush=True)
            raw_out = call_sonnet(sys_prompt, user_prompt)
            parsed = parse_json_from_response(raw_out)
            
            b = parsed.get("bond", parsed.get("Bond", "NA"))
            g = parsed.get("goal", parsed.get("Goal", "NA"))
            t = parsed.get("task", parsed.get("Task", "NA"))
            
            item_out = {"sample_id": sid, "bond": b, "goal": g, "task": t}
            p_records.append(item_out)
            progress[cond_key][sid] = item_out
            save_live_progress(progress)
            
            print(f"Done -> Bond={b}, Goal={g}, Task={t}", flush=True)

            # Save to llm_responses.jsonl cache
            record_cache = {
                "sample_id": sid,
                "dataset": "ESConv",
                "stage": 1,
                "model": "claude-sonnet-5",
                "condition": cond_key,
                "run_idx": 0,
                "dimension": "all",
                "raw_output": raw_out,
                "parsed_output": {"reasoning": raw_out, "scores": {"Bond": b, "Goal": g, "Task": t}},
                "rubric_sha256": "06fbdb6bfc78865cda4b9ba47349eb133a3db1787b7f40d9c6a581bfd1138e63",
                "estimated_cost": 0.003,
                "timestamp": time.time()
            }
            with open(CACHE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_cache, ensure_ascii=False) + "\n")

        # Save condition raw JSON
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(p_records, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved condition {p_label} predictions to {out_json_path}", flush=True)

    print("\n[OK] All 4 Prompt conditions executed live with Sonnet 5 and FULL DTA-Rubric v1.0!", flush=True)

    # Run stage1_prompt_ablation.py to calculate final tables
    import subprocess
    print("\nExecuting stage1_prompt_ablation.py to compute Stage 1 tables...", flush=True)
    res = subprocess.run([sys.executable, "scripts/stage1_prompt_ablation.py"], capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.stderr:
        print("Stderr:", res.stderr, flush=True)

if __name__ == "__main__":
    run_live_stage1_sonnet5()
