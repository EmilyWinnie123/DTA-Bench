import os
import sys
import json
import time
import hashlib
import threading
import requests
import urllib3
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
SHEETS_DIR = BASE_DIR / "annotation" / "sheets"
CACHE_DIR = BASE_DIR / "cache"

STUDY2_PROGRESS_FILE = CACHE_DIR / "study2_ai_generation_progress.json"
STUDY2_RESPONSES_FILE = CACHE_DIR / "study2_ai_responses.json"

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

# Generator A: Gemini Flash
GEN_A_MODEL = os.getenv("GEN_A_MODEL", "Gemini-3.8-Flash/Antigravity")
GEN_A_URL = os.getenv("GEN_A_URL", "https://api.meow61.my/v1/chat/completions")
GEN_A_KEY = os.getenv("GEMINI_API_KEY", "")

# Generator B: DeepSeek V4 Pro
GEN_B_MODEL = os.getenv("GEN_B_MODEL", "deepseek-v4-pro-0813")
GEN_B_URL = os.getenv("GEN_B_URL", "https://api.jichuanai.com/v1/chat/completions")
GEN_B_KEY = os.getenv("DEEPSEEK_API_KEY", "")

SYSTEM_PROMPT = "You are a supportive listener. Respond to the person's latest message."

file_lock = threading.Lock()

def build_generator_user_prompt(ctx: str, seeker: str) -> str:
    parts = []
    if ctx and ctx.strip():
        parts.append(f"[Dialogue Context]\n{ctx.strip()}")
    if seeker and seeker.strip():
        parts.append(f"[Person's Latest Message]\n{seeker.strip()}")
    res = "\n\n".join(parts)
    if not res.strip():
        return "[Initial Turn]\nThe person has just initiated the conversation and is seeking support. Please provide a warm opening supportive response."
    return res

def call_model_generator(url: str, key: str, model_id: str, sys_prompt: str, user_prompt: str, max_retries: int = 8) -> tuple[str, dict]:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=45)
            if r.status_code == 200:
                res = r.json()
                content = res["choices"][0]["message"]["content"]
                usage = res.get("usage", {})
                return content, usage
            elif r.status_code == 429:
                err_msg = "HTTP 429 Rate limit hit"
                last_err = err_msg
                print(f"    [{model_id} Attempt {attempt}/{max_retries}] 429 Rate Limit, backing off...", flush=True)
                time.sleep(3 * attempt)
                continue
            else:
                err_msg = f"HTTP {r.status_code}: {r.text[:120]}"
                last_err = err_msg
                print(f"    [{model_id} Attempt {attempt}/{max_retries}] {err_msg}", flush=True)
        except Exception as e:
            err_msg = f"Exception: {e}"
            last_err = err_msg
            print(f"    [{model_id} Attempt {attempt}/{max_retries}] {err_msg}", flush=True)

        time.sleep(2 * attempt)

    raise RuntimeError(f"CRITICAL API FAILURE on generator model {model_id}. Last Error: {last_err}")

def load_progress():
    if STUDY2_PROGRESS_FILE.exists():
        try:
            with open(STUDY2_PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_progress_safe(progress):
    with file_lock:
        with open(STUDY2_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

def process_single_sample(sample):
    sid = sample["sample_id"]
    ds = sample["dataset"]
    ctx = sample["dialogue_context"]
    seeker = sample["seeker_utterance"]

    user_prompt = build_generator_user_prompt(ctx, seeker)

    # Call Generator A (Gemini)
    resp_gemini, usage_gemini = call_model_generator(
        GEN_A_URL, GEN_A_KEY, GEN_A_MODEL, SYSTEM_PROMPT, user_prompt
    )

    # Call Generator B (DeepSeek)
    resp_deepseek, usage_deepseek = call_model_generator(
        GEN_B_URL, GEN_B_KEY, GEN_B_MODEL, SYSTEM_PROMPT, user_prompt
    )

    item_res = {
        "sample_id": sid,
        "dataset": ds,
        "dialogue_context": ctx,
        "seeker_utterance": seeker,
        "human_response": sample["supporter_response"],
        "gemini_response": resp_gemini,
        "deepseek_response": resp_deepseek,
        "usage": {
            "gemini": usage_gemini,
            "deepseek": usage_deepseek
        },
        "timestamp": time.time()
    }

    return sid, item_res

def run_study2_generation(test_mode: bool = True):
    mode_str = "3-SAMPLE PILOT TEST" if test_mode else "FULL 150-SAMPLE EXECUTION"
    print("=" * 100, flush=True)
    print(f" STUDY 2 — AI RESPONSE GENERATION EXPERIMENT ({mode_str}) ", flush=True)
    print("=" * 100, flush=True)

    eval_gold_file = SHEETS_DIR / "gold_standard_eval_150.xlsx"
    assert eval_gold_file.exists(), f"CRITICAL ERROR: Human Gold Standard file missing at {eval_gold_file}"

    df_gold = pd.read_excel(eval_gold_file, keep_default_na=False)
    df_eval = df_gold[df_gold["pool_type"] == "eval"].copy() if "pool_type" in df_gold.columns else df_gold.copy()
    assert len(df_eval) == 150, f"DATA PROVENANCE ERROR: Expected 150 eval samples, got {len(df_eval)}"

    file_mtime = time.ctime(os.path.getmtime(eval_gold_file))
    xls = pd.ExcelFile(eval_gold_file)
    sheet_name = xls.sheet_names[0]
    cell_a1 = str(df_gold.columns[0])

    prompt_sha256 = hashlib.sha256(SYSTEM_PROMPT.encode('utf-8')).hexdigest()

    print("\n--- DATA & ENVIRONMENT PROVENANCE ---", flush=True)
    print(f"  - Human Gold Standard File: {eval_gold_file.resolve()}", flush=True)
    print(f"  - File Last Modified:       {file_mtime}", flush=True)
    print(f"  - Sheet Name:               {sheet_name}", flush=True)
    print(f"  - A1 Cell Header:           '{cell_a1}'", flush=True)
    print(f"  - Total Eval Samples:       {len(df_eval)} (ESConv={sum(df_eval['dataset']=='ESConv')}, AnnoMI={sum(df_eval['dataset']=='AnnoMI')})", flush=True)
    print(f"  - Generator A (Gemini):     {GEN_A_MODEL} (URL: {GEN_A_URL})", flush=True)
    print(f"  - Generator B (DeepSeek):   {GEN_B_MODEL} (URL: {GEN_B_URL})", flush=True)
    print(f"  - System Prompt:            '{SYSTEM_PROMPT}'", flush=True)
    print(f"  - System Prompt SHA-256:    {prompt_sha256}", flush=True)
    print(f"  - Generation Parameters:    temperature=0.7", flush=True)
    print(f"  - Model Switch on Failure:  Disabled", flush=True)

    samples = []
    for idx, r in df_eval.iterrows():
        samples.append({
            "sample_id": str(r["sample_id"]).strip(),
            "dataset": str(r["dataset"]).strip(),
            "dialogue_context": str(r["dialogue_context"]).strip(),
            "seeker_utterance": str(r["seeker_utterance"]).strip(),
            "supporter_response": str(r["supporter_response"]).strip()
        })

    if test_mode:
        samples_to_run = samples[:3]
    else:
        samples_to_run = samples

    progress = load_progress()
    pending = [s for s in samples_to_run if s["sample_id"] not in progress]

    total_to_run = len(samples_to_run)
    done_count = total_to_run - len(pending)

    print(f"\n--- EXECUTING AI GENERATION ({len(pending)} PENDING / {done_count} CACHED OUT OF {total_to_run}) ---", flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_sample = {executor.submit(process_single_sample, s): s for s in pending}
            for future in as_completed(future_to_sample):
                s = future_to_sample[future]
                try:
                    sid, item_res = future.result()
                    progress[sid] = item_res
                    save_progress_safe(progress)
                    done_count += 1
                    print(f"  [{done_count:03d}/{total_to_run}] Generated {sid:20s} ({s['dataset']:6s})", flush=True)
                except Exception as e:
                    print(f"  [ERROR] Sample {s['sample_id']} generation failed: {e}", flush=True)
                    raise e

    # Save final responses file
    with open(STUDY2_RESPONSES_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Progress and responses saved to {STUDY2_RESPONSES_FILE}", flush=True)

    if test_mode:
        print("\n" + "=" * 80, flush=True)
        print(" PILOT TEST (3 SAMPLES) OUTPUT INSPECTION ", flush=True)
        print("=" * 80, flush=True)
        for idx, s in enumerate(samples[:3], 1):
            sid = s["sample_id"]
            res = progress[sid]
            print(f"\n--- Sample #{idx}: {sid} (Dataset: {s['dataset']}) ---")
            print(f"  [Seeker Utterance]\n  {s['seeker_utterance'][:150]}...")
            print(f"  [Human Original Response]\n  {s['supporter_response']}")
            print(f"  [Gemini Response (Gemini-3.8-Flash)]\n  {res['gemini_response']}")
            print(f"  [DeepSeek Response (deepseek-v4-pro-0813)]\n  {res['deepseek_response']}")
        print("\n" + "=" * 80, flush=True)
        print(" [OK] Pilot test complete and verified. Ready for full 150-sample execution! ", flush=True)
        print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full 150 samples instead of 3-sample pilot test")
    args = parser.parse_args()

    run_study2_generation(test_mode=not args.full)
