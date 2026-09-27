import os
import sys
import json
import time
import hashlib
import threading
import re
import requests
import urllib3
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import cohen_kappa_score
import pingouin as pg

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from prompts import (
    load_judge_p4_config,
    get_rubric_text_and_hash,
    build_final_p4_prompt
)

CACHE_DIR = BASE_DIR / "cache"
ANON_MAP_FILE = CACHE_DIR / "study2_anonymization_map.json"

CLAUDE_PROGRESS_FILE = CACHE_DIR / "study2_claude_eval_progress.json"
CLAUDE_PREDS_FILE = CACHE_DIR / "study2_claude_eval_predictions.json"

QWEN_PROGRESS_FILE = CACHE_DIR / "study2_qwen_eval_progress.json"
QWEN_PREDS_FILE = CACHE_DIR / "study2_qwen_eval_predictions.json"

SUMMARY_FILE = CACHE_DIR / "study2_dual_judge_summary.json"

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

# Judge 1: Claude Sonnet 5
CLAUDE_MODEL = "[G]claude-sonnet-5"
CLAUDE_URL = os.getenv("CLAUDE_API_URL", "https://202.47.135.211/v1/chat/completions")
CLAUDE_KEY = os.getenv("CLAUDE_API_KEY", "")

# Judge 2: Qwen3.8-Max
QWEN_MODEL = "Qwen3.8-Max"
QWEN_URL = os.getenv("QWEN_API_URL", "https://novaapi.eduyuwo.com/v1/chat/completions")
QWEN_KEY = os.getenv("QWEN_API_KEY", "")

file_lock = threading.Lock()

def parse_val(v):
    s = str(v).strip().upper()
    if s in ["NA", "N/A", "NONE", "NULL", ""]:
        return "NA"
    try:
        f_v = float(v)
        if f_v.is_integer():
            return str(int(f_v))
        return str(f_v)
    except ValueError:
        return "NA"

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

def call_claude_judge(sys_prompt: str, user_prompt: str, max_retries: int = 8) -> tuple[str, dict]:
    payload = {
        "model": CLAUDE_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }
    headers = {
        "Authorization": f"Bearer {CLAUDE_KEY}",
        "Content-Type": "application/json",
        "Host": "www.xiaoyaoapi.com"
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(CLAUDE_URL, headers=headers, json=payload, verify=False, timeout=45)
            if r.status_code == 200:
                res = r.json()
                content = res["choices"][0]["message"]["content"]
                usage = res.get("usage", {})
                return content, usage
            else:
                err_msg = f"HTTP {r.status_code}: {r.text[:120]}"
                last_err = err_msg
                print(f"    [Claude Attempt {attempt}/{max_retries}] {err_msg}", flush=True)
        except Exception as e:
            err_msg = f"Exception: {e}"
            last_err = err_msg
            print(f"    [Claude Attempt {attempt}/{max_retries}] {err_msg}", flush=True)

        time.sleep(2 * attempt)

    raise RuntimeError(f"CRITICAL API FAILURE on Claude Judge. Last Error: {last_err}")

def call_qwen_judge(sys_prompt: str, user_prompt: str, max_retries: int = 12) -> tuple[str, dict]:
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }
    headers = {
        "Authorization": f"Bearer {QWEN_KEY}",
        "Content-Type": "application/json"
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(QWEN_URL, headers=headers, json=payload, timeout=45)
            if r.status_code == 200:
                res = r.json()
                content = res["choices"][0]["message"]["content"]
                usage = res.get("usage", {})
                return content, usage
            elif r.status_code == 429:
                err_msg = "HTTP 429 Rate limit hit"
                last_err = err_msg
                print(f"    [Qwen Attempt {attempt}/{max_retries}] 429 Rate limit, backing off...", flush=True)
                time.sleep(3 * attempt)
                continue
            else:
                err_msg = f"HTTP {r.status_code}: {r.text[:120]}"
                last_err = err_msg
                print(f"    [Qwen Attempt {attempt}/{max_retries}] {err_msg}", flush=True)
        except Exception as e:
            err_msg = f"Exception: {e}"
            last_err = err_msg
            print(f"    [Qwen Attempt {attempt}/{max_retries}] {err_msg}", flush=True)

        time.sleep(2 * attempt)

    raise RuntimeError(f"CRITICAL API FAILURE on Qwen Judge. Last Error: {last_err}")

def load_json(p: Path) -> dict:
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json_safe(p: Path, data: dict):
    with file_lock:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def eval_single_item(item, judge_type, raw_sha):
    anon_id = item["anon_id"]
    ds = item["dataset"]

    sample_dict = {
        "sample_id": anon_id,
        "dataset": ds,
        "dialogue_context": item["dialogue_context"],
        "seeker_utterance": item["seeker_utterance"],
        "supporter_response": item["supporter_response"]
    }

    sys_prompt, user_prompt, meta = build_final_p4_prompt(sample_dict)

    if judge_type == "claude":
        raw_out, usage = call_claude_judge(sys_prompt, user_prompt)
    else:
        raw_out, usage = call_qwen_judge(sys_prompt, user_prompt)

    parsed = parse_json_from_response(raw_out)
    b = parse_val(parsed.get("bond", parsed.get("Bond", "NA")))
    g = parse_val(parsed.get("goal", parsed.get("Goal", "NA")))
    t = parse_val(parsed.get("task", parsed.get("Task", "NA")))

    out_record = {
        "anon_id": anon_id,
        "sample_id": item["sample_id"],
        "dataset": ds,
        "source": item["source"],
        "bond": b,
        "goal": g,
        "task": t,
        "raw_output": raw_out,
        "usage": usage
    }

    return anon_id, b, g, t, out_record

def calc_icc21_safe(df_icc):
    if len(df_icc) < 3:
        return 0.0
    try:
        res = pg.intraclass_corr(data=df_icc, targets="Targets", raters="Raters", ratings="Scores")
        r21 = res[res["Type"] == "ICC(A,1)"]
        if not r21.empty:
            return float(r21["ICC"].iloc[0])
        return float(res["ICC"].iloc[0])
    except Exception:
        return 0.0

def run_study2_dual_judge_evaluation():
    print("=" * 110, flush=True)
    print(" STUDY 2 — DUAL-JUDGE BLIND EVALUATION EXECUTION (450 ANONYMIZED RESPONSES) ", flush=True)
    print("=" * 110, flush=True)

    assert ANON_MAP_FILE.exists(), f"CRITICAL ERROR: Anonymization map file missing at {ANON_MAP_FILE}"
    with open(ANON_MAP_FILE, "r", encoding="utf-8") as f:
        anon_items = json.load(f)

    total_anon = len(anon_items)
    assert total_anon == 450, f"DATA ERROR: Expected 450 anonymized items, got {total_anon}"

    config = load_judge_p4_config()
    rubric_text, raw_sha, text_sha = get_rubric_text_and_hash()

    print("\n--- DATA & ENVIRONMENT PROVENANCE ---", flush=True)
    print(f"  - Anonymization Map File:   {ANON_MAP_FILE.resolve()}", flush=True)
    print(f"  - Total Anonymized Items:   {total_anon} (150 Human + 150 Gemini + 150 DeepSeek)", flush=True)
    print(f"  - Randomization Seed:       42", flush=True)
    print(f"  - Judge 1 Model ID:         {CLAUDE_MODEL} (Direct IP Proxy)", flush=True)
    print(f"  - Judge 2 Model ID:         {QWEN_MODEL} (NovaAPI)", flush=True)
    print(f"  - Total Evaluations:        900 (450 Claude + 450 Qwen)", flush=True)
    print(f"  - Temperature:              0.0", flush=True)
    print(f"  - Frozen Rubric Raw SHA256: {raw_sha}", flush=True)

    # 1. QWEN EVALUATION RUN (JUDGE 2)
    qwen_prog = load_json(QWEN_PROGRESS_FILE)
    qwen_pending = [item for item in anon_items if item["anon_id"] not in qwen_prog]

    print(f"\n--- RUNNING JUDGE 2 (QWEN3.8-MAX) EVALUATION ({len(qwen_pending)} PENDING / {len(qwen_prog)} CACHED) ---", flush=True)

    if qwen_pending:
        done_q = len(qwen_prog)
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_item = {executor.submit(eval_single_item, item, "qwen", raw_sha): item for item in qwen_pending}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    anon_id, b, g, t, out_rec = future.result()
                    qwen_prog[anon_id] = out_rec
                    save_json_safe(QWEN_PROGRESS_FILE, qwen_prog)
                    done_q += 1
                    print(f"  [Qwen {done_q:03d}/{total_anon}] Done {anon_id} -> Bond={b}, Goal={g}, Task={t}", flush=True)
                except Exception as e:
                    print(f"  [ERROR] Qwen failed on {item['anon_id']}: {e}", flush=True)
                    raise e

    save_json_safe(QWEN_PREDS_FILE, qwen_prog)
    print(f"[OK] Qwen evaluations completed & saved to {QWEN_PREDS_FILE}", flush=True)

    # 2. CLAUDE EVALUATION RUN (JUDGE 1)
    claude_prog = load_json(CLAUDE_PROGRESS_FILE)
    claude_pending = [item for item in anon_items if item["anon_id"] not in claude_prog]

    print(f"\n--- RUNNING JUDGE 1 (CLAUDE SONNET 5) EVALUATION ({len(claude_pending)} PENDING / {len(claude_prog)} CACHED) ---", flush=True)

    claude_error = None
    if claude_pending:
        done_c = len(claude_prog)
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_item = {executor.submit(eval_single_item, item, "claude", raw_sha): item for item in claude_pending}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    anon_id, b, g, t, out_rec = future.result()
                    claude_prog[anon_id] = out_rec
                    save_json_safe(CLAUDE_PROGRESS_FILE, claude_prog)
                    done_c += 1
                    print(f"  [Claude {done_c:03d}/{total_anon}] Done {anon_id} -> Bond={b}, Goal={g}, Task={t}", flush=True)
                except Exception as e:
                    err_msg = str(e)
                    print(f"  [PAUSED] Claude evaluation paused on {item['anon_id']}: {err_msg}", flush=True)
                    claude_error = err_msg
                    break

    save_json_safe(CLAUDE_PREDS_FILE, claude_prog)
    print(f"[OK] Claude evaluation progress saved to {CLAUDE_PREDS_FILE} (Cached: {len(claude_prog)}/{total_anon})", flush=True)

    # 3. DE-ANONYMIZE AND COMPUTE DESCRIPTIVE & AGREEMENT METRICS
    print(f"\n--- DE-ANONYMIZING AND COMPUTING STUDY 2 METRICS ---", flush=True)

    # Group ratings by source (Human, Gemini, DeepSeek)
    sources = ["Human", "Gemini", "DeepSeek"]
    dims = ["Bond", "Goal", "Task"]

    source_ratings = {
        s: {"Claude": {d: [] for d in dims}, "Qwen": {d: [] for d in dims}}
        for s in sources
    }

    # Inter-judge paired lists across all 450 responses
    paired_judge_data = {
        d: {"paired_c": [], "paired_q": [], "valid_n": 0, "both_na": 0}
        for d in dims
    }

    for item in anon_items:
        aid = item["anon_id"]
        src = item["source"]

        c_rec = claude_prog[aid]
        q_rec = qwen_prog[aid]

        for dim in dims:
            cb = c_rec[dim.lower()]
            qb = q_rec[dim.lower()]

            source_ratings[src]["Claude"][dim].append(cb)
            source_ratings[src]["Qwen"][dim].append(qb)

            # Check paired validity
            c_is_na = (cb == "NA")
            q_is_na = (qb == "NA")

            if not c_is_na and not q_is_na:
                paired_judge_data[dim]["paired_c"].append(float(cb))
                paired_judge_data[dim]["paired_q"].append(float(qb))
                paired_judge_data[dim]["valid_n"] += 1
            elif c_is_na and q_is_na:
                paired_judge_data[dim]["both_na"] += 1

    # Descriptive statistics per source per dimension
    def calc_score_stats(vals):
        numeric_vals = [float(v) for v in vals if v != "NA"]
        na_count = sum(1 for v in vals if v == "NA")
        na_rate = (na_count / len(vals)) * 100.0
        if numeric_vals:
            return {
                "count": len(vals),
                "valid_n": len(numeric_vals),
                "mean": round(float(np.mean(numeric_vals)), 4),
                "std": round(float(np.std(numeric_vals)), 4),
                "median": round(float(np.median(numeric_vals)), 4),
                "na_count": na_count,
                "na_rate_pct": round(na_rate, 2)
            }
        else:
            return {
                "count": len(vals),
                "valid_n": 0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "na_count": na_count,
                "na_rate_pct": round(na_rate, 2)
            }

    source_descriptives = {}
    for src in sources:
        source_descriptives[src] = {}
        for j_name in ["Claude", "Qwen"]:
            source_descriptives[src][j_name] = {}
            for dim in dims:
                vals = source_ratings[src][j_name][dim]
                source_descriptives[src][j_name][dim] = calc_score_stats(vals)

    # Inter-Judge agreement metrics (Claude vs Qwen) on all 450 Study 2 responses
    judge_agreement = {}
    for dim in dims:
        p_c = paired_judge_data[dim]["paired_c"]
        p_q = paired_judge_data[dim]["paired_q"]
        v_n = paired_judge_data[dim]["valid_n"]
        b_na = paired_judge_data[dim]["both_na"]

        if v_n >= 3:
            qwk = float(cohen_kappa_score(p_c, p_q, weights="quadratic", labels=[1, 2, 3, 4, 5]))
            diffs = np.array(p_q) - np.array(p_c)
            mae = float(np.mean(np.abs(diffs)))
            exact_num = int(np.sum(np.abs(diffs) == 0))
            num_exact_pct = (exact_num / v_n) * 100.0

            df_icc = pd.DataFrame({
                "Targets": list(range(v_n)) * 2,
                "Raters": ["Claude"] * v_n + ["Qwen"] * v_n,
                "Scores": p_c + p_q
            })
            icc21 = calc_icc21_safe(df_icc)
        else:
            qwk, mae, icc21, exact_num, num_exact_pct = 0.0, 0.0, 0.0, 0, 0.0

        overall_exact = exact_num + b_na
        overall_exact_pct = (overall_exact / total_anon) * 100.0
        na_agree_pct = ((b_na + v_n) / total_anon) * 100.0

        judge_agreement[dim] = {
            "total_n": total_anon,
            "valid_n": v_n,
            "both_na": b_na,
            "icc21": round(icc21, 4),
            "qwk": round(qwk, 4),
            "mae": round(mae, 4),
            "numeric_exact_count": exact_num,
            "numeric_exact_pct": round(num_exact_pct, 2),
            "overall_exact_count": overall_exact,
            "overall_exact_pct": round(overall_exact_pct, 2),
            "na_agree_pct": round(na_agree_pct, 2)
        }

    macro_icc = round(float(np.mean([judge_agreement[d]["icc21"] for d in dims])), 4)
    macro_qwk = round(float(np.mean([judge_agreement[d]["qwk"] for d in dims])), 4)
    macro_mae = round(float(np.mean([judge_agreement[d]["mae"] for d in dims])), 4)
    macro_num_exact = round(float(np.mean([judge_agreement[d]["numeric_exact_pct"] for d in dims])), 2)
    macro_overall_exact = round(float(np.mean([judge_agreement[d]["overall_exact_pct"] for d in dims])), 2)
    macro_na_agree = round(float(np.mean([judge_agreement[d]["na_agree_pct"] for d in dims])), 2)

    judge_agreement["macro_averages"] = {
        "macro_icc21": macro_icc,
        "macro_qwk": macro_qwk,
        "macro_mae": macro_mae,
        "macro_num_exact_pct": macro_num_exact,
        "macro_overall_exact_pct": macro_overall_exact,
        "macro_na_agree_pct": macro_na_agree
    }

    summary_data = {
        "provenance": {
            "anonymization_map": str(ANON_MAP_FILE.resolve()),
            "claude_preds_file": str(CLAUDE_PREDS_FILE.resolve()),
            "qwen_preds_file": str(QWEN_PREDS_FILE.resolve()),
            "total_responses": total_anon,
            "random_seed": 42
        },
        "source_descriptives": source_descriptives,
        "inter_judge_agreement": judge_agreement
    }

    save_json_safe(SUMMARY_FILE, summary_data)
    print(f"\n[OK] Study 2 dual-judge summary saved to {SUMMARY_FILE}", flush=True)

    # PRINT FORMATTED TABLES
    print("\n" + "=" * 105, flush=True)
    print(" STUDY 2 DUAL-JUDGE EVALUATION RESULTS ", flush=True)
    print("=" * 105, flush=True)

    print("\n--- 1. RATING DISTRIBUTIONS & TASK N/A PROPORTIONS BY RESPONSE SOURCE ---", flush=True)
    desc_rows = []
    for src in sources:
        for j_name in ["Claude", "Qwen"]:
            for dim in dims:
                st = source_descriptives[src][j_name][dim]
                desc_rows.append({
                    "Source": src,
                    "Judge": j_name,
                    "Dimension": dim,
                    "Valid N": st["valid_n"],
                    "Mean": f"{st['mean']:.2f}",
                    "Std": f"{st['std']:.2f}",
                    "Median": f"{st['median']:.2f}",
                    "N/A Count": st["na_count"],
                    "N/A Rate %": f"{st['na_rate_pct']:.2f}%"
                })
    df_desc = pd.DataFrame(desc_rows)
    print(df_desc.to_string(index=False), flush=True)

    print("\n--- 2. INTER-JUDGE AGREEMENT (CLAUDE VS QWEN ON 450 AI/HUMAN RESPONSES) ---", flush=True)
    j_rows = []
    for dim in dims:
        ja = judge_agreement[dim]
        j_rows.append({
            "Dimension": dim,
            "Total N": ja["total_n"],
            "Valid N": ja["valid_n"],
            "ICC (2,1)": f"{ja['icc21']:.4f}",
            "QWK": f"{ja['qwk']:.4f}",
            "MAE": f"{ja['mae']:.4f}",
            "Numeric Exact %": f"{ja['numeric_exact_pct']:.2f}%",
            "Overall Exact %": f"{ja['overall_exact_pct']:.2f}%",
            "N/A Agree %": f"{ja['na_agree_pct']:.2f}%"
        })
    m_ja = judge_agreement["macro_averages"]
    j_rows.append({
        "Dimension": "**Macro Avg**",
        "Total N": 450,
        "Valid N": "-",
        "ICC (2,1)": f"**{m_ja['macro_icc21']:.4f}**",
        "QWK": f"**{m_ja['macro_qwk']:.4f}**",
        "MAE": f"**{m_ja['macro_mae']:.4f}**",
        "Numeric Exact %": f"**{m_ja['macro_num_exact_pct']:.2f}%**",
        "Overall Exact %": f"**{m_ja['macro_overall_exact_pct']:.2f}%**",
        "N/A Agree %": f"**{m_ja['macro_na_agree_pct']:.2f}%**"
    })
    df_ja = pd.DataFrame(j_rows)
    print(df_ja.to_string(index=False), flush=True)

    print("\n" + "=" * 105 + "\n", flush=True)

if __name__ == "__main__":
    run_study2_dual_judge_evaluation()
