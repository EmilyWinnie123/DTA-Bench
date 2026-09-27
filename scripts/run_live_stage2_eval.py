import os
import sys
import json
import time
import hashlib
import threading
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
    build_final_p4_prompt,
    format_dialogue_window
)

SHEETS_DIR = BASE_DIR / "annotation" / "sheets"
CACHE_DIR = BASE_DIR / "cache"
CACHE_FILE = CACHE_DIR / "llm_responses.jsonl"
STAGE2_PROGRESS_FILE = CACHE_DIR / "stage2_live_progress.json"
STAGE2_SUMMARY_FILE = CACHE_DIR / "stage2_evaluation_summary.json"
STAGE2_PREDS_FILE = CACHE_DIR / "stage2_p4_predictions.json"

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("CLAUDE_API_KEY", "")
PRIMARY_MODEL = os.getenv("CLAUDE_MODEL", "[G]claude-sonnet-5")
DIRECT_API_URL = os.getenv("CLAUDE_API_URL", "https://202.47.135.211/v1/chat/completions")

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

def call_judge_model_live(sys_prompt: str, user_prompt: str, max_retries: int = 8) -> str:
    """Calls the fixed Judge model [G]claude-sonnet-5 via direct IP proxy. Prohibits switching models."""
    payload = {
        "model": PRIMARY_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Host": "www.xiaoyaoapi.com"
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(DIRECT_API_URL, headers=headers, json=payload, verify=False, timeout=45)
            if r.status_code == 200:
                res = r.json()
                content = res["choices"][0]["message"]["content"]
                return content
            else:
                err_msg = f"HTTP {r.status_code}: {r.text[:120]}"
                last_err = err_msg
                print(f"    [Model {PRIMARY_MODEL} Attempt {attempt}/{max_retries}] {err_msg}", flush=True)
        except Exception as e:
            err_msg = f"Exception: {e}"
            last_err = err_msg
            print(f"    [Model {PRIMARY_MODEL} Attempt {attempt}/{max_retries}] {err_msg}", flush=True)

        time.sleep(2 * attempt)

    raise RuntimeError(f"CRITICAL API FAILURE: Failed after {max_retries} attempts on fixed model {PRIMARY_MODEL}. Last Error: {last_err}")

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

def load_stage2_progress():
    if STAGE2_PROGRESS_FILE.exists():
        try:
            with open(STAGE2_PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_stage2_progress_safe(progress):
    with file_lock:
        with open(STAGE2_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

def calc_icc_safe(df_icc):
    if len(df_icc) < 4:
        return 0.0
    try:
        res = pg.intraclass_corr(data=df_icc, targets="Targets", raters="Raters", ratings="Scores")
        icc_row = res[res["Type"].str.contains("ICC2k|ICC2|ICC\\(A,1\\)|ICC1|ICC\\(1,1\\)", case=False, na=False)]
        if not icc_row.empty:
            return float(icc_row["ICC"].iloc[0])
        return float(res["ICC"].iloc[0])
    except Exception:
        return 0.0

def process_single_sample(s, raw_sha):
    sid = s["sample_id"]
    ds = s["dataset"]

    # Build prompt using frozen P4 spec (Zero-shot across dimensions, NO gold standard answers)
    sys_prompt, user_prompt, meta = build_final_p4_prompt(s)

    raw_out = call_judge_model_live(sys_prompt, user_prompt)
    parsed = parse_json_from_response(raw_out)

    b = parse_val(parsed.get("bond", parsed.get("Bond", "NA")))
    g = parse_val(parsed.get("goal", parsed.get("Goal", "NA")))
    t = parse_val(parsed.get("task", parsed.get("Task", "NA")))

    out_item = {"sample_id": sid, "dataset": ds, "bond": b, "goal": g, "task": t, "raw_output": raw_out}

    record_cache = {
        "sample_id": sid,
        "dataset": ds,
        "stage": 2,
        "model": PRIMARY_MODEL,
        "condition": "p4_final",
        "run_idx": 0,
        "dimension": "all",
        "raw_output": raw_out,
        "parsed_output": {"reasoning": raw_out, "scores": {"Bond": b, "Goal": g, "Task": t}},
        "rubric_sha256": raw_sha,
        "estimated_cost": 0.003,
        "timestamp": time.time()
    }

    return sid, ds, b, g, t, out_item, record_cache

def run_stage2_evaluation():
    print("=" * 90, flush=True)
    print(" STAGE 2 — LIVE EVALUATION EXECUTION (150 EVAL SAMPLES - NEW KEY SK-9ZD) ", flush=True)
    print("=" * 90, flush=True)

    # 1. PROVENANCE & DATASET CHECK
    eval_gold_file = SHEETS_DIR / "gold_standard_eval_150.xlsx"
    if not eval_gold_file.exists():
        eval_gold_file = SHEETS_DIR / "gold_standard.xlsx"
    assert eval_gold_file.exists(), f"CRITICAL ERROR: Human Gold Standard Excel missing at {eval_gold_file}"

    df_gold = pd.read_excel(eval_gold_file, keep_default_na=False)
    if "pool_type" in df_gold.columns:
        df_eval_150 = df_gold[df_gold["pool_type"] == "eval"].copy()
        if len(df_eval_150) == 0:
            df_eval_150 = df_gold.copy()
    else:
        df_eval_150 = df_gold.copy()

    assert len(df_eval_150) == 150, f"DATA PROVENANCE ERROR: Expected exactly 150 eval samples, got {len(df_eval_150)}"

    file_mtime = time.ctime(os.path.getmtime(eval_gold_file))
    xls = pd.ExcelFile(eval_gold_file)
    sheet_name = xls.sheet_names[0]
    cell_a1 = df_gold.columns[0]

    # Verify Frozen Rubric
    config = load_judge_p4_config()
    rubric_text, raw_sha, text_sha = get_rubric_text_and_hash()

    print("\n--- DATA & ENVIRONMENT PROVENANCE ---", flush=True)
    print(f"  - Human Gold Standard File: {eval_gold_file.resolve()}", flush=True)
    print(f"  - File Last Modified:       {file_mtime}", flush=True)
    print(f"  - Worksheet Name:           {sheet_name}", flush=True)
    print(f"  - A1 Cell Header:           '{cell_a1}'", flush=True)
    print(f"  - Total Eval Samples:       {len(df_eval_150)}", flush=True)
    print(f"  - Dataset Breakdown:        ESConv={sum(df_eval_150['dataset']=='ESConv')}, AnnoMI={sum(df_eval_150['dataset']=='AnnoMI')}", flush=True)
    print(f"  - Frozen Judge Model:       {PRIMARY_MODEL}", flush=True)
    print(f"  - Direct API URL:           {DIRECT_API_URL}", flush=True)
    print(f"  - Active API Key:           {API_KEY[:14]}...", flush=True)
    print(f"  - Temperature:              0.0", flush=True)
    print(f"  - Model Switch on Failure:  {config['model_config']['allow_model_switch']} (Disabled)", flush=True)
    print(f"  - Frozen Rubric Raw SHA256: {raw_sha}", flush=True)

    # Build evaluation list
    eval_samples = []
    human_gold = {}
    for idx, r in df_eval_150.iterrows():
        sid = str(r["sample_id"]).strip()
        ds = str(r["dataset"]).strip()
        ctx = str(r["dialogue_context"]).strip()
        su = str(r["seeker_utterance"]).strip()
        sr = str(r["supporter_response"]).strip()

        eval_samples.append({
            "sample_id": sid,
            "dataset": ds,
            "dialogue_context": ctx,
            "seeker_utterance": su,
            "supporter_response": sr
        })
        human_gold[sid] = {
            "dataset": ds,
            "Bond": parse_val(r["Bond (1-5/NA)"]),
            "Goal": parse_val(r["Goal (1-5/NA)"]),
            "Task": parse_val(r["Task (1-5/NA)"])
        }

    progress = load_stage2_progress()
    llm_predictions = {}
    pending_samples = []

    for s in eval_samples:
        sid = s["sample_id"]
        ds = s["dataset"]
        if sid in progress:
            res_item = progress[sid]
            b = parse_val(res_item.get("bond", "NA"))
            g = parse_val(res_item.get("goal", "NA"))
            t = parse_val(res_item.get("task", "NA"))
            llm_predictions[sid] = {"dataset": ds, "Bond": b, "Goal": g, "Task": t}
        else:
            pending_samples.append(s)

    total_samples = len(eval_samples)
    already_done = len(llm_predictions)
    print(f"\n--- EXECUTING LIVE P4 EVALUATION ON {len(pending_samples)} PENDING SAMPLES (Cached: {already_done}/{total_samples}) ---", flush=True)

    if pending_samples:
        completed_count = already_done
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_sample = {}
            for s in pending_samples:
                f = executor.submit(process_single_sample, s, raw_sha)
                future_to_sample[f] = s
                time.sleep(0.1)

            for future in as_completed(future_to_sample):
                s = future_to_sample[future]
                try:
                    sid, ds, b, g, t, out_item, record_cache = future.result()
                    progress[sid] = out_item
                    save_stage2_progress_safe(progress)

                    with file_lock:
                        with open(CACHE_FILE, "a", encoding="utf-8") as f_cache:
                            f_cache.write(json.dumps(record_cache, ensure_ascii=False) + "\n")

                    llm_predictions[sid] = {"dataset": ds, "Bond": b, "Goal": g, "Task": t}
                    completed_count += 1
                    print(f"  [{completed_count:03d}/{total_samples}] Done {sid:20s} ({ds:6s}) -> Bond={b}, Goal={g}, Task={t}", flush=True)

                except Exception as e:
                    print(f"  [ERROR] Sample {s['sample_id']} failed: {e}", flush=True)
                    raise e

    # Save raw predictions JSON
    with open(STAGE2_PREDS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Raw predictions saved to {STAGE2_PREDS_FILE}", flush=True)

    # 2. CALCULATE EVALUATION METRICS BY SUBGROUPS (Combined, ESConv, AnnoMI)
    subgroups = {
        "Combined": list(human_gold.keys()),
        "ESConv": [sid for sid, h in human_gold.items() if h["dataset"] == "ESConv"],
        "AnnoMI": [sid for sid, h in human_gold.items() if h["dataset"] == "AnnoMI"]
    }

    dims = ["Bond", "Goal", "Task"]
    group_results = {}

    for grp_name, sample_ids in subgroups.items():
        grp_metrics = {}
        for dim in dims:
            h_vals = [human_gold[sid][dim] for sid in sample_ids]
            m_vals = [llm_predictions[sid][dim] for sid in sample_ids]

            both_score, both_na, h_na_m_score, h_score_m_na = 0, 0, 0, 0
            paired_h, paired_m = [], []

            for h_v, m_v in zip(h_vals, m_vals):
                h_is_na = (h_v == "NA")
                m_is_na = (m_v == "NA")

                if not h_is_na and not m_is_na:
                    both_score += 1
                    paired_h.append(float(h_v))
                    paired_m.append(float(m_v))
                elif h_is_na and m_is_na:
                    both_na += 1
                elif h_is_na and not m_is_na:
                    h_na_m_score += 1
                elif not h_is_na and m_is_na:
                    h_score_m_na += 1

            total_n = len(sample_ids)
            valid_n = len(paired_h)
            na_agree_count = both_score + both_na
            na_agree_pct = (na_agree_count / total_n) * 100.0

            if valid_n >= 3:
                qwk = float(cohen_kappa_score(paired_h, paired_m, weights="quadratic", labels=[1, 2, 3, 4, 5]))
                diffs = np.array(paired_m) - np.array(paired_h)
                abs_diffs = np.abs(diffs)
                mae = float(np.mean(abs_diffs))
                mean_diff = float(np.mean(diffs))

                exact_numeric = int(np.sum(abs_diffs == 0))
                exact_agree_pct = ((exact_numeric + both_na) / total_n) * 100.0

                df_icc = pd.DataFrame({
                    "Targets": list(range(valid_n)) + list(range(valid_n)),
                    "Raters": ["Human"] * valid_n + ["LLM"] * valid_n,
                    "Scores": paired_h + paired_m
                })
                icc_val = calc_icc_safe(df_icc)
            else:
                qwk, icc_val, mae, mean_diff = 0.0, 0.0, 0.0, 0.0
                exact_agree_pct = (both_na / total_n) * 100.0

            grp_metrics[dim] = {
                "total_n": total_n,
                "valid_n": valid_n,
                "icc": round(icc_val, 4),
                "qwk": round(qwk, 4),
                "mae": round(mae, 4),
                "exact_agree_pct": round(exact_agree_pct, 2),
                "na_agree_pct": round(na_agree_pct, 2)
            }

        macro_icc = round(float(np.mean([grp_metrics[d]["icc"] for d in dims])), 4)
        macro_qwk = round(float(np.mean([grp_metrics[d]["qwk"] for d in dims])), 4)
        macro_mae = round(float(np.mean([grp_metrics[d]["mae"] for d in dims])), 4)
        macro_exact = round(float(np.mean([grp_metrics[d]["exact_agree_pct"] for d in dims])), 2)
        macro_na_agree = round(float(np.mean([grp_metrics[d]["na_agree_pct"] for d in dims])), 2)

        group_results[grp_name] = {
            "per_dimension": grp_metrics,
            "macro_averages": {
                "macro_icc": macro_icc,
                "macro_qwk": macro_qwk,
                "macro_mae": macro_mae,
                "exact_agree_pct": macro_exact,
                "na_agree_pct": macro_na_agree
            }
        }

    # Save summary results JSON
    with open(STAGE2_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(group_results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Summary metrics saved to {STAGE2_SUMMARY_FILE}", flush=True)

    # 3. PRINT FORMATTED RESULTS TABLES
    print("\n" + "=" * 100, flush=True)
    print(" STAGE 2 — EVALUATION RESULTS TABLE (COMBINED, ESCONV, ANNOMI) ", flush=True)
    print("=" * 100, flush=True)

    rows = []
    for grp_name, grp_data in group_results.items():
        for dim in dims:
            d_info = grp_data["per_dimension"][dim]
            rows.append({
                "Group": grp_name,
                "Dimension": dim,
                "Total N": d_info["total_n"],
                "Valid N": d_info["valid_n"],
                "ICC (2,1)": f"{d_info['icc']:.4f}",
                "QWK": f"{d_info['qwk']:.4f}",
                "MAE": f"{d_info['mae']:.4f}",
                "Exact Agree %": f"{d_info['exact_agree_pct']:.2f}%",
                "N/A Agree %": f"{d_info['na_agree_pct']:.2f}%"
            })
        m_info = grp_data["macro_averages"]
        rows.append({
            "Group": f"**{grp_name} (Macro Avg)**",
            "Dimension": "Average",
            "Total N": group_results[grp_name]["per_dimension"]["Bond"]["total_n"],
            "Valid N": "-",
            "ICC (2,1)": f"**{m_info['macro_icc']:.4f}**",
            "QWK": f"**{m_info['macro_qwk']:.4f}**",
            "MAE": f"**{m_info['macro_mae']:.4f}**",
            "Exact Agree %": f"**{m_info['exact_agree_pct']:.2f}%**",
            "N/A Agree %": f"**{m_info['na_agree_pct']:.2f}%**"
        })

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False), flush=True)

    print("=" * 100 + "\n", flush=True)

if __name__ == "__main__":
    run_stage2_evaluation()
