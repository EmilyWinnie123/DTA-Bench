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
    build_final_p4_prompt
)

SHEETS_DIR = BASE_DIR / "annotation" / "sheets"
CACHE_DIR = BASE_DIR / "cache"

QWEN_PROGRESS_FILE = CACHE_DIR / "qwen_stage2_live_progress.json"
QWEN_PREDS_FILE = CACHE_DIR / "qwen_stage2_p4_predictions.json"
CLAUDE_PREDS_FILE = CACHE_DIR / "stage2_p4_predictions.json"
COMPARATIVE_SUMMARY_FILE = CACHE_DIR / "qwen_vs_claude_stage2_summary.json"

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("QWEN_API_KEY", "")
PRIMARY_MODEL = os.getenv("QWEN_MODEL", "Qwen3.8-Max")
API_URL = os.getenv("QWEN_API_URL", "https://novaapi.eduyuwo.com/v1/chat/completions")

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

def call_qwen_model_live(sys_prompt: str, user_prompt: str, max_retries: int = 12) -> tuple[str, dict]:
    """Calls fixed Qwen3.8-Max model via NovaAPI endpoint. Prohibits switching models."""
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
        "Content-Type": "application/json"
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=45)
            if r.status_code == 200:
                res = r.json()
                content = res["choices"][0]["message"]["content"]
                usage = res.get("usage", {})
                return content, usage
            elif r.status_code == 429:
                err_msg = f"HTTP 429 Rate Limit hit"
                last_err = err_msg
                print(f"    [Model {PRIMARY_MODEL} Attempt {attempt}/{max_retries}] {err_msg}, backing off...", flush=True)
                time.sleep(3 * attempt)
                continue
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

def load_qwen_progress():
    if QWEN_PROGRESS_FILE.exists():
        try:
            with open(QWEN_PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_qwen_progress_safe(progress):
    with file_lock:
        with open(QWEN_PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

def calc_icc21_safe(df_icc):
    if len(df_icc) < 3:
        return 0.0
    try:
        res = pg.intraclass_corr(data=df_icc, targets="Targets", raters="Raters", ratings="Scores")
        r21 = res[res["Type"] == "ICC(A,1)"]
        if not r21.empty:
            return float(r21["ICC"].iloc[0])
        r2 = res[res["Type"].str.contains("ICC2|ICC\\(2,1\\)", case=False, na=False)]
        if not r2.empty:
            return float(r2["ICC"].iloc[0])
        return float(res["ICC"].iloc[0])
    except Exception:
        return 0.0

def process_single_sample(s, raw_sha):
    sid = s["sample_id"]
    ds = s["dataset"]

    sys_prompt, user_prompt, meta = build_final_p4_prompt(s)

    raw_out, usage = call_qwen_model_live(sys_prompt, user_prompt)
    parsed = parse_json_from_response(raw_out)

    b = parse_val(parsed.get("bond", parsed.get("Bond", "NA")))
    g = parse_val(parsed.get("goal", parsed.get("Goal", "NA")))
    t = parse_val(parsed.get("task", parsed.get("Task", "NA")))

    out_item = {
        "sample_id": sid,
        "dataset": ds,
        "bond": b,
        "goal": g,
        "task": t,
        "raw_output": raw_out,
        "usage": usage
    }

    return sid, ds, b, g, t, out_item

def compute_pairwise_metrics(rater1_dict, rater2_dict, sample_ids, dims):
    metrics = {}
    for dim in dims:
        vals1 = [rater1_dict[sid][dim] for sid in sample_ids]
        vals2 = [rater2_dict[sid][dim] for sid in sample_ids]

        both_score, both_na = 0, 0
        paired1, paired2 = [], []

        for v1, v2 in zip(vals1, vals2):
            is_na1 = (v1 == "NA")
            is_na2 = (v2 == "NA")

            if not is_na1 and not is_na2:
                both_score += 1
                paired1.append(float(v1))
                paired2.append(float(v2))
            elif is_na1 and is_na2:
                both_na += 1

        total_n = len(sample_ids)
        valid_n = len(paired1)
        na_agree_count = both_score + both_na
        na_agree_pct = (na_agree_count / total_n) * 100.0

        if valid_n >= 3:
            qwk = float(cohen_kappa_score(paired1, paired2, weights="quadratic", labels=[1, 2, 3, 4, 5]))
            diffs = np.array(paired2) - np.array(paired1)
            abs_diffs = np.abs(diffs)
            mae = float(np.mean(abs_diffs))

            exact_numeric = int(np.sum(abs_diffs == 0))
            exact_agree_pct = ((exact_numeric + both_na) / total_n) * 100.0

            df_icc = pd.DataFrame({
                "Targets": list(range(valid_n)) + list(range(valid_n)),
                "Raters": ["Rater1"] * valid_n + ["Rater2"] * valid_n,
                "Scores": paired1 + paired2
            })
            icc_val = calc_icc21_safe(df_icc)
        else:
            qwk, icc_val, mae = 0.0, 0.0, 0.0
            exact_agree_pct = (both_na / total_n) * 100.0

        metrics[dim] = {
            "total_n": total_n,
            "valid_n": valid_n,
            "icc21": round(icc_val, 4),
            "qwk": round(qwk, 4),
            "mae": round(mae, 4),
            "exact_agree_pct": round(exact_agree_pct, 2),
            "na_agree_pct": round(na_agree_pct, 2)
        }

    macro_icc = round(float(np.mean([metrics[d]["icc21"] for d in dims])), 4)
    macro_qwk = round(float(np.mean([metrics[d]["qwk"] for d in dims])), 4)
    macro_mae = round(float(np.mean([metrics[d]["mae"] for d in dims])), 4)
    macro_exact = round(float(np.mean([metrics[d]["exact_agree_pct"] for d in dims])), 2)
    macro_na_agree = round(float(np.mean([metrics[d]["na_agree_pct"] for d in dims])), 2)

    return {
        "per_dimension": metrics,
        "macro_averages": {
            "macro_icc21": macro_icc,
            "macro_qwk": macro_qwk,
            "macro_mae": macro_mae,
            "exact_agree_pct": macro_exact,
            "na_agree_pct": macro_na_agree
        }
    }

def run_qwen_stage2_evaluation():
    print("=" * 100, flush=True)
    print(" STUDY 1 SECOND JUDGE — QWEN3.8-MAX LIVE EVALUATION EXECUTION (150 EVAL SAMPLES) ", flush=True)
    print("=" * 100, flush=True)

    # 1. PROVENANCE & DATASETS
    eval_gold_file = SHEETS_DIR / "gold_standard_eval_150.xlsx"
    assert eval_gold_file.exists(), f"CRITICAL ERROR: Human Gold Standard Excel missing at {eval_gold_file}"
    df_gold = pd.read_excel(eval_gold_file, keep_default_na=False)
    df_eval_150 = df_gold[df_gold["pool_type"] == "eval"].copy() if "pool_type" in df_gold.columns else df_gold.copy()
    assert len(df_eval_150) == 150, f"DATA PROVENANCE ERROR: Expected 150 eval samples, got {len(df_eval_150)}"

    file_mtime = time.ctime(os.path.getmtime(eval_gold_file))
    xls = pd.ExcelFile(eval_gold_file)
    sheet_name = xls.sheet_names[0]

    # Rubric Check
    config = load_judge_p4_config()
    rubric_text, raw_sha, text_sha = get_rubric_text_and_hash()

    print("\n--- DATA & ENVIRONMENT PROVENANCE ---", flush=True)
    print(f"  - Human Gold Standard File: {eval_gold_file.resolve()}", flush=True)
    print(f"  - File Last Modified:       {file_mtime}", flush=True)
    print(f"  - Sheet Name:               {sheet_name}", flush=True)
    print(f"  - Total Eval Samples:       {len(df_eval_150)} (ESConv={sum(df_eval_150['dataset']=='ESConv')}, AnnoMI={sum(df_eval_150['dataset']=='AnnoMI')})", flush=True)
    print(f"  - Second Judge Model ID:    {PRIMARY_MODEL}", flush=True)
    print(f"  - API Provider / Base URL:  NovaAPI (https://novaapi.eduyuwo.com/v1)", flush=True)
    print(f"  - API Key:                  {API_KEY[:14]}...", flush=True)
    print(f"  - Temperature / Top_p:      0.0 / 1.0", flush=True)
    print(f"  - Model Failover:           Disabled", flush=True)
    print(f"  - Frozen Rubric Raw SHA256: {raw_sha}", flush=True)

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

    progress = load_qwen_progress()
    qwen_predictions = {}
    pending_samples = []

    for s in eval_samples:
        sid = s["sample_id"]
        ds = s["dataset"]
        if sid in progress:
            res_item = progress[sid]
            b = parse_val(res_item.get("bond", "NA"))
            g = parse_val(res_item.get("goal", "NA"))
            t = parse_val(res_item.get("task", "NA"))
            qwen_predictions[sid] = {"dataset": ds, "Bond": b, "Goal": g, "Task": t}
        else:
            pending_samples.append(s)

    total_samples = len(eval_samples)
    already_done = len(qwen_predictions)
    print(f"\n--- EXECUTING LIVE P4 EVALUATION WITH QWEN3.8-MAX ({len(pending_samples)} PENDING / {already_done} CACHED) ---", flush=True)

    total_prompt_tokens = sum(item.get("usage", {}).get("prompt_tokens", 0) for item in progress.values())
    total_completion_tokens = sum(item.get("usage", {}).get("completion_tokens", 0) for item in progress.values())

    if pending_samples:
        completed_count = already_done
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_sample = {}
            for s in pending_samples:
                f = executor.submit(process_single_sample, s, raw_sha)
                future_to_sample[f] = s
                time.sleep(0.3)

            for future in as_completed(future_to_sample):
                s = future_to_sample[future]
                try:
                    sid, ds, b, g, t, out_item = future.result()
                    progress[sid] = out_item
                    save_qwen_progress_safe(progress)

                    usage = out_item.get("usage", {})
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)

                    qwen_predictions[sid] = {"dataset": ds, "Bond": b, "Goal": g, "Task": t}
                    completed_count += 1
                    print(f"  [{completed_count:03d}/{total_samples}] Done {sid:20s} ({ds:6s}) -> Bond={b}, Goal={g}, Task={t}", flush=True)
                except Exception as e:
                    print(f"  [ERROR] Sample {s['sample_id']} failed: {e}", flush=True)
                    raise e

    # Save Qwen predictions
    with open(QWEN_PREDS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Qwen predictions saved to {QWEN_PREDS_FILE}", flush=True)

    # 2. LOAD CLAUDE PREDICTIONS
    assert CLAUDE_PREDS_FILE.exists(), f"CRITICAL ERROR: Claude predictions missing at {CLAUDE_PREDS_FILE}"
    with open(CLAUDE_PREDS_FILE, "r", encoding="utf-8") as f:
        claude_raw_preds = json.load(f)

    claude_predictions = {}
    for sid, item in claude_raw_preds.items():
        claude_predictions[sid] = {
            "dataset": item["dataset"],
            "Bond": parse_val(item.get("bond", "NA")),
            "Goal": parse_val(item.get("goal", "NA")),
            "Task": parse_val(item.get("task", "NA"))
        }

    # 3. COMPUTE COMPARATIVE METRICS FOR 3 PAIRS across 3 SUBGROUPS
    subgroups = {
        "Combined": list(human_gold.keys()),
        "ESConv": [sid for sid, h in human_gold.items() if h["dataset"] == "ESConv"],
        "AnnoMI": [sid for sid, h in human_gold.items() if h["dataset"] == "AnnoMI"]
    }
    dims = ["Bond", "Goal", "Task"]

    comparative_results = {}

    for grp_name, sample_ids in subgroups.items():
        qwen_v_human = compute_pairwise_metrics(human_gold, qwen_predictions, sample_ids, dims)
        claude_v_human = compute_pairwise_metrics(human_gold, claude_predictions, sample_ids, dims)
        claude_v_qwen = compute_pairwise_metrics(claude_predictions, qwen_predictions, sample_ids, dims)

        comparative_results[grp_name] = {
            "Qwen_vs_Human": qwen_v_human,
            "Claude_vs_Human": claude_v_human,
            "Claude_vs_Qwen": claude_v_qwen
        }

    total_tokens = total_prompt_tokens + total_completion_tokens

    summary_output = {
        "provenance": {
            "gold_file": str(eval_gold_file.resolve()),
            "gold_file_mtime": file_mtime,
            "rubric_raw_sha256": raw_sha,
            "qwen_model_id": PRIMARY_MODEL,
            "claude_model_id": "[G]claude-sonnet-5",
            "api_endpoint": API_URL,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "plan_info": "2元日卡 (NovaAPI)"
        },
        "comparative_metrics": comparative_results
    }

    with open(COMPARATIVE_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2, ensure_ascii=False)
    print(f"[OK] Full comparative summary saved to {COMPARATIVE_SUMMARY_FILE}", flush=True)

    # 4. PRINT FORMATTED COMPARATIVE TABLES
    print("\n" + "=" * 115, flush=True)
    print(" STUDY 1 JUDGE AGREEMENT COMPARISON (QWEN3.8-MAX VS CLAUDE-SONNET-5 VS HUMAN GOLD) ", flush=True)
    print("=" * 115, flush=True)

    print(f"\n--- TOKEN USAGE & API COST SUMMARY ---", flush=True)
    print(f"  - Total Prompt Tokens:     {total_prompt_tokens:,}", flush=True)
    print(f"  - Total Completion Tokens: {total_completion_tokens:,}", flush=True)
    print(f"  - Total Combined Tokens:   {total_tokens:,}", flush=True)
    print(f"  - Subscription/API Cost:   Activated Plan '2元日卡' (Flat rate RMB 2.00 / day unlimited quota)", flush=True)

    for grp_name in ["Combined", "ESConv", "AnnoMI"]:
        print(f"\n======================================== SUBGROUP: {grp_name.upper()} ========================================", flush=True)
        table_rows = []
        grp_data = comparative_results[grp_name]

        for pair_key, pair_label in [
            ("Qwen_vs_Human", "Qwen vs Human Gold"),
            ("Claude_vs_Human", "Claude vs Human Gold"),
            ("Claude_vs_Qwen", "Claude vs Qwen (LLM-LLM)")
        ]:
            p_data = grp_data[pair_key]
            for dim in dims:
                d_info = p_data["per_dimension"][dim]
                table_rows.append({
                    "Comparison Pair": pair_label,
                    "Dimension": dim,
                    "Total N": d_info["total_n"],
                    "Valid N": d_info["valid_n"],
                    "ICC (2,1)": f"{d_info['icc21']:.4f}",
                    "QWK": f"{d_info['qwk']:.4f}",
                    "MAE": f"{d_info['mae']:.4f}",
                    "Exact Agree %": f"{d_info['exact_agree_pct']:.2f}%",
                    "N/A Agree %": f"{d_info['na_agree_pct']:.2f}%"
                })
            m_info = p_data["macro_averages"]
            table_rows.append({
                "Comparison Pair": f"**{pair_label}**",
                "Dimension": "**Macro Avg**",
                "Total N": p_data["per_dimension"]["Bond"]["total_n"],
                "Valid N": "-",
                "ICC (2,1)": f"**{m_info['macro_icc21']:.4f}**",
                "QWK": f"**{m_info['macro_qwk']:.4f}**",
                "MAE": f"**{m_info['macro_mae']:.4f}**",
                "Exact Agree %": f"**{m_info['exact_agree_pct']:.2f}%**",
                "N/A Agree %": f"**{m_info['na_agree_pct']:.2f}%**"
            })

        df_table = pd.DataFrame(table_rows)
        print(df_table.to_string(index=False), flush=True)

    print("\n" + "=" * 115 + "\n", flush=True)

if __name__ == "__main__":
    run_qwen_stage2_evaluation()
