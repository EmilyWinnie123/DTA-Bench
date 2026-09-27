import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import pingouin as pg

BASE_DIR = Path(__file__).resolve().parent.parent
SHEETS_DIR = BASE_DIR / "annotation" / "sheets"
ANNOTATION_DIR = BASE_DIR / "annotation"
CACHE_DIR = BASE_DIR / "cache"
CACHE_FILE = CACHE_DIR / "llm_responses.jsonl"

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

def calc_icc_safe(df_icc):
    if len(df_icc) < 4:
        return 0.0
    try:
        res = pg.intraclass_corr(data=df_icc, targets="Targets", raters="Raters", ratings="Scores")
        icc_row = res[res["Type"].str.contains("ICC1|ICC\\(1,1\\)|ICC2k|ICC\\(A,1\\)", case=False, na=False)]
        if not icc_row.empty:
            return float(icc_row["ICC"].iloc[0])
        return float(res["ICC"].iloc[0])
    except Exception:
        return 0.0

def run_stage1_prompt_ablation():
    print("=" * 80)
    print(" STAGE 1 — PROMPT ABLATION EXECUTION & EVALUATION ")
    print("=" * 80)

    # 1. DATA PROVENANCE CHECK
    gold_file = SHEETS_DIR / "gold_standard_dev_20.xlsx"
    if not gold_file.exists():
        gold_file = SHEETS_DIR / "gold_standard.xlsx"

    if not gold_file.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: Human Gold Standard Excel missing at {gold_file}")

    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: LLM responses cache missing at {CACHE_FILE}")

    # Load Gold Standard (20 dev items)
    df_gold_raw = pd.read_excel(gold_file, keep_default_na=False)
    if "pool_type" in df_gold_raw.columns:
        df_gold_dev = df_gold_raw[df_gold_raw["pool_type"] == "dev"].copy()
    else:
        df_gold_dev = df_gold_raw.copy()

    if len(df_gold_dev) != 20:
        # Fallback to dev_20 explicit sheet if present
        gold_dev_20 = SHEETS_DIR / "gold_standard_dev_20.xlsx"
        if gold_dev_20.exists():
            df_gold_dev = pd.read_excel(gold_dev_20, keep_default_na=False)

    assert len(df_gold_dev) == 20, f"DATA PROVENANCE ERROR: Expected exactly 20 development samples, got {len(df_gold_dev)}"

    # Get Excel provenance info
    gold_mtime = time.ctime(os.path.getmtime(gold_file))
    xls = pd.ExcelFile(gold_file)
    first_sheet_name = xls.sheet_names[0]
    cell_a1_val = df_gold_raw.columns[0]
    llm_mtime = time.ctime(os.path.getmtime(CACHE_FILE))

    print("\n--- DATA PROVENANCE ---")
    print(f"  - Human Gold Standard File Path: {gold_file.resolve()}")
    print(f"  - Human Gold Standard Last Modified: {gold_mtime}")
    print(f"  - First Worksheet Name: {first_sheet_name}")
    print(f"  - A1 Cell Header: '{cell_a1_val}'")
    print(f"  - LLM Result File Path: {CACHE_FILE.resolve()}")
    print(f"  - LLM Result File Last Modified: {llm_mtime}")

    dev_sample_ids = [str(s).strip() for s in df_gold_dev["sample_id"]]
    assert len(set(dev_sample_ids)) == 20, "DATA PROVENANCE ERROR: Duplicate sample_ids found in development set!"

    # Parse Human Gold Ratings for 20 dev items
    human_gold = {}
    for idx, r in df_gold_dev.iterrows():
        sid = str(r["sample_id"]).strip()
        human_gold[sid] = {
            "Bond": parse_val(r["Bond (1-5/NA)"]),
            "Goal": parse_val(r["Goal (1-5/NA)"]),
            "Task": parse_val(r["Task (1-5/NA)"])
        }

    # Load LLM Evaluation Cache
    llm_records = []
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sid = str(item.get("sample_id")).strip()
                if sid in human_gold:
                    llm_records.append(item)

    print(f"\n[OK] Loaded {len(llm_records)} matching LLM records for the 20 development samples.")

    # 2. SAVE RAW RESULTS FOR THE 4 PROMPT CONDITIONS
    # Map condition names: P1=zero_shot, P2=cot_then_score (structured), P3=few_shot, P4=dimension_wise
    prompt_mapping = {
        "P1 Zero-shot": ("zero_shot", CACHE_DIR / "stage1_p1_zero_shot.json"),
        "P2 Structured": ("cot_then_score", CACHE_DIR / "stage1_p2_structured.json"),
        "P3 Few-shot": ("few_shot", CACHE_DIR / "stage1_p3_few_shot.json"),
        "P4 Dimension-wise": ("dimension_wise", CACHE_DIR / "stage1_p4_dimensionwise.json")
    }

    prompt_predictions = {}

    for p_name, (cond_key, json_out_path) in prompt_mapping.items():
        cond_recs = [r for r in llm_records if r.get("condition") == cond_key]
        raw_list = []

        # Deduplicate/aggregate by sample_id
        pred_map = {}
        for sid in dev_sample_ids:
            s_recs = [r for r in cond_recs if str(r.get("sample_id")).strip() == sid]
            
            b_val, g_val, t_val = "NA", "NA", "NA"
            if s_recs:
                # Average / majority voted score across runs & models
                scores_b, scores_g, scores_t = [], [], []
                for sr in s_recs:
                    parsed = sr.get("parsed_output", {})
                    if isinstance(parsed, dict) and "scores" in parsed:
                        sc = parsed["scores"]
                        b = parse_val(sc.get("Bond"))
                        g = parse_val(sc.get("Goal"))
                        t = parse_val(sc.get("Task"))
                        if b != "NA": scores_b.append(float(b))
                        if g != "NA": scores_g.append(float(g))
                        if t != "NA": scores_t.append(float(t))

                b_val = str(int(round(np.mean(scores_b)))) if scores_b else "NA"
                g_val = str(int(round(np.mean(scores_g)))) if scores_g else "NA"
                t_val = str(int(round(np.mean(scores_t)))) if scores_t else "NA"

            item_dict = {
                "sample_id": sid,
                "bond": int(b_val) if b_val != "NA" else "NA",
                "goal": int(g_val) if g_val != "NA" else "NA",
                "task": int(t_val) if t_val != "NA" else "NA"
            }
            raw_list.append(item_dict)
            pred_map[sid] = {
                "Bond": b_val,
                "Goal": g_val,
                "Task": t_val
            }

        prompt_predictions[p_name] = pred_map

        # Save raw JSON file
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(raw_list, f, indent=2, ensure_ascii=False)
        print(f"  - Saved raw predictions: {p_name:20s} -> {json_out_path.name}")

    # 3. CALCULATE METRICS PER PROMPT & PER DIMENSION
    dims = ["Bond", "Goal", "Task"]
    summary_results = {}
    dimension_detail_results = []

    for p_name, pred_map in prompt_predictions.items():
        dim_metrics = {}

        for dim in dims:
            h_vals = [human_gold[sid][dim] for sid in dev_sample_ids]
            m_vals = [pred_map[sid][dim] for sid in dev_sample_ids]

            # N/A Decision Metrics
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

            total_n = len(dev_sample_ids)
            valid_n = len(paired_h)
            na_excl_pct = ((total_n - valid_n) / total_n) * 100.0
            na_agree_pct = ((both_score + both_na) / total_n) * 100.0

            # Paired numeric metrics
            if valid_n >= 3:
                qwk = float(cohen_kappa_score(paired_h, paired_m, weights="quadratic", labels=[1, 2, 3, 4, 5]))
                diffs = np.array(paired_m) - np.array(paired_h)
                abs_diffs = np.abs(diffs)
                mae = float(np.mean(abs_diffs))
                mean_diff = float(np.mean(diffs))

                diff_0_pct = float(np.mean(abs_diffs == 0)) * 100.0
                diff_1_pct = float(np.mean(abs_diffs == 1)) * 100.0
                diff_ge2_pct = float(np.mean(abs_diffs >= 2)) * 100.0
                exact_agree_pct = diff_0_pct

                # ICC(2,1)
                df_icc = pd.DataFrame({
                    "Targets": list(range(valid_n)) + list(range(valid_n)),
                    "Raters": ["Human"] * valid_n + ["LLM"] * valid_n,
                    "Scores": paired_h + paired_m
                })
                icc_val = calc_icc_safe(df_icc)
            else:
                qwk, icc_val, mae, mean_diff = 0.0, 0.0, 0.0, 0.0
                diff_0_pct, diff_1_pct, diff_ge2_pct, exact_agree_pct = 0.0, 0.0, 0.0, 0.0

            dim_metrics[dim] = {
                "valid_n": valid_n,
                "na_excl_pct": round(na_excl_pct, 2),
                "na_agree_pct": round(na_agree_pct, 2),
                "qwk": round(qwk, 4),
                "icc": round(icc_val, 4),
                "exact_agree_pct": round(exact_agree_pct, 2),
                "mae": round(mae, 4),
                "mean_diff": round(mean_diff, 4),
                "diff_0_pct": round(diff_0_pct, 2),
                "diff_1_pct": round(diff_1_pct, 2),
                "diff_ge2_pct": round(diff_ge2_pct, 2)
            }

            dimension_detail_results.append({
                "Prompt": p_name,
                "Dimension": dim,
                "Valid n": valid_n,
                "N/A excluded %": round(na_excl_pct, 2),
                "QWK": round(qwk, 4),
                "ICC": round(icc_val, 4),
                "Exact Agreement %": round(exact_agree_pct, 2),
                "MAE": round(mae, 4),
                "Mean Difference": round(mean_diff, 4),
                "±0 %": round(diff_0_pct, 2),
                "±1 %": round(diff_1_pct, 2),
                "≥2 %": round(diff_ge2_pct, 2)
            })

        # Calculate Macro Averages
        macro_qwk = round(float(np.mean([dim_metrics[d]["qwk"] for d in dims])), 4)
        mean_mae = round(float(np.mean([dim_metrics[d]["mae"] for d in dims])), 4)
        mean_exact = round(float(np.mean([dim_metrics[d]["exact_agree_pct"] for d in dims])), 2)
        mean_na_agree = round(float(np.mean([dim_metrics[d]["na_agree_pct"] for d in dims])), 2)

        summary_results[p_name] = {
            "Bond QWK": dim_metrics["Bond"]["qwk"],
            "Goal QWK": dim_metrics["Goal"]["qwk"],
            "Task QWK": dim_metrics["Task"]["qwk"],
            "Macro QWK": macro_qwk,
            "Bond ICC": dim_metrics["Bond"]["icc"],
            "Goal ICC": dim_metrics["Goal"]["icc"],
            "Task ICC": dim_metrics["Task"]["icc"],
            "Mean MAE": mean_mae,
            "Exact Agreement": mean_exact,
            "N/A Agreement": mean_na_agree
        }

    # 4. PRINT REQUIRED STAGE 1 SUMMARY TABLE
    print("\n" + "=" * 100)
    print(" REQUIRED STAGE 1 SUMMARY TABLE ")
    print("=" * 100)
    df_summary = pd.DataFrame.from_dict(summary_results, orient="index")
    print(df_summary.to_string())

    # 5. PRINT DIMENSION-LEVEL DETAIL TABLE
    print("\n" + "=" * 110)
    print(" DIMENSION-LEVEL DETAIL TABLE ")
    print("=" * 110)
    df_detail = pd.DataFrame(dimension_detail_results)
    print(df_detail.to_string(index=False))

    # 6. SELECTION RULE EXECUTOR (Primary Metric: Macro QWK)
    # Sort candidates by: Macro QWK (desc), Mean MAE (asc), N/A Agreement (desc), Exact Agreement (desc)
    best_p_name = max(
        summary_results.keys(),
        key=lambda k: (
            summary_results[k]["Macro QWK"],
            -summary_results[k]["Mean MAE"],
            summary_results[k]["N/A Agreement"],
            summary_results[k]["Exact Agreement"]
        )
    )

    best_res = summary_results[best_p_name]
    p_code = best_p_name.split()[0]  # P1, P2, P3, or P4

    print("\n" + "=" * 80)
    print(" FINAL STAGE 1 OUTPUT ")
    print("=" * 80)
    print(f"Selected Prompt: {p_code}")
    print("\nSelection basis:")
    print(f"* Macro QWK:     {best_res['Macro QWK']:.4f}")
    print(f"* Bond QWK:      {best_res['Bond QWK']:.4f}")
    print(f"* Goal QWK:      {best_res['Goal QWK']:.4f}")
    print(f"* Task QWK:      {best_res['Task QWK']:.4f}")
    print(f"* Mean MAE:      {best_res['Mean MAE']:.4f}")
    print(f"* N/A Agreement: {best_res['N/A Agreement']:.2f}%")
    print("\n\"The selected prompting strategy was determined exclusively using the predefined development set. The 150-item evaluation set remained untouched during prompt selection.\"")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_stage1_prompt_ablation()
