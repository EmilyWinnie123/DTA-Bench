import json
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = BASE_DIR / "annotation"
SHEETS_DIR = ANNOTATION_DIR / "sheets"

def get_latest_file(default_name: str, alt_names: list) -> Path:
    candidates = [SHEETS_DIR / default_name] + [SHEETS_DIR / alt for alt in alt_names]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(f"No valid sheet found for {default_name}")
    return max(existing, key=lambda p: p.stat().st_mtime)

FILE_A = get_latest_file("annotator_A.xlsx", ["annotator_A_scored.xlsx", "annotator_A_DONE.xlsx"])
FILE_B = get_latest_file("annotator_B.xlsx", ["annotator_B_DONE.xlsx", "annotator_B_scored.xlsx"])
REPORT_JSON = ANNOTATION_DIR / "round1_agreement_report.json"

def parse_score(val):
    if pd.isna(val):
        return "NA"
    val_str = str(val).strip().upper()
    if val_str in ["NA", "N/A", "NONE", "NULL"]:
        return "NA"
    try:
        num = float(val_str)
        if num.is_integer():
            int_v = int(num)
            if 1 <= int_v <= 5:
                return int_v
    except ValueError:
        pass
    return val_str

def extract_icc2k_safe(icc_res: pd.DataFrame) -> tuple[float, float, float]:
    if icc_res.empty:
        return 0.0, 0.0, 0.0

    types = icc_res["Type"].astype(str).tolist()
    target_row = None
    for t in types:
        if "icc2k" in t.lower() or "icc(a,k)" in t.lower() or "icc(2,k)" in t.lower() or "icc2" in t.lower():
            target_row = icc_res[icc_res["Type"] == t].iloc[0]
            break

    if target_row is None:
        target_row = icc_res.iloc[-1]

    icc_val = float(target_row["ICC"])

    ci_col = [c for c in icc_res.columns if "CI" in c]
    if ci_col:
        ci_val = target_row[ci_col[0]]
        if isinstance(ci_val, (list, tuple, np.ndarray)) and len(ci_val) == 2:
            ci_lower, ci_upper = float(ci_val[0]), float(ci_val[1])
        else:
            ci_lower, ci_upper = icc_val, icc_val
    else:
        ci_lower, ci_upper = icc_val, icc_val

    return icc_val, ci_lower, ci_upper

def run_round1_agreement_analysis():
    print("=" * 80)
    print(" DIGITAL THERAPEUTIC ALLIANCE (DTA) ROUND 1 CALIBRATION AGREEMENT REPORT ")
    print("=" * 80)
    print(f"[INPUT FILE A] {FILE_A.name} (Last modified: {FILE_A.stat().st_mtime})")
    print(f"[INPUT FILE B] {FILE_B.name} (Last modified: {FILE_B.stat().st_mtime})")

    # ---------------------------------------------------------
    # Section 1: Pre-checks
    # ---------------------------------------------------------
    if not FILE_A.exists() or not FILE_B.exists():
        raise FileNotFoundError(f"[ERROR] Missing scored sheets: {FILE_A} or {FILE_B}")

    df_A_raw = pd.read_excel(FILE_A)
    df_B_raw = pd.read_excel(FILE_B)

    if len(df_A_raw) != 20:
        raise ValueError(f"[PRE-CHECK FAILED] Sheet A has {len(df_A_raw)} rows, expected exactly 20.")
    if len(df_B_raw) != 20:
        raise ValueError(f"[PRE-CHECK FAILED] Sheet B has {len(df_B_raw)} rows, expected exactly 20.")

    dims = ["Bond", "Goal", "Task"]

    for name, df in [("Annotator_A", df_A_raw), ("Annotator_B", df_B_raw)]:
        for dim in dims:
            if dim not in df.columns:
                raise KeyError(f"[PRE-CHECK FAILED] Column '{dim}' missing in {name}")
            for idx, val in enumerate(df[dim]):
                parsed = parse_score(val)
                if parsed == "" or parsed is None:
                    raise ValueError(f"[PRE-CHECK FAILED] Blank cell in {name} row {idx+1} column '{dim}'")

    set_A = set(df_A_raw["sample_id"].astype(str))
    set_B = set(df_B_raw["sample_id"].astype(str))
    if set_A != set_B:
        raise ValueError(f"[PRE-CHECK FAILED] sample_id set mismatch! A-B: {set_A - set_B}, B-A: {set_B - set_A}")

    valid_set = {1, 2, 3, 4, 5, "NA"}
    for name, df in [("Annotator_A", df_A_raw), ("Annotator_B", df_B_raw)]:
        for dim in dims:
            for idx, val in enumerate(df[dim]):
                parsed = parse_score(val)
                if parsed not in valid_set:
                    raise ValueError(f"[PRE-CHECK FAILED] Invalid score in {name} row {idx+1} '{dim}': {val} (parsed: {parsed})")

    for name, df in [("Annotator_A", df_A_raw), ("Annotator_B", df_B_raw)]:
        mock_cnt = df["dialogue_context"].astype(str).str.contains(r"\[MOCK\]", regex=True, case=False).sum()
        if mock_cnt > 0:
            raise ValueError(f"[PRE-CHECK FAILED] {name} dialogue_context contains {mock_cnt} [MOCK] tags! Must use real data.")

    print("[PRE-CHECKS PASSED] All 4 pre-check validation rules passed successfully!")

    merged = pd.merge(df_A_raw, df_B_raw, on="sample_id", suffixes=("_A", "_B"))

    # ---------------------------------------------------------
    # Section 2 & 3: NA Handling & Independent Dimension Metrics
    # ---------------------------------------------------------
    dimension_metrics = {}
    na_handling_summary = {}

    for dim in dims:
        col_A = f"{dim}_A"
        col_B = f"{dim}_B"

        parsed_A = [parse_score(v) for v in merged[col_A]]
        parsed_B = [parse_score(v) for v in merged[col_B]]

        valid_pairs = []
        excluded_ids = []

        for idx, row in merged.iterrows():
            sid = row["sample_id"]
            vA = parsed_A[idx]
            vB = parsed_B[idx]

            if vA == "NA" or vB == "NA":
                excluded_ids.append(sid)
            else:
                valid_pairs.append({
                    "sample_id": sid,
                    "score_A": int(vA),
                    "score_B": int(vB)
                })

        n_valid = len(valid_pairs)
        n_excluded = len(excluded_ids)

        na_handling_summary[dim] = {
            "valid_count": n_valid,
            "excluded_count": n_excluded,
            "excluded_sample_ids": excluded_ids
        }

        scores_A = np.array([p["score_A"] for p in valid_pairs])
        scores_B = np.array([p["score_B"] for p in valid_pairs])

        icc_df = pd.DataFrame({
            "Targets": list(range(n_valid)) * 2,
            "Raters": ["Annotator_A"] * n_valid + ["Annotator_B"] * n_valid,
            "Scores": list(scores_A) + list(scores_B)
        })
        icc_res = pg.intraclass_corr(data=icc_df, targets="Targets", raters="Raters", ratings="Scores")
        icc2k_val, ci_lower, ci_upper = extract_icc2k_safe(icc_res)

        qwk = float(cohen_kappa_score(scores_A, scores_B, labels=[1, 2, 3, 4, 5], weights="quadratic"))

        exact_match = float(np.mean(scores_A == scores_B))

        abs_diffs = np.abs(scores_A - scores_B)
        adjacent_match = float(np.mean(abs_diffs <= 1))

        mae = float(np.mean(abs_diffs))

        diff_dist = {int(d): int(np.sum(abs_diffs == d)) for d in range(5)}

        verdict = "PASSED" if icc2k_val >= 0.75 else "FAILED"

        dimension_metrics[dim] = {
            "ICC2k": {
                "val": round(icc2k_val, 4),
                "ci95_lower": round(ci_lower, 4),
                "ci95_upper": round(ci_upper, 4),
                "ci95_str": f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            },
            "QWK": round(qwk, 4),
            "exact_agreement_rate": round(exact_match, 4),
            "adjacent_agreement_rate": round(adjacent_match, 4),
            "mae": round(mae, 4),
            "difference_distribution": diff_dist,
            "verdict": verdict
        }

    # ---------------------------------------------------------
    # Section 4: Discrepancy List (|diff| >= 2)
    # ---------------------------------------------------------
    discrepancy_list = []

    for idx, row in merged.iterrows():
        sid = row["sample_id"]
        ds = row.get("dataset_A", row.get("dataset_B", "ESConv"))
        ctx = row["dialogue_context_A"]
        resp = row["supporter_response_A"]

        scores_map_A = {d: parse_score(row[f"{d}_A"]) for d in dims}
        scores_map_B = {d: parse_score(row[f"{d}_B"]) for d in dims}

        diffs = {}
        for d in dims:
            vA = scores_map_A[d]
            vB = scores_map_B[d]
            if isinstance(vA, int) and isinstance(vB, int):
                diffs[d] = abs(vA - vB)

        if diffs:
            max_d_dim = max(diffs.keys(), key=lambda k: diffs[k])
            max_d_val = diffs[max_d_dim]
        else:
            max_d_dim = "None"
            max_d_val = 0

        if max_d_val >= 2:
            discrepancy_list.append({
                "sample_id": sid,
                "dataset": ds,
                "dialogue_context": str(ctx),
                "supporter_response": str(resp),
                "scores_A": scores_map_A,
                "scores_B": scores_map_B,
                "max_diff_dimension": max_d_dim,
                "max_difference": max_d_val
            })

    discrepancy_list.sort(key=lambda x: (-x["max_difference"], x["sample_id"]))

    # ---------------------------------------------------------
    # Section 5: Distribution Comparison
    # ---------------------------------------------------------
    distribution_comparison = {}

    for dim in dims:
        parsed_A = [parse_score(v) for v in merged[f"{dim}_A"]]
        parsed_B = [parse_score(v) for v in merged[f"{dim}_B"]]

        cnt_A = {str(k): parsed_A.count(k) for k in [1, 2, 3, 4, 5, "NA"]}
        cnt_B = {str(k): parsed_B.count(k) for k in [1, 2, 3, 4, 5, "NA"]}

        num_A = [v for v in parsed_A if isinstance(v, int)]
        num_B = [v for v in parsed_B if isinstance(v, int)]

        mean_A = float(np.mean(num_A)) if num_A else 0.0
        mean_B = float(np.mean(num_B)) if num_B else 0.0
        mean_diff = mean_B - mean_A

        distribution_comparison[dim] = {
            "Annotator_A_counts": cnt_A,
            "Annotator_B_counts": cnt_B,
            "Annotator_A_mean": round(mean_A, 4),
            "Annotator_B_mean": round(mean_B, 4),
            "mean_difference_B_minus_A": round(mean_diff, 4)
        }

    # ---------------------------------------------------------
    # Section 6: NA Discrepancy Specialized Analysis
    # ---------------------------------------------------------
    na_discrepancies = {}

    for dim in dims:
        parsed_A = [parse_score(v) for v in merged[f"{dim}_A"]]
        parsed_B = [parse_score(v) for v in merged[f"{dim}_B"]]

        items_na_disc = []
        for idx, row in merged.iterrows():
            sid = row["sample_id"]
            vA = parsed_A[idx]
            vB = parsed_B[idx]

            is_A_na = (vA == "NA")
            is_B_na = (vB == "NA")

            if (is_A_na and not is_B_na) or (is_B_na and not is_A_na):
                items_na_disc.append({
                    "sample_id": sid,
                    "score_A": vA,
                    "score_B": vB
                })

        na_discrepancies[dim] = {
            "discrepancy_count": len(items_na_disc),
            "items": items_na_disc
        }

    # ---------------------------------------------------------
    # Section 7 & 8: Save JSON & Full Console Printout
    # ---------------------------------------------------------
    report_data = {
        "na_handling_summary": na_handling_summary,
        "dimension_metrics": dimension_metrics,
        "discrepancies_diff_ge_2": discrepancy_list,
        "distribution_comparison": distribution_comparison,
        "na_discrepancies_specialized": na_discrepancies,
        "acceptance_verdict_summary": {
            dim: dimension_metrics[dim]["verdict"] for dim in dims
        }
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print("\n--- [1. NA EXCLUSION SUMMARY (SECTION II)] ---")
    for dim in dims:
        info = na_handling_summary[dim]
        print(f"  * {dim:5s}: Valid n = {info['valid_count']:2d} | Excluded = {info['excluded_count']:2d} | Excluded Sample IDs: {info['excluded_sample_ids']}")

    print("\n--- [2. DIMENSION INDEPENDENT METRICS (SECTION III)] ---")
    for dim in dims:
        m = dimension_metrics[dim]
        icc_info = m["ICC2k"]
        print(f"\n  Dimension: [{dim.upper()}] -> Verdict: [{m['verdict']}]")
        print(f"    1. ICC(2,k): {icc_info['val']:.4f}  (95% CI: {icc_info['ci95_str']})")
        print(f"    2. Quadratic Weighted Kappa (QWK): {m['QWK']:.4f}")
        print(f"    3. Exact Agreement Rate: {m['exact_agreement_rate']*100:.2f}%")
        print(f"    4. Adjacent Agreement Rate (|diff| <= 1): {m['adjacent_agreement_rate']*100:.2f}%")
        print(f"    5. Mean Absolute Error (MAE): {m['mae']:.4f}")
        print(f"    6. Score Difference Distribution (0/1/2/3/4): {m['difference_distribution']}")

    print("\n--- [3. INDEPENDENT ACCEPTANCE VERDICTS (SECTION VII)] ---")
    for dim in dims:
        v = dimension_metrics[dim]["verdict"]
        val = dimension_metrics[dim]["ICC2k"]["val"]
        print(f"  - {dim:5s}: ICC(2,k) = {val:.4f} -> [{v}] (Threshold: >= 0.75)")

    print("\n--- [4. DISTRIBUTION COMPARISON & SYSTEMATIC BIAS (SECTION V)] ---")
    for dim in dims:
        d_comp = distribution_comparison[dim]
        print(f"\n  Dimension: [{dim}]")
        print(f"    - Annotator A Counts (1/2/3/4/5/NA): {d_comp['Annotator_A_counts']} | Mean Score: {d_comp['Annotator_A_mean']:.4f}")
        print(f"    - Annotator B Counts (1/2/3/4/5/NA): {d_comp['Annotator_B_counts']} | Mean Score: {d_comp['Annotator_B_mean']:.4f}")
        print(f"    - Mean Delta (Annotator B - Annotator A): {d_comp['mean_difference_B_minus_A']:+.4f}")

    print("\n--- [5. NA DISCREPANCY SPECIALIZED ANALYSIS (SECTION VI)] ---")
    for dim in dims:
        na_info = na_discrepancies[dim]
        print(f"  * {dim:5s}: {na_info['discrepancy_count']} items with ONE NA vs ONE Numeric Score")
        for item in na_info["items"]:
            print(f"      - {item['sample_id']:20s}: Annotator A = {item['score_A']}, Annotator B = {item['score_B']}")

    print("\n--- [6. MAJOR DISCREPANCY LIST (|diff| >= 2) (SECTION IV)] ---")
    print(f"  Total Flagged Discrepancy Items (|diff| >= 2): {len(discrepancy_list)}")
    for idx, disc in enumerate(discrepancy_list, 1):
        print(f"\n  Item #{idx}: {disc['sample_id']} (Dataset: {disc['dataset']}) | Max Diff Dim: {disc['max_diff_dimension']} (Max Diff: {disc['max_difference']})")
        print(f"    Ratings -> Bond: A={disc['scores_A']['Bond']} vs B={disc['scores_B']['Bond']} | Goal: A={disc['scores_A']['Goal']} vs B={disc['scores_B']['Goal']} | Task: A={disc['scores_A']['Task']} vs B={disc['scores_B']['Task']}")
        print(f"    Dialogue Context (dialogue_context):\n      {disc['dialogue_context'][:200]}...")
        print(f"    Supporter Response (supporter_response):\n      {disc['supporter_response']}")

    print("\n" + "=" * 80)
    print(f" JSON Report saved to: {REPORT_JSON}")
    print(" ROUND 1 AGREEMENT REPORT ANALYSIS COMPLETED SUCCESSFULLY ")
    print("=" * 80)

if __name__ == "__main__":
    run_round1_agreement_analysis()
