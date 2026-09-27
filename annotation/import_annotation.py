import os
import json
import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import pingouin as pg

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANNOTATION_DIR = BASE_DIR / "annotation"
SHEETS_DIR = ANNOTATION_DIR / "sheets"

RANDOM_SEED = 42

def ensure_consolidated_sheets():
    """
    Ensures annotator_A_main.xlsx and annotator_B_main.xlsx exist and contain the full 190 rated samples
    by merging 20 calibration rated samples with 170 main set rated samples if necessary.
    """
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    out_A_path = SHEETS_DIR / "annotator_A_main.xlsx"
    out_B_path = SHEETS_DIR / "annotator_B_main.xlsx"

    # Always re-consolidate 20 calibration + 170 main rated items to reflect any user updates to rated sheets

    # Source files for 20 calibration + 170 main rated items
    calib_A_path = SHEETS_DIR / "annotator_A_scored.xlsx"
    calib_B_path = SHEETS_DIR / "annotator_B_DONE.xlsx"
    main_A_path = SHEETS_DIR / "annotator_A_main_rated.xlsx"
    main_B_path = SHEETS_DIR / "annotator_B_main_rated.xlsx"

    if not (calib_A_path.exists() and calib_B_path.exists() and main_A_path.exists() and main_B_path.exists()):
        raise FileNotFoundError("Source annotation files missing for 190 sample consolidation.")

    # Annotator A
    df_a_calib = pd.read_excel(calib_A_path, keep_default_na=False)
    df_a_main = pd.read_excel(main_A_path, sheet_name="Sheet1", keep_default_na=False)
    df_a_calib.rename(columns={"Bond": "Bond (1-5/NA)", "Goal": "Goal (1-5/NA)", "Task": "Task (1-5/NA)"}, inplace=True)
    if "pool_type" not in df_a_calib.columns:
        df_a_calib["pool_type"] = "dev"
    if "seeker_utterance" not in df_a_calib.columns:
        df_a_calib["seeker_utterance"] = ""
    if "Comments" not in df_a_calib.columns:
        df_a_calib["Comments"] = ""

    comb_A = pd.concat([df_a_calib, df_a_main], ignore_index=True)

    # Annotator B
    df_b_calib = pd.read_excel(calib_B_path, keep_default_na=False)
    df_b_main = pd.read_excel(main_B_path, keep_default_na=False)
    df_b_calib.rename(columns={"Bond": "Bond (1-5/NA)", "Goal": "Goal (1-5/NA)", "Task": "Task (1-5/NA)"}, inplace=True)
    if "pool_type" not in df_b_calib.columns:
        df_b_calib["pool_type"] = "dev"
    if "seeker_utterance" not in df_b_calib.columns:
        df_b_calib["seeker_utterance"] = ""
    if "Comments" not in df_b_calib.columns:
        df_b_calib["Comments"] = ""

    comb_B = pd.concat([df_b_calib, df_b_main], ignore_index=True)

    comb_A.to_excel(out_A_path, index=False)
    comb_B.to_excel(out_B_path, index=False)

    print(f"[import_annotation] Consolidated 190 complete annotation rows into {out_A_path} and {out_B_path}.")
    return out_A_path, out_B_path


def run_prechecks(df_A: pd.DataFrame, df_B: pd.DataFrame):
    """
    Section 1: Pre-checks.
    Raises ValueError immediately if any check fails.
    """
    dims = ["Bond (1-5/NA)", "Goal (1-5/NA)", "Task (1-5/NA)"]

    # 1. Check length & zero missing cells
    if len(df_A) != 190:
        raise ValueError(f"Precheck 1 FAILED: Annotator A sheet must contain exactly 190 rows, got {len(df_A)}")
    if len(df_B) != 190:
        raise ValueError(f"Precheck 1 FAILED: Annotator B sheet must contain exactly 190 rows, got {len(df_B)}")

    for name, df in [("Annotator A", df_A), ("Annotator B", df_B)]:
        for col in dims:
            if col not in df.columns:
                raise ValueError(f"Precheck 1 FAILED: Missing rating column '{col}' in {name}")
            empty_count = sum(1 for v in df[col] if str(v).strip() == "" or pd.isna(v))
            if empty_count > 0:
                raise ValueError(f"Precheck 1 FAILED: {name} has {empty_count} empty/missing cells in column '{col}'")

    # 2. Check sample_id sets equality
    ids_A = set(df_A["sample_id"].astype(str))
    ids_B = set(df_B["sample_id"].astype(str))
    if ids_A != ids_B:
        diff_A = ids_A - ids_B
        diff_B = ids_B - ids_A
        raise ValueError(f"Precheck 2 FAILED: sample_id sets do not match! Unique to A: {diff_A}, Unique to B: {diff_B}")

    # 3. Check value domain {1, 2, 3, 4, 5, NA}
    valid_domain = {"1", "2", "3", "4", "5", "1.0", "2.0", "3.0", "4.0", "5.0", "NA", "N/A"}
    for name, df in [("Annotator A", df_A), ("Annotator B", df_B)]:
        for col in dims:
            invalid_entries = []
            for idx, val in enumerate(df[col]):
                s_val = str(val).strip().upper()
                if s_val not in valid_domain:
                    invalid_entries.append((idx, val))
            if invalid_entries:
                raise ValueError(f"Precheck 3 FAILED: {name} column '{col}' has values outside {{1,2,3,4,5,NA}}: {invalid_entries[:5]}")

    print("[import_annotation] ALL PRECHECKS PASSED:")
    print("  [OK] 1. Both files are complete with exactly 190 rows and zero missing cells.")
    print("  [OK] 2. sample_id sets match 100% and aligned by sample_id.")
    print("  [OK] 3. All values strictly belong to {1, 2, 3, 4, 5, NA}.\n")


def compute_agreement_metrics(merged: pd.DataFrame):
    """
    Section 2 & Section 5: Compute agreement metrics and score distribution histograms.
    """
    dims = ["Bond", "Goal", "Task"]
    agreement_report = {
        "na_handling_summary": {},
        "dimension_metrics": {},
        "distribution_comparison": {}
    }

    print("=" * 80)
    print(" INTER-ANNOTATOR AGREEMENT METRICS (190 SAMPLES) ")
    print("=" * 80)

    for dim in dims:
        col_A = f"{dim} (1-5/NA)_A"
        col_B = f"{dim} (1-5/NA)_B"

        def parse_score(v):
            s = str(v).strip().upper()
            if s in ["NA", "N/A", ""]:
                return np.nan
            return float(v)

        s_A = merged[col_A].apply(parse_score)
        s_B = merged[col_B].apply(parse_score)

        valid_mask = s_A.notna() & s_B.notna()
        clean_A = s_A[valid_mask].astype(int)
        clean_B = s_B[valid_mask].astype(int)

        na_count = int((~valid_mask).sum())
        na_rate = float(na_count / len(merged))

        # QWK
        qwk = float(cohen_kappa_score(clean_A, clean_B, weights="quadratic", labels=[1, 2, 3, 4, 5]))

        # Exact, Adjacent, MAE
        diffs = np.abs(clean_A - clean_B)
        exact_acc = float((clean_A == clean_B).mean())
        adj_acc = float((diffs <= 1).mean())
        mae = float(diffs.mean())

        diff_counts = pd.Series(diffs).value_counts().to_dict()
        diff_dist = {str(i): int(diff_counts.get(i, 0)) for i in range(5)}

        # Pingouin ICC(2,k) -> Type ICC(A,k)
        n = len(clean_A)
        icc_df = pd.DataFrame({
            "Targets": list(range(n)) + list(range(n)),
            "Raters": ["Annotator_A"] * n + ["Annotator_B"] * n,
            "Scores": list(clean_A) + list(clean_B)
        })
        icc_res = pg.intraclass_corr(data=icc_df, targets="Targets", raters="Raters", ratings="Scores")
        icc2k_row = icc_res[icc_res["Type"].str.contains("ICC\\(A,k\\)|ICC2k", case=False, na=False)].iloc[0]
        icc_val = float(icc2k_row["ICC"])
        ci95_lower = float(icc2k_row["CI95"][0])
        ci95_upper = float(icc2k_row["CI95"][1])

        # Distribution histograms (1-5, NA)
        def get_hist(series_raw):
            counts = {str(i): 0 for i in range(1, 6)}
            counts["NA"] = 0
            for v in series_raw:
                s = str(v).strip().upper()
                if s in ["NA", "N/A", ""]:
                    counts["NA"] += 1
                else:
                    try:
                        iv = str(int(float(v)))
                        if iv in counts:
                            counts[iv] += 1
                    except Exception:
                        pass
            return counts

        hist_A = get_hist(merged[col_A])
        hist_B = get_hist(merged[col_B])

        mean_A = float(clean_A.mean()) if len(clean_A) > 0 else 0.0
        mean_B = float(clean_B.mean()) if len(clean_B) > 0 else 0.0

        agreement_report["na_handling_summary"][dim] = {
            "valid_count": int(valid_mask.sum()),
            "na_count": na_count,
            "na_rate": round(na_rate, 4)
        }

        agreement_report["dimension_metrics"][dim] = {
            "ICC2k": {
                "val": round(icc_val, 4),
                "ci95_lower": round(ci95_lower, 4),
                "ci95_upper": round(ci95_upper, 4),
                "ci95_str": f"[{ci95_lower:.4f}, {ci95_upper:.4f}]"
            },
            "QWK": round(qwk, 4),
            "exact_agreement_rate": round(exact_acc, 4),
            "adjacent_agreement_rate": round(adj_acc, 4),
            "mae": round(mae, 4),
            "difference_distribution": diff_dist,
            "verdict": "PASSED" if icc_val >= 0.75 else "FAILED"
        }

        agreement_report["distribution_comparison"][dim] = {
            "Annotator_A_counts": hist_A,
            "Annotator_B_counts": hist_B,
            "Annotator_A_mean": round(mean_A, 4),
            "Annotator_B_mean": round(mean_B, 4),
            "mean_difference_B_minus_A": round(mean_B - mean_A, 4)
        }

        print(f"\n--- Dimension: {dim} ---")
        print(f"  - Valid Pair Count: {len(clean_A)} | NA Exclusion Count: {na_count} (Exclusion Rate: {na_rate*100:.2f}%)")
        print(f"  - ICC(2,k): {icc_val:.4f} (95% CI: [{ci95_lower:.4f}, {ci95_upper:.4f}])")
        print(f"  - QWK: {qwk:.4f}")
        print(f"  - Exact Agreement: {exact_acc*100:.2f}% | Adjacent Agreement: {adj_acc*100:.2f}% | MAE: {mae:.4f}")
        print(f"  - Score Diff Distribution | 0: {diff_dist['0']}, 1: {diff_dist['1']}, 2: {diff_dist['2']}, 3: {diff_dist['3']}, 4: {diff_dist['4']}")
        print(f"  - Annotator A Histogram: {hist_A}")
        print(f"  - Annotator B Histogram: {hist_B}")

    return agreement_report


def process_gold_labels_and_discrepancies(merged: pd.DataFrame, use_mock: bool = False):
    """
    Section 3: Gold labels synthesis & discrepancy export.
    Rules:
    - diff <= 1: average
    - diff >= 2 OR one NA / one numeric: export to disagreements_pending.xlsx
    - Cannot generate gold_labels.json until all pending items are resolved!
    """
    dims = ["Bond", "Goal", "Task"]
    discrepancies = []
    gold_labels = {}

    def parse_val(v):
        s = str(v).strip().upper()
        if s in ["NA", "N/A", ""]:
            return np.nan
        return float(v)

    for idx, row in merged.iterrows():
        sid = str(row["sample_id"])
        dataset = row.get("dataset_A", row.get("dataset_B", "ESConv"))
        pool_type = row.get("pool_type_A", row.get("pool_type_B", "eval"))

        gold_scores = {}
        reasons = []

        for dim in dims:
            col_A = f"{dim} (1-5/NA)_A"
            col_B = f"{dim} (1-5/NA)_B"
            vA = parse_val(row[col_A])
            vB = parse_val(row[col_B])

            if pd.isna(vA) and pd.isna(vB):
                gold_scores[dim] = None
            elif pd.isna(vA) or pd.isna(vB):
                reasons.append(f"{dim}: One NA, one numeric (A={row[col_A]}, B={row[col_B]})")
                gold_scores[dim] = None
            else:
                diff = abs(vA - vB)
                if diff <= 1:
                    gold_scores[dim] = float((vA + vB) / 2.0)
                else:
                    reasons.append(f"{dim}: Diff={diff:.1f} (A={vA}, B={vB})")
                    gold_scores[dim] = None

        if reasons:
            discrepancies.append({
                "sample_id": sid,
                "dataset": dataset,
                "pool_type": pool_type,
                "dialogue_context": row.get("dialogue_context_A", row.get("dialogue_context", "")),
                "seeker_utterance": row.get("seeker_utterance_A", row.get("seeker_utterance", "")),
                "supporter_response": row.get("supporter_response_A", row.get("supporter_response", "")),
                "Bond_A": row["Bond (1-5/NA)_A"],
                "Bond_B": row["Bond (1-5/NA)_B"],
                "Goal_A": row["Goal (1-5/NA)_A"],
                "Goal_B": row["Goal (1-5/NA)_B"],
                "Task_A": row["Task (1-5/NA)_A"],
                "Task_B": row["Task (1-5/NA)_B"],
                "Disagreement_Reason": "; ".join(reasons),
                "Consensus_Bond (1-5/NA)": "",
                "Consensus_Goal (1-5/NA)": "",
                "Consensus_Task (1-5/NA)": "",
                "Consensus_Notes": ""
            })

        gold_labels[sid] = {
            "sample_id": sid,
            "dataset": dataset,
            "pool_type": pool_type,
            "dialogue_context": row.get("dialogue_context_A", row.get("dialogue_context", "")),
            "seeker_utterance": row.get("seeker_utterance_A", row.get("seeker_utterance", "")),
            "supporter_response": row.get("supporter_response_A", row.get("supporter_response", "")),
            "scores": gold_scores
        }

    disc_path = ANNOTATION_DIR / "disagreements_pending.xlsx"
    gold_path = ANNOTATION_DIR / "gold_labels.json"

    # Check if manual consensus file already exists and has completed resolutions
    has_unresolved = False
    if discrepancies:
        if disc_path.exists():
            try:
                df_existing_disc = pd.read_excel(disc_path, keep_default_na=False)
                # Check if consensus columns exist and are filled
                consensus_cols = ["Consensus_Bond (1-5/NA)", "Consensus_Goal (1-5/NA)", "Consensus_Task (1-5/NA)"]
                if all(c in df_existing_disc.columns for c in consensus_cols):
                    # Check if all rows have consensus scores
                    unfilled_count = 0
                    for _, d_row in df_existing_disc.iterrows():
                        if any(str(d_row[c]).strip() == "" for c in consensus_cols):
                            unfilled_count += 1
                    if unfilled_count == 0:
                        # Incorporate resolved consensus into gold_labels
                        for _, d_row in df_existing_disc.iterrows():
                            d_sid = str(d_row["sample_id"])
                            if d_sid in gold_labels:
                                for dim in dims:
                                    c_val = str(d_row[f"Consensus_{dim} (1-5/NA)"]).strip().upper()
                                    if c_val in ["NA", "N/A"]:
                                        gold_labels[d_sid]["scores"][dim] = None
                                    else:
                                        gold_labels[d_sid]["scores"][dim] = float(c_val)
                    else:
                        has_unresolved = True
                else:
                    has_unresolved = True
            except Exception:
                has_unresolved = True
        else:
            has_unresolved = True

    if discrepancies:
        disc_df = pd.DataFrame(discrepancies)
        disc_df.to_excel(disc_path, index=False)
        print(f"\n[import_annotation] EXPORTED DISCREPANCIES: {len(discrepancies)} items requiring manual consensus -> {disc_path}")

    if has_unresolved:
        print(f"\n[import_annotation] BLOCKING: {len(discrepancies)} pending discrepancy items found!")
        print(f"[import_annotation] BLOCKING: gold_labels.json WILL NOT BE GENERATED until all items in {disc_path} are resolved.")
    else:
        with open(gold_path, "w", encoding="utf-8") as f:
            json.dump(gold_labels, f, indent=2, ensure_ascii=False)
        print(f"\n[import_annotation] SUCCESS: Generated {len(gold_labels)} gold label entries to {gold_path}.")

    return gold_labels, discrepancies


def select_few_shot_exemplars(merged: pd.DataFrame, gold_labels: dict):
    """
    Section 4: Pick 3 exemplars from dev pool covering Low (1-2), Mid (3), High (4-5) score ranges.
    Includes assertion verifying all exemplars come from the dev pool.
    """
    dims = ["Bond", "Goal", "Task"]

    def parse_val(v):
        s = str(v).strip().upper()
        if s in ["NA", "N/A", ""]:
            return np.nan
        return float(v)

    dev_candidates = []

    for idx, row in merged.iterrows():
        sid = str(row["sample_id"])
        pool_type = row.get("pool_type_A", row.get("pool_type_B", "eval"))
        if pool_type != "dev":
            continue

        scores = {}
        valid = True
        for dim in dims:
            col_A = f"{dim} (1-5/NA)_A"
            col_B = f"{dim} (1-5/NA)_B"
            vA = parse_val(row[col_A])
            vB = parse_val(row[col_B])
            if pd.isna(vA) and pd.isna(vB):
                scores[dim] = None
            elif pd.isna(vA) or pd.isna(vB):
                valid = False
                break
            else:
                if abs(vA - vB) >= 2:
                    valid = False
                    break
                scores[dim] = (vA + vB) / 2.0

        if valid:
            valid_scores = [v for v in scores.values() if v is not None]
            if valid_scores:
                mean_s = float(np.mean(valid_scores))
                dev_candidates.append({
                    "sample_id": sid,
                    "dataset": row.get("dataset_A", row.get("dataset_B", "ESConv")),
                    "pool_type": "dev",
                    "dialogue_context": row.get("dialogue_context_A", row.get("dialogue_context", "")),
                    "seeker_utterance": row.get("seeker_utterance_A", row.get("seeker_utterance", "")),
                    "supporter_response": row.get("supporter_response_A", row.get("supporter_response", "")),
                    "scores": scores,
                    "mean_score": mean_s
                })

    low_candidates = [item for item in dev_candidates if item["mean_score"] <= 2.3]
    mid_candidates = [item for item in dev_candidates if 2.3 < item["mean_score"] <= 3.7]
    high_candidates = [item for item in dev_candidates if item["mean_score"] > 3.7]

    low_ex = low_candidates[0] if low_candidates else dev_candidates[0]
    mid_ex = mid_candidates[0] if mid_candidates else dev_candidates[1]
    high_ex = high_candidates[0] if high_candidates else dev_candidates[2]

    exemplars = [
        {"range": "low_score (1-2)", **low_ex},
        {"range": "mid_score (3)", **mid_ex},
        {"range": "high_score (4-5)", **high_ex}
    ]

    # Section 4 Assertion
    assert all(e["pool_type"] == "dev" for e in exemplars), "CRITICAL LEAK: All few-shot exemplars must originate strictly from the dev pool!"

    exemplar_path = ANNOTATION_DIR / "few_shot_exemplars.json"
    with open(exemplar_path, "w", encoding="utf-8") as f:
        json.dump(exemplars, f, indent=2, ensure_ascii=False)

    print(f"\n[import_annotation] FEW-SHOT EXEMPLARS GENERATED -> {exemplar_path}")
    print(f"  - Low Score Exemplar:  {low_ex['sample_id']} (Mean: {low_ex['mean_score']:.2f})")
    print(f"  - Mid Score Exemplar:  {mid_ex['sample_id']} (Mean: {mid_ex['mean_score']:.2f})")
    print(f"  - High Score Exemplar: {high_ex['sample_id']} (Mean: {high_ex['mean_score']:.2f})")


def process_annotations(use_mock: bool = False):
    sheet_A_path, sheet_B_path = ensure_consolidated_sheets()

    df_A = pd.read_excel(sheet_A_path, keep_default_na=False)
    df_B = pd.read_excel(sheet_B_path, keep_default_na=False)

    # Section 1: Pre-checks
    run_prechecks(df_A, df_B)

    # Align by sample_id
    merged = pd.merge(df_A, df_B, on="sample_id", suffixes=("_A", "_B"))

    # Section 2 & 5: Compute agreement metrics & histogram score distributions
    agreement_report = compute_agreement_metrics(merged)

    report_path = ANNOTATION_DIR / "round1_agreement_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(agreement_report, f, indent=2, ensure_ascii=False)
    print(f"\n[import_annotation] Full Agreement & Distribution Report saved to {report_path}")

    # Section 3: Gold labels synthesis & discrepancy export
    gold_labels, discrepancies = process_gold_labels_and_discrepancies(merged, use_mock=use_mock)

    # Section 4: Pick few-shot exemplars with assertion
    select_few_shot_exemplars(merged, gold_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import human annotations & calculate agreement metrics")
    parser.add_argument("--mock", action="store_true", help="Force synthetic mock annotations")
    args = parser.parse_args()

    process_annotations(use_mock=args.mock)
