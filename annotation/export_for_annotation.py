import os
import json
import random
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANNOTATION_DIR = BASE_DIR / "annotation"
SHEETS_DIR = ANNOTATION_DIR / "sheets"

RANDOM_SEED = 42

def export_annotation_sheets():
    random.seed(RANDOM_SEED)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    dev_path = PROCESSED_DIR / "dev_set.json"
    eval_path = PROCESSED_DIR / "eval_set.json"

    if not dev_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Processed datasets dev_set.json / eval_set.json not found. Run scripts/preprocess_data.py first.")

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_set = json.load(f)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    # 1. Take all 40 dev_set samples
    annotated_dev = dev_set

    # 2. Sample 150 items from eval_set (100 ESConv, 50 AnnoMI, fallback top-up)
    esconv_eval = [s for s in eval_set if s["dataset"] == "ESConv"]
    annomi_eval = [s for s in eval_set if s["dataset"] == "AnnoMI"]

    random.shuffle(esconv_eval)
    random.shuffle(annomi_eval)

    n_annomi = min(50, len(annomi_eval))
    n_esconv = 150 - n_annomi

    annotated_eval_esconv = esconv_eval[:n_esconv]
    annotated_eval_annomi = annomi_eval[:n_annomi]
    annotated_eval = annotated_eval_esconv + annotated_eval_annomi

    total_sampled = annotated_dev + annotated_eval
    assert len(total_sampled) == 190, f"Expected 190 total sampled items, got {len(total_sampled)}"

    # Save gold_sample_ids.json (150 eval gold sample IDs) for parallel Study 2 context selection
    gold_sample_ids = [s["sample_id"] for s in annotated_eval]
    with open(ANNOTATION_DIR / "gold_sample_ids.json", "w", encoding="utf-8") as f:
        json.dump(gold_sample_ids, f, indent=2, ensure_ascii=False)

    print(f"[export_for_annotation] Exported gold_sample_ids.json ({len(gold_sample_ids)} IDs) for Study 2 sampling isolation.")

    # Build DataFrame format
    records = []
    for s in total_sampled:
        records.append({
            "sample_id": s["sample_id"],
            "dataset": s["dataset"],
            "pool_type": s["pool_type"],
            "dialogue_context": s["dialogue_context_text"],
            "seeker_utterance": s["seeker_utterance"],
            "supporter_response": s["supporter_response"],
            "Bond (1-5/NA)": "",
            "Goal (1-5/NA)": "",
            "Task (1-5/NA)": "",
            "Comments": ""
        })

    records_A = list(records)
    records_B = list(records)

    random.seed(101)
    random.shuffle(records_A)

    random.seed(202)
    random.shuffle(records_B)

    df_A = pd.DataFrame(records_A)
    df_B = pd.DataFrame(records_B)

    sheet_A_path = SHEETS_DIR / "annotator_A.xlsx"
    sheet_B_path = SHEETS_DIR / "annotator_B.xlsx"

    df_A.to_excel(sheet_A_path, index=False)
    df_B.to_excel(sheet_B_path, index=False)

    print(f"[export_for_annotation] Successfully exported randomized annotation forms:")
    print(f"  - {sheet_A_path} ({len(df_A)} rows)")
    print(f"  - {sheet_B_path} ({len(df_B)} rows)")

if __name__ == "__main__":
    export_annotation_sheets()
