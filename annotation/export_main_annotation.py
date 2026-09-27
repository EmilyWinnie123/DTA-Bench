import os
import json
import random
import pandas as pd
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Alignment
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANNOTATION_DIR = BASE_DIR / "annotation"
SHEETS_DIR = ANNOTATION_DIR / "sheets"

RANDOM_SEED = 42

def export_main_annotation_sheets():
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    dev_path = PROCESSED_DIR / "dev_set.json"
    eval_path = PROCESSED_DIR / "eval_set.json"
    calib_path = SHEETS_DIR / "annotator_A.xlsx"

    if not dev_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Processed datasets dev_set.json / eval_set.json missing.")
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration sheet missing at {calib_path}")

    # Load 20 calibration sample_ids to exclude
    df_calib = pd.read_excel(calib_path)
    calib_sample_ids = set(df_calib["sample_id"].astype(str))
    assert len(calib_sample_ids) == 20, f"Expected 20 calibration IDs, got {len(calib_sample_ids)}"

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_set = json.load(f)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    random.seed(RANDOM_SEED)

    # 1. Remaining 20 dev items (excluding calibration IDs)
    dev_rem = [s for s in dev_set if str(s["sample_id"]) not in calib_sample_ids]
    random.shuffle(dev_rem)
    main_dev = dev_rem[:20]

    # 2. 150 eval items (100 ESConv, 50 AnnoMI)
    esconv_eval = [s for s in eval_set if s["dataset"] == "ESConv" and str(s["sample_id"]) not in calib_sample_ids]
    annomi_eval = [s for s in eval_set if s["dataset"] == "AnnoMI" and str(s["sample_id"]) not in calib_sample_ids]

    random.shuffle(esconv_eval)
    random.shuffle(annomi_eval)

    n_annomi = min(50, len(annomi_eval))
    n_esconv = 150 - n_annomi
    main_eval = esconv_eval[:n_esconv] + annomi_eval[:n_annomi]

    # 3. Main 170 items
    main_items = main_dev + main_eval
    assert len(main_items) == 170, f"Expected 170 main items, got {len(main_items)}"

    # Strict Zero Overlap Check with Calibration IDs
    main_ids = set(str(s["sample_id"]) for s in main_items)
    overlap = main_ids & calib_sample_ids
    assert len(overlap) == 0, f"CRITICAL ERROR: {len(overlap)} Calibration IDs overlapped with main set!"

    records = []
    for s in main_items:
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

    # Independent random row shuffling for Annotator A and Annotator B
    records_A = list(records)
    records_B = list(records)

    random.seed(303)
    random.shuffle(records_A)

    random.seed(404)
    random.shuffle(records_B)

    out_A_path = SHEETS_DIR / "annotator_A_main.xlsx"
    out_B_path = SHEETS_DIR / "annotator_B_main.xlsx"

    save_formatted_excel(records_A, out_A_path)
    save_formatted_excel(records_B, out_B_path)

    print("=" * 80)
    print(" MAIN ANNOTATION SET EXPORTED SUCCESSFULLY ")
    print("=" * 80)
    print(f"  - Total Main Set Items: {len(main_items)} items (20 dev + 150 eval)")
    print(f"  - Excluded Calibration Items: {len(calib_sample_ids)} items (Round 1)")
    print(f"  - Zero Overlap Verified with Calibration Set.")
    print(f"  - Exported Sheet A: {out_A_path} ({len(records_A)} rows)")
    print(f"  - Exported Sheet B: {out_B_path} ({len(records_B)} rows)")

def save_formatted_excel(records: list, file_path: Path):
    df = pd.DataFrame(records)
    df.to_excel(file_path, index=False)

    wb = openpyxl.load_workbook(file_path)
    ws = wb.active

    # Data validation for score entries (1, 2, 3, 4, 5, NA)
    dv = DataValidation(type="list", formula1='"1,2,3,4,5,NA"', allow_blank=True)
    ws.add_data_validation(dv)

    dv.add(f"G2:G{len(records)+1}")
    dv.add(f"H2:H{len(records)+1}")
    dv.add(f"I2:I{len(records)+1}")

    for row in ws.iter_rows(min_row=2, max_row=len(records)+1, min_col=1, max_col=10):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    column_widths = {
        "A": 18,
        "B": 10,
        "C": 10,
        "D": 45,
        "E": 30,
        "F": 35,
        "G": 14,
        "H": 14,
        "I": 14,
        "J": 25
    }
    for col_letter, width in column_widths.items():
        ws.column_dimensions[col_letter].width = width

    wb.save(file_path)

if __name__ == "__main__":
    export_main_annotation_sheets()
