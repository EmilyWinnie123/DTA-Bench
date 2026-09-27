import os
import json
import random
import re
import pandas as pd
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.styles import Alignment, Font
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANNOTATION_DIR = BASE_DIR / "annotation"
SHEETS_DIR = ANNOTATION_DIR / "sheets"

RANDOM_SEED = 42

# --- Feature Categories Matching DTA_Rubric.md §7 Edge Cases ---

PATTERNS = {
    "E2_vent_but_advised": (
        re.compile(r"\b(vent|venting|talk|listen|get this off|express|just need|just want|someone to talk|share|speak|frustrated|sadness|hard time)\b", re.IGNORECASE),
        re.compile(r"\b(should|try|need to|must|recommend|suggest|how about|consider|make sure|instead of|could|advice|walk|exercise|action|do)\b", re.IGNORECASE)
    ),
    "E3_premature_advice": re.compile(r"\b(should|try|need to|must|recommend|suggest|how about|consider|make sure|you can)\b", re.IGNORECASE),
    "E4_question_only": re.compile(r"\?$", re.IGNORECASE),
    "E5_comparison_comfort": re.compile(r"\b(at least|could be worse|others have|other people|look on the bright side|things could be worse|worse off)\b", re.IGNORECASE),
    "E6_constraint_ignored": (
        re.compile(r"\b(can't afford|no time|don't have time|impossible|cannot|no money|too expensive|can not|rent|budget)\b", re.IGNORECASE),
        re.compile(r"\b(should|try|need to|must|recommend|suggest|how about|consider|move|go|find|seek)\b", re.IGNORECASE)
    ),
    "E9_self_disclosure": re.compile(r"\b(i went through|i had a similar|in my experience|i also felt|i remember when i|my experience|when i was|i try to|i find that|i always feel)\b", re.IGNORECASE)
}

def classify_sample(sample: dict) -> list[str]:
    seeker = sample.get("seeker_utterance", "").lower()
    supp = sample.get("supporter_response", "").lower()
    ctx = sample.get("dialogue_context_text", "").lower()
    words_supp = re.findall(r"\b[a-z]+\b", supp)
    words_seeker = set(re.findall(r"\b[a-z]+\b", seeker))
    stopwords = set(["the", "a", "an", "and", "or", "to", "in", "of", "you", "i", "it", "that", "is", "was", "for", "on", "with", "have", "be", "so", "this", "my", "your", "are", "but", "what", "how", "can", "if", "we"])

    cats = []

    # E2: Venting goal but advised
    p_vent, p_adv = PATTERNS["E2_vent_but_advised"]
    if (p_vent.search(seeker) or p_vent.search(ctx)) and p_adv.search(supp):
        cats.append("E2_vent_but_advised")

    # E3: Premature advice
    if PATTERNS["E3_premature_advice"].search(supp):
        cats.append("E3_premature_advice")

    # E4: Question only
    if supp.strip().endswith("?") and not any(r in supp for r in ["sounds like", "it seems", "you feel", "i hear"]):
        cats.append("E4_question_only")

    # E5: Comparison comfort
    if PATTERNS["E5_comparison_comfort"].search(supp):
        cats.append("E5_comparison_comfort")

    # E6: Constraint ignored
    p_cons, p_cadv = PATTERNS["E6_constraint_ignored"]
    if (p_cons.search(seeker) or p_cons.search(ctx)) and p_cadv.search(supp):
        cats.append("E6_constraint_ignored")

    # E8: Generic fluent (long, no overlap)
    seeker_cw = words_seeker - stopwords
    supp_cw = set(words_supp) - stopwords
    if len(words_supp) >= 35 and len(seeker_cw & supp_cw) == 0:
        cats.append("E8_generic_fluent")

    # E9: Self disclosure
    if PATTERNS["E9_self_disclosure"].search(supp):
        cats.append("E9_self_disclosure")

    if not cats:
        cats.append("CONTROL")

    return cats

def export_calibration_set():
    random.seed(RANDOM_SEED)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    dev_path = PROCESSED_DIR / "dev_set.json"
    if not dev_path.exists():
        raise FileNotFoundError("dev_set.json missing! Run scripts/preprocess_data.py first.")

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_samples = json.load(f)

    # Classify all dev samples
    classified = []
    for s in dev_samples:
        cats = classify_sample(s)
        classified.append({"sample": s, "categories": cats})

    # The 8 target categories
    target_categories = [
        "E2_vent_but_advised",
        "E3_premature_advice",
        "E4_question_only",
        "E5_comparison_comfort",
        "E6_constraint_ignored",
        "E8_generic_fluent",
        "E9_self_disclosure",
        "CONTROL"
    ]

    selected = []
    selected_ids = set()
    primary_category_map = {}

    # Target allocation for 20 total items (2-3 items per category)
    # E2=3, E3=2, E4=3, E5=2, E6=2, E8=2, E9=3, CONTROL=3 -> Total 20
    category_quotas = {
        "E2_vent_but_advised": 3,
        "E3_premature_advice": 2,
        "E4_question_only": 3,
        "E5_comparison_comfort": 2,
        "E6_constraint_ignored": 2,
        "E8_generic_fluent": 2,
        "E9_self_disclosure": 3,
        "CONTROL": 3
    }

    # Pass 1: Select quota items for each category
    for cat in target_categories:
        quota = category_quotas[cat]
        candidates = [item for item in classified if cat in item["categories"] and item["sample"]["sample_id"] not in selected_ids]
        random.shuffle(candidates)

        # Mix ESConv and AnnoMI
        es_cands = [item for item in candidates if item["sample"]["dataset"] == "ESConv"]
        an_cands = [item for item in candidates if item["sample"]["dataset"] == "AnnoMI"]

        cat_picks = []
        for _ in range(quota):
            if an_cands and sum(1 for p in selected if p["sample"]["dataset"] == "AnnoMI") < 5 and random.random() < 0.4:
                pick = an_cands.pop(0)
            elif es_cands:
                pick = es_cands.pop(0)
            elif an_cands:
                pick = an_cands.pop(0)
            else:
                break

            if pick["sample"]["sample_id"] not in selected_ids:
                selected.append(pick)
                selected_ids.add(pick["sample"]["sample_id"])
                primary_category_map[pick["sample"]["sample_id"]] = cat

    # Assert exactly 20 samples
    assert len(selected) == 20, f"Expected 20 selected samples, got {len(selected)}"

    # Check E2 count >= 2
    e2_selected_count = sum(1 for item in selected if "E2_vent_but_advised" in item["categories"])
    assert e2_selected_count >= 2, f"CRITICAL REQUIREMENT FAILED: E2_vent_but_advised count must be >= 2, got {e2_selected_count}"

    # Save internal metadata for research analysis (NOT exported to raters)
    calib_meta = []
    for item in selected:
        s = item["sample"]
        calib_meta.append({
            "sample_id": s["sample_id"],
            "dataset": s["dataset"],
            "transcript_id": s["transcript_id"],
            "primary_category": primary_category_map.get(s["sample_id"], item["categories"][0]),
            "all_hit_categories": item["categories"]
        })

    with open(ANNOTATION_DIR / "calibration_sample_metadata.json", "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2, ensure_ascii=False)

    # Export Excel Sheets A & B (Columns: sample_id, dataset, dialogue_context, supporter_response, Bond, Goal, Task, Comments)
    raw_rows = []
    for item in selected:
        s = item["sample"]
        raw_rows.append({
            "sample_id": s["sample_id"],
            "dataset": s["dataset"],
            "dialogue_context": s["dialogue_context_text"],
            "supporter_response": s["supporter_response"],
            "Bond": "",
            "Goal": "",
            "Task": "",
            "Comments": ""
        })

    rows_A = list(raw_rows)
    rows_B = list(raw_rows)

    random.seed(101)
    random.shuffle(rows_A)

    random.seed(202)
    random.shuffle(rows_B)

    order_A = [r["sample_id"] for r in rows_A]
    order_B = [r["sample_id"] for r in rows_B]
    assert order_A != order_B, "Annotator A and Annotator B must have DIFFERENT row orders!"

    def save_formatted_excel(rows: list, file_path: Path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "DTA_Calibration"

        headers = ["sample_id", "dataset", "dialogue_context", "supporter_response", "Bond", "Goal", "Task", "Comments"]
        ws.append(headers)

        header_font = Font(name="Calibri", size=11, bold=True)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        col_widths = {
            "A": 22,  # sample_id
            "B": 12,  # dataset
            "C": 60,  # dialogue_context
            "D": 50,  # supporter_response
            "E": 10,  # Bond
            "F": 10,  # Goal
            "G": 10,  # Task
            "H": 30   # Comments
        }
        for col_letter, width in col_widths.items():
            ws.column_dimensions[col_letter].width = width

        for r_idx, r in enumerate(rows, start=2):
            ws.append([
                r["sample_id"],
                r["dataset"],
                r["dialogue_context"],
                r["supporter_response"],
                "", "", "", ""
            ])

            for col_idx in range(1, 9):
                c = ws.cell(row=r_idx, column=col_idx)
                c.alignment = Alignment(vertical="top", wrap_text=True)

        dv = DataValidation(type="list", formula1='"1,2,3,4,5,NA"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add("E2:G21")

        wb.save(file_path)

    sheet_A_path = SHEETS_DIR / "annotator_A.xlsx"
    sheet_B_path = SHEETS_DIR / "annotator_B.xlsx"

    save_formatted_excel(rows_A, sheet_A_path)
    save_formatted_excel(rows_B, sheet_B_path)

    # Verification Output
    print("\n" + "=" * 70)
    print(" CALIBRATION SAMPLING VERIFICATION REPORT (8 CATEGORIES) ")
    print("=" * 70)

    print(f"1. Total Selected Samples: {len(selected)}")
    print(f"   - ESConv Count: {sum(1 for item in selected if item['sample']['dataset'] == 'ESConv')}")
    print(f"   - AnnoMI Count: {sum(1 for item in selected if item['sample']['dataset'] == 'AnnoMI')}")

    print("\n2. Category Distribution across 20 Selected Samples:")
    prim_counts = {}
    for item in selected:
        cat = primary_category_map[item["sample"]["sample_id"]]
        prim_counts[cat] = prim_counts.get(cat, 0) + 1

    for cat_name in target_categories:
        count = prim_counts.get(cat_name, 0)
        print(f"   - {cat_name:25s}: {count} samples")

    print(f"\n3. E2_vent_but_advised Core Mechanism Check:")
    print(f"   - E2 Selected Count: {e2_selected_count} (Requirement >= 2) -> {'PASSED' if e2_selected_count >= 2 else 'FAILED'}")

    df_A = pd.read_excel(sheet_A_path)
    df_B = pd.read_excel(sheet_B_path)

    print(f"\n4. Excel Sheets Verification:")
    print(f"   - Sheet A Rows: {len(df_A)} | Columns: {list(df_A.columns)}")
    print(f"   - Sheet B Rows: {len(df_B)} | Columns: {list(df_B.columns)}")
    print(f"   - Row Orders Differ: {order_A != order_B}")

    assert len(df_A) == 20 and len(df_B) == 20, "Excel rows must be 20"
    assert order_A != order_B, "Row orders must differ"

    print("\n" + "=" * 70)
    print(" CALIBRATION SAMPLING COMPLETED WITH ZERO ERRORS ")
    print("=" * 70)

if __name__ == "__main__":
    export_calibration_set()
