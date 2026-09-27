import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from pathlib import Path
import json

BASE_DIR = Path(r"C:\Users\1rog\dta_project")
SHEETS_DIR = BASE_DIR / "annotation" / "sheets"
ANNOTATION_DIR = BASE_DIR / "annotation"

def generate_gold_standard():
    # 1. Load adjudication file with keep_default_na=False
    adj_file = SHEETS_DIR / "adjudication_disagreements.xlsx"
    df_adj = pd.read_excel(adj_file, keep_default_na=False)

    adj_na_map = {}
    for idx, r in df_adj.iterrows():
        dt = str(r["Difference_Type"]).strip()
        if dt == "N/A disagreement":
            sid = str(r["sample_id"]).strip()
            dim = str(r["Dimension"]).strip()
            fs = str(r["Final_Score"]).strip()
            try:
                f_num = float(fs)
                if f_num.is_integer():
                    fs = str(int(f_num))
                else:
                    fs = str(f_num)
            except ValueError:
                if fs.upper() in ["NA", "N/A", ""]:
                    fs = "NA"
            adj_na_map[(sid, dim)] = fs

    print(f"[OK] Adjudicated NA entries loaded: {len(adj_na_map)}")

    # 2. Load 170 main rated sheets
    file_a_170 = SHEETS_DIR / "annotator_A_main_rated.xlsx"
    file_b_170 = SHEETS_DIR / "annotator_B_main_rated.xlsx"

    df_a_170 = pd.read_excel(file_a_170, header=None, keep_default_na=False)
    if df_a_170.iloc[0, 1] == "dataset":
        cols = [
            "sample_id", "dataset", "pool_type", "dialogue_context",
            "seeker_utterance", "supporter_response", "Bond (1-5/NA)",
            "Goal (1-5/NA)", "Task (1-5/NA)", "Comments"
        ]
        df_a_170.columns = cols
        df_a_170 = df_a_170.iloc[1:].copy()

    df_b_170 = pd.read_excel(file_b_170, keep_default_na=False)

    merged_170 = pd.merge(df_a_170, df_b_170, on="sample_id", suffixes=("_A", "_B"))

    def resolve_gold_score(sid, dim, vA_raw, vB_raw):
        vA = str(vA_raw).strip().upper()
        vB = str(vB_raw).strip().upper()
        if vA in ["N/A", ""]: vA = "NA"
        if vB in ["N/A", ""]: vB = "NA"
        
        # Priority 1: Adjudicated NA in adjudication_disagreements.xlsx
        if (sid, dim) in adj_na_map:
            return adj_na_map[(sid, dim)], "Adjudicated NA"
        
        # Priority 2: Exact agreement
        if vA == vB:
            return vA, "Exact Agreement"
        
        # Priority 3: Score difference -> DTA Rubric P6 min score
        try:
            nA = float(vA)
            nB = float(vB)
            min_v = min(nA, nB)
            return str(int(min_v)), "DTA Rubric P6 Min Score"
        except ValueError:
            return "NA", "Fallback NA"

    records_170 = []
    gold_json_dict = {}

    for idx, r in merged_170.iterrows():
        sid = str(r["sample_id"]).strip()
        ds = r["dataset_A"]
        pt = r["pool_type_A"]
        dc = r["dialogue_context_A"]
        su = r["seeker_utterance_A"]
        sr = r["supporter_response_A"]
        
        rec = {
            "sample_id": sid,
            "dataset": ds,
            "pool_type": pt,
            "dialogue_context": dc,
            "seeker_utterance": su,
            "supporter_response": sr,
        }
        
        dim_scores = {}
        for dim in ["Bond", "Goal", "Task"]:
            colA = f"{dim} (1-5/NA)_A"
            colB = f"{dim} (1-5/NA)_B"
            g_val, src = resolve_gold_score(sid, dim, r[colA], r[colB])
            rec[f"{dim} (1-5/NA)"] = g_val
            dim_scores[dim] = None if g_val == "NA" else float(g_val)
            
        comments_A = str(r.get("Comments_A", "")).strip()
        comments_B = str(r.get("Comments_B", "")).strip()
        combined_comments = []
        if comments_A and comments_A.lower() != "nan": combined_comments.append(f"A: {comments_A}")
        if comments_B and comments_B.lower() != "nan": combined_comments.append(f"B: {comments_B}")
        rec["Comments"] = " | ".join(combined_comments)
        
        records_170.append(rec)
        gold_json_dict[sid] = {
            "sample_id": sid,
            "dataset": ds,
            "pool_type": pt,
            "dialogue_context": dc,
            "seeker_utterance": su,
            "supporter_response": sr,
            "scores": dim_scores
        }

    # 3. Load 20 calibration samples for 190 consolidated set
    df_calib_a = pd.read_excel(SHEETS_DIR / "annotator_A_scored.xlsx", keep_default_na=False)
    df_calib_b = pd.read_excel(SHEETS_DIR / "annotator_B_DONE.xlsx", keep_default_na=False)

    merged_calib = pd.merge(df_calib_a, df_calib_b, on="sample_id", suffixes=("_A", "_B"))
    records_calib = []

    for idx, r in merged_calib.iterrows():
        sid = str(r["sample_id"]).strip()
        ds = r.get("dataset_A", r.get("dataset_B", "ESConv"))
        pt = "dev"
        dc = r.get("dialogue_context_A", r.get("dialogue_context", ""))
        su = r.get("seeker_utterance_A", r.get("seeker_utterance", ""))
        sr = r.get("supporter_response_A", r.get("supporter_response", ""))
        
        rec = {
            "sample_id": sid,
            "dataset": ds,
            "pool_type": pt,
            "dialogue_context": dc,
            "seeker_utterance": su,
            "supporter_response": sr,
        }
        
        dim_scores = {}
        for dim in ["Bond", "Goal", "Task"]:
            vA = str(r.get(f"{dim}_A", r.get(f"{dim} (1-5/NA)_A", ""))).strip().upper()
            vB = str(r.get(f"{dim}_B", r.get(f"{dim} (1-5/NA)_B", ""))).strip().upper()
            g_val, src = resolve_gold_score(sid, dim, vA, vB)
            rec[f"{dim} (1-5/NA)"] = g_val
            dim_scores[dim] = None if g_val == "NA" else float(g_val)
            
        comments_A = str(r.get("Comments_A", "")).strip()
        comments_B = str(r.get("Comments_B", "")).strip()
        combined_comments = []
        if comments_A and comments_A.lower() != "nan": combined_comments.append(f"A: {comments_A}")
        if comments_B and comments_B.lower() != "nan": combined_comments.append(f"B: {comments_B}")
        rec["Comments"] = " | ".join(combined_comments)
        
        records_calib.append(rec)
        gold_json_dict[sid] = {
            "sample_id": sid,
            "dataset": ds,
            "pool_type": pt,
            "dialogue_context": dc,
            "seeker_utterance": su,
            "supporter_response": sr,
            "scores": dim_scores
        }

    records_190 = records_calib + records_170

    print(f"[OK] Synthesized 170 records count: {len(records_170)}")
    print(f"[OK] Synthesized 190 records count: {len(records_190)}")
    print(f"[OK] Total gold_json_dict keys: {len(gold_json_dict)}")

    def save_pretty_excel(records, file_path):
        df = pd.DataFrame(records)
        df.to_excel(file_path, index=False)
        
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        ws.views.sheetView[0].showGridLines = True
        
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        cell_font = Font(name="Segoe UI", size=10)
        cell_align_left = Alignment(horizontal="left", vertical="top", wrap_text=True)
        cell_align_center = Alignment(horizontal="center", vertical="top", wrap_text=True)
        
        thin_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9")
        )
        
        ws.row_dimensions[1].height = 28
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            
        for row in ws.iter_rows(min_row=2, max_row=len(records)+1, min_col=1, max_col=10):
            for cell in row:
                cell.font = cell_font
                cell.border = thin_border
                col_letter = cell.column_letter
                if col_letter in ["A", "B", "C", "G", "H", "I"]:
                    cell.alignment = cell_align_center
                else:
                    cell.alignment = cell_align_left
                    
        column_widths = {
            "A": 18,
            "B": 12,
            "C": 12,
            "D": 45,
            "E": 35,
            "F": 40,
            "G": 15,
            "H": 15,
            "I": 15,
            "J": 30
        }
        for col_letter, width in column_widths.items():
            ws.column_dimensions[col_letter].width = width
            
        dv = DataValidation(type="list", formula1='"1,2,3,4,5,NA"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"G2:G{len(records)+1}")
        dv.add(f"H2:H{len(records)+1}")
        dv.add(f"I2:I{len(records)+1}")
        
        wb.save(file_path)
        print(f"[OK] Saved styled excel to {file_path}")

    save_pretty_excel(records_170, SHEETS_DIR / "gold_standard_main_rated.xlsx")
    save_pretty_excel(records_190, SHEETS_DIR / "gold_standard_main_consolidated.xlsx")
    save_pretty_excel(records_170, SHEETS_DIR / "gold_standard.xlsx")
    save_pretty_excel(records_calib, SHEETS_DIR / "gold_standard_calibration_20.xlsx")

    records_eval_150 = [r for r in records_170 if r["pool_type"] == "eval"]
    records_dev_20 = [r for r in records_170 if r["pool_type"] == "dev"]

    save_pretty_excel(records_eval_150, SHEETS_DIR / "gold_standard_eval_150.xlsx")
    save_pretty_excel(records_dev_20, SHEETS_DIR / "gold_standard_dev_20.xlsx")

    with open(ANNOTATION_DIR / "gold_labels.json", "w", encoding="utf-8") as f:
        json.dump(gold_json_dict, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved gold_labels.json to {ANNOTATION_DIR / 'gold_labels.json'}")

if __name__ == "__main__":
    generate_gold_standard()
