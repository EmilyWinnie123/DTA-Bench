import os
import sys
import json
import hashlib
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

S1_CLAUDE_FILE = CACHE_DIR / "stage2_p4_predictions.json"
S1_QWEN_FILE = CACHE_DIR / "qwen_stage2_p4_predictions.json"
S2_AI_FILE = CACHE_DIR / "study2_ai_responses.json"
S2_MAP_FILE = CACHE_DIR / "study2_anonymization_map.json"
S2_CLAUDE_FILE = CACHE_DIR / "study2_claude_eval_predictions.json"
S2_QWEN_FILE = CACHE_DIR / "study2_qwen_eval_predictions.json"
CONFIG_FILE = BASE_DIR / "config" / "judge_p4_config.json"
RUBRIC_FILE = BASE_DIR / "rubric" / "DTA_Rubric.md"

def get_file_provenance(p: Path) -> dict:
    if not p.exists():
        return {"exists": False, "path": str(p)}
    data = open(p, "rb").read()
    h = hashlib.sha256(data).hexdigest()
    mtime = time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime(os.path.getmtime(p)))
    return {
        "exists": True,
        "path": str(p.resolve()),
        "size_bytes": len(data),
        "sha256": h,
        "modified_at": mtime
    }

def calc_wilcoxon_r(z_val, n):
    if n <= 0:
        return 0.0
    return abs(z_val) / np.sqrt(n)

def bootstrap_paired_diff_ci(x, y, n_boot=2000, ci=0.95):
    diffs = np.array(x) - np.array(y)
    if len(diffs) == 0:
        return 0.0, 0.0, 0.0
    mean_diff = float(np.mean(diffs))
    np.random.seed(42)
    boot_means = []
    n = len(diffs)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        boot_means.append(np.mean(diffs[idx]))
    alpha = (1 - ci) / 2
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1 - alpha) * 100))
    return round(mean_diff, 4), round(low, 4), round(high, 4)

def cochran_q_test(df_binary):
    # df_binary has columns ['Human', 'Gemini', 'DeepSeek']
    # Matrix X: k=3 columns, N rows
    X = df_binary[['Human', 'Gemini', 'DeepSeek']].values
    k = X.shape[1]
    n = X.shape[0]
    
    C = np.sum(X, axis=0) # Column sums
    R = np.sum(X, axis=1) # Row sums
    
    num = (k - 1) * (k * np.sum(C**2) - (np.sum(C))**2)
    den = k * np.sum(R) - np.sum(R**2)
    
    if den == 0:
        return 0.0, 1.0, 2
    
    q_stat = float(num / den)
    p_val = float(1.0 - stats.chi2.cdf(q_stat, df=k-1))
    return round(q_stat, 4), round(p_val, 6), k-1

def run_study2_statistical_analysis():
    print("=" * 110)
    print(" STUDY 2 — FORMAL STATISTICAL ANALYSIS PIPELINE ")
    print("=" * 110)

    # Provenance gathering
    prov_files = {
        "s1_claude": S1_CLAUDE_FILE,
        "s1_qwen": S1_QWEN_FILE,
        "s2_ai": S2_AI_FILE,
        "s2_map": S2_MAP_FILE,
        "s2_claude": S2_CLAUDE_FILE,
        "s2_qwen": S2_QWEN_FILE,
        "config": CONFIG_FILE,
        "rubric": RUBRIC_FILE
    }
    prov_meta = {k: get_file_provenance(v) for k, v in prov_files.items()}
    print("\n--- INPUT FILE PROVENANCE ---")
    for k, meta in prov_meta.items():
        print(f"  [{k}] SHA256: {meta['sha256'][:16]}... | MTime: {meta['modified_at']} | Size: {meta['size_bytes']}B")

    # Load data
    s1_claude = json.load(open(S1_CLAUDE_FILE, encoding="utf-8"))
    s1_qwen = json.load(open(S1_QWEN_FILE, encoding="utf-8"))
    s2_ai = json.load(open(S2_AI_FILE, encoding="utf-8"))
    s2_map = json.load(open(S2_MAP_FILE, encoding="utf-8"))
    s2_claude = json.load(open(S2_CLAUDE_FILE, encoding="utf-8"))
    s2_qwen = json.load(open(S2_QWEN_FILE, encoding="utf-8"))

    # Map anon_id -> metadata
    anon_dict = {item["anon_id"]: item for item in s2_map}

    # Word count dictionary by sample_id and source
    # s2_ai has 150 items, each has responses for Human, Gemini, DeepSeek
    word_counts = {}
    for sid, item in s2_ai.items():
        word_counts[sid] = {
            "Human": len(item["human_response"].split()),
            "Gemini": len(item["gemini_response"].split()),
            "DeepSeek": len(item["deepseek_response"].split())
        }

    # =========================================================================
    # REQUIREMENT 1: TABLE 5 — STUDY 1 VS STUDY 2 HUMAN RESPONSE AUDIT
    # =========================================================================
    print("\n" + "=" * 110)
    print(" REQUIREMENT 1: TABLE 5 — HUMAN RESPONSE CONSISTENCY AUDIT (STUDY 1 VS STUDY 2) ")
    print("=" * 110)

    human_items = [item for item in s2_map if item["source"] == "Human"]
    audit_table = []
    table_s2_results = []

    for judge_name, s1_data, s2_data in [("Claude", s1_claude, s2_claude), ("Qwen", s1_qwen, s2_qwen)]:
        for dim in ["bond", "goal", "task"]:
            exact_cnt = 0
            both_na_cnt = 0
            both_num_cnt = 0
            na_to_num_cnt = 0
            num_to_na_cnt = 0
            changed_na_cnt = 0
            abs_diffs = []

            for item in human_items:
                sid = item["sample_id"]
                aid = item["anon_id"]
                
                v1 = str(s1_data[sid][dim]).upper()
                v2 = str(s2_data[aid][dim]).upper()

                if v1 == v2:
                    exact_cnt += 1

                v1_is_na = (v1 == "NA")
                v2_is_na = (v2 == "NA")

                if v1_is_na and v2_is_na:
                    both_na_cnt += 1
                elif not v1_is_na and not v2_is_na:
                    both_num_cnt += 1
                    abs_diffs.append(abs(float(v1) - float(v2)))
                elif v1_is_na and not v2_is_na:
                    na_to_num_cnt += 1
                    changed_na_cnt += 1
                elif not v1_is_na and v2_is_na:
                    num_to_na_cnt += 1
                    changed_na_cnt += 1

            mae = round(float(np.mean(abs_diffs)), 4) if abs_diffs else 0.0
            exact_pct = round((exact_cnt / len(human_items)) * 100.0, 2)
            na_agree_pct = round(((both_na_cnt + both_num_cnt) / len(human_items)) * 100.0, 2)

            audit_table.append({
                "Judge": judge_name,
                "Dimension": dim.capitalize(),
                "Total N": len(human_items),
                "Exact Match Count": exact_cnt,
                "Exact Match %": f"{exact_pct}%",
                "Both Numeric N": both_num_cnt,
                "Numeric MAE": mae,
                "Both N/A N": both_na_cnt,
                "N/A Status Changed N": changed_na_cnt
            })

            table_s2_results.append({
                "Judge": judge_name,
                "Dimension": dim.capitalize(),
                "Both NA": both_na_cnt,
                "Both Numeric": both_num_cnt,
                "NA -> Numeric": na_to_num_cnt,
                "Numeric -> NA": num_to_na_cnt,
                "N/A Agreement": f"{na_agree_pct}%"
            })

    df_t5 = pd.DataFrame(audit_table)
    print(df_t5.to_string(index=False))

    # =========================================================================
    # REQUIREMENT 2, 3, 4, 5, 6, 7: TABLE 6 — PAIRED STATISTICAL ANALYSIS
    # =========================================================================
    print("\n" + "=" * 110)
    print(" REQUIREMENT 2-7: TABLE 6 — STUDY 2 PAIRED STATISTICAL ANALYSIS ")
    print("=" * 110)

    # Build structured dataset per judge
    # sample_id -> {Human: {bond, goal, task}, Gemini: {...}, DeepSeek: {...}}
    def build_judge_dataset(s2_preds):
        sample_dict = {}
        for item in s2_map:
            aid = item["anon_id"]
            sid = item["sample_id"]
            src = item["source"]
            rec = s2_preds[aid]
            
            if sid not in sample_dict:
                sample_dict[sid] = {}
            
            sample_dict[sid][src] = {
                "bond": rec["bond"],
                "goal": rec["goal"],
                "task": rec["task"],
                "word_count": word_counts[sid][src]
            }
        return sample_dict

    claude_ds = build_judge_dataset(s2_claude)
    qwen_ds = build_judge_dataset(s2_qwen)

    table6_results = {}

    for j_name, j_ds in [("Claude", claude_ds), ("Qwen", qwen_ds)]:
        table6_results[j_name] = {}
        print(f"\n >>> JUDGE: {j_name.upper()} <<<")

        for dim in ["bond", "goal", "task"]:
            print(f"\n  --- Dimension: {dim.capitalize()} ---")

            # Extract scores per sample
            rows = []
            for sid, src_data in j_ds.items():
                rows.append({
                    "sample_id": sid,
                    "Human": src_data["Human"][dim],
                    "Gemini": src_data["Gemini"][dim],
                    "DeepSeek": src_data["DeepSeek"][dim],
                    "wc_Human": src_data["Human"]["word_count"],
                    "wc_Gemini": src_data["Gemini"]["word_count"],
                    "wc_DeepSeek": src_data["DeepSeek"]["word_count"]
                })
            df_dim = pd.DataFrame(rows)

            # 1. N/A Analysis
            df_na_bin = pd.DataFrame({
                "Human": (df_dim["Human"].astype(str).str.upper() == "NA").astype(int),
                "Gemini": (df_dim["Gemini"].astype(str).str.upper() == "NA").astype(int),
                "DeepSeek": (df_dim["DeepSeek"].astype(str).str.upper() == "NA").astype(int)
            })

            na_counts = {
                "Human": int(df_na_bin["Human"].sum()),
                "Gemini": int(df_na_bin["Gemini"].sum()),
                "DeepSeek": int(df_na_bin["DeepSeek"].sum())
            }
            na_pcts = {src: round((cnt / 150.0) * 100.0, 2) for src, cnt in na_counts.items()}

            q_stat, q_pval, q_df = cochran_q_test(df_na_bin)

            # McNemar post-hoc if Cochran Q significant
            mcnemar_res = {}
            if q_pval < 0.05:
                pairs = [("Gemini", "Human"), ("DeepSeek", "Human"), ("Gemini", "DeepSeek")]
                raw_pvals = []
                mcn_details = []
                for p1, p2 in pairs:
                    # Contigency table for binary N/A
                    # b: p1=1, p2=0; c: p1=0, p2=1
                    b = int(np.sum((df_na_bin[p1] == 1) & (df_na_bin[p2] == 0)))
                    c = int(np.sum((df_na_bin[p1] == 0) & (df_na_bin[p2] == 1)))
                    # McNemar exact binomial or chi2
                    res = stats.binomtest(b, b + c, 0.5) if (b + c) > 0 else None
                    pv = res.pvalue if res else 1.0
                    raw_pvals.append(pv)
                    mcn_details.append({"pair": f"{p1} vs {p2}", "b": b, "c": c, "p_raw": round(pv, 6)})

                adj_p = multipletests(raw_pvals, method="holm")[1]
                for idx, (p1, p2) in enumerate(pairs):
                    mcnemar_res[f"{p1}_vs_{p2}"] = {
                        "p_raw": round(raw_pvals[idx], 6),
                        "p_adj": round(float(adj_p[idx]), 6),
                        "significant": bool(adj_p[idx] < 0.05)
                    }

            # 2. Numerical Score Analysis (3-way complete cases)
            valid_mask = (
                (df_dim["Human"].astype(str).str.upper() != "NA") &
                (df_dim["Gemini"].astype(str).str.upper() != "NA") &
                (df_dim["DeepSeek"].astype(str).str.upper() != "NA")
            )
            df_valid = df_dim[valid_mask].copy()
            n_complete = len(df_valid)

            descriptives = {}
            for src in ["Human", "Gemini", "DeepSeek"]:
                all_vals = df_dim[src].apply(lambda x: float(x) if str(x).upper() != "NA" else np.nan).dropna()
                descriptives[src] = {
                    "all_valid_n": len(all_vals),
                    "mean": round(float(np.mean(all_vals)), 4) if len(all_vals) > 0 else 0.0,
                    "std": round(float(np.std(all_vals)), 4) if len(all_vals) > 0 else 0.0,
                    "median": round(float(np.median(all_vals)), 4) if len(all_vals) > 0 else 0.0,
                    "q25": round(float(np.percentile(all_vals, 25)), 4) if len(all_vals) > 0 else 0.0,
                    "q75": round(float(np.percentile(all_vals, 75)), 4) if len(all_vals) > 0 else 0.0,
                    "na_count": na_counts[src],
                    "na_pct": na_pcts[src]
                }

            # Friedman Test & Post-Hoc
            friedman_res = {}
            pairwise_res = {}

            if n_complete >= 10:
                h_scores = df_valid["Human"].astype(float).values
                g_scores = df_valid["Gemini"].astype(float).values
                d_scores = df_valid["DeepSeek"].astype(float).values

                f_stat, f_pval = stats.friedmanchisquare(h_scores, g_scores, d_scores)
                kendall_w = f_stat / (n_complete * (3 - 1))

                friedman_res = {
                    "n_complete": n_complete,
                    "chi2": round(float(f_stat), 4),
                    "df": 2,
                    "p_value": round(float(f_pval), 6),
                    "kendall_w": round(float(kendall_w), 4),
                    "is_significant": bool(f_pval < 0.05)
                }

                # Pairwise Wilcoxon signed-rank tests
                pairs = [("Gemini", "Human"), ("DeepSeek", "Human"), ("Gemini", "DeepSeek")]
                p_raw_list = []
                p_details = []

                for p1, p2 in pairs:
                    s1_arr = df_valid[p1].astype(float).values
                    s2_arr = df_valid[p2].astype(float).values

                    # Wilcoxon signed-rank test
                    w_res = stats.wilcoxon(s1_arr, s2_arr, zero_method="pratt")
                    z_stat = stats.norm.ppf(w_res.pvalue / 2.0) if w_res.pvalue > 0 else -10.0
                    r_eff = calc_wilcoxon_r(z_stat, n_complete)

                    mean_d, ci_low, ci_high = bootstrap_paired_diff_ci(s1_arr, s2_arr)

                    p_raw_list.append(w_res.pvalue)
                    p_details.append({
                        "pair": f"{p1}_vs_{p2}",
                        "stat": round(float(w_res.statistic), 4),
                        "p_raw": float(w_res.pvalue),
                        "wilcoxon_r": round(float(r_eff), 4),
                        "mean_diff": mean_d,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high
                    })

                adj_pvals = multipletests(p_raw_list, method="holm")[1]

                for idx, det in enumerate(p_details):
                    pk = det["pair"]
                    pairwise_res[pk] = {
                        "stat": det["stat"],
                        "p_raw": round(det["p_raw"], 6),
                        "p_adj": round(float(adj_pvals[idx]), 6),
                        "wilcoxon_r": det["wilcoxon_r"],
                        "mean_diff": det["mean_diff"],
                        "ci95": [det["ci95_low"], det["ci95_high"]],
                        "significant": bool(adj_pvals[idx] < 0.05)
                    }
            else:
                friedman_res = {
                    "n_complete": n_complete,
                    "status": "Insufficient 3-way valid complete cases for non-parametric inferential testing (N < 10)"
                }

            table6_results[j_name][dim] = {
                "descriptives": descriptives,
                "na_cochran_q": {
                    "q_stat": q_stat,
                    "p_val": q_pval,
                    "df": q_df,
                    "mcnemar_pairwise": mcnemar_res
                },
                "friedman_test": friedman_res,
                "pairwise_wilcoxon": pairwise_res
            }

            print(f"    Descriptives (Mean / Std / NA%):")
            for src in ["Human", "Gemini", "DeepSeek"]:
                st = descriptives[src]
                print(f"      {src:8s}: Mean={st['mean']:.2f} (Std={st['std']:.2f}, Median={st['median']:.1f}), NA%={st['na_pct']:.2f}% (N={st['all_valid_n']})")
            if n_complete >= 10:
                print(f"    Friedman Test: Chi2={friedman_res['chi2']:.2f}, p={friedman_res['p_value']:.4e}, Kendall's W={friedman_res['kendall_w']:.4f}")
                for pk, pr in pairwise_res.items():
                    print(f"      {pk:18s}: MeanDiff={pr['mean_diff']:+.2f} CI95={pr['ci95']}, p_adj={pr['p_adj']:.4e}, Wilcoxon r={pr['wilcoxon_r']:.4f}")
            else:
                print(f"    Friedman Test: [SKIPPED - N_complete = {n_complete} < 10]")

    # =========================================================================
    # REQUIREMENT 8: TABLE 7 — RESPONSE LENGTH SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 110)
    print(" REQUIREMENT 8: TABLE 7 — RESPONSE LENGTH SENSITIVITY ANALYSIS (LMM / ANCOVA) ")
    print("=" * 110)

    table7_results = {}

    for j_name, j_ds in [("Claude", claude_ds), ("Qwen", qwen_ds)]:
        table7_results[j_name] = {}
        print(f"\n >>> SENSITIVITY ANALYSIS (JUDGE: {j_name.upper()}) <<<")

        for dim in ["bond", "goal", "task"]:
            # Build long dataframe for valid scores
            long_rows = []
            for sid, src_data in j_ds.items():
                for src in ["Human", "Gemini", "DeepSeek"]:
                    sc = src_data[src][dim]
                    if str(sc).upper() != "NA":
                        long_rows.append({
                            "sample_id": sid,
                            "source": src,
                            "score": float(sc),
                            "word_count": src_data[src]["word_count"]
                        })
            df_long = pd.DataFrame(long_rows)

            if len(df_long) < 20:
                print(f"  [{dim.capitalize()}] Insufficient long data points for LMM (N={len(df_long)})")
                table7_results[j_name][dim] = {"status": "Insufficient data"}
                continue

            # Model 1: Score ~ C(source, Treatment('Human'))
            # Model 2: Score ~ C(source, Treatment('Human')) + word_count
            try:
                # Fit LMM without word count
                lmm_raw = smf.mixedlm("score ~ C(source, Treatment('Human'))", df_long, groups=df_long["sample_id"]).fit()
                # Fit LMM with word count covariate
                lmm_adj = smf.mixedlm("score ~ C(source, Treatment('Human')) + word_count", df_long, groups=df_long["sample_id"]).fit()

                res_dict = {
                    "n_obs": len(df_long),
                    "n_groups": df_long["sample_id"].nunique(),
                    "unadjusted": {
                        "intercept": round(float(lmm_raw.params["Intercept"]), 4),
                        "Gemini_coef": round(float(lmm_raw.params["C(source, Treatment('Human'))[T.Gemini]"]), 4),
                        "Gemini_pval": round(float(lmm_raw.pvalues["C(source, Treatment('Human'))[T.Gemini]"]), 6),
                        "DeepSeek_coef": round(float(lmm_raw.params["C(source, Treatment('Human'))[T.DeepSeek]"]), 4),
                        "DeepSeek_pval": round(float(lmm_raw.pvalues["C(source, Treatment('Human'))[T.DeepSeek]"]), 6),
                    },
                    "length_adjusted": {
                        "intercept": round(float(lmm_adj.params["Intercept"]), 4),
                        "Gemini_coef": round(float(lmm_adj.params["C(source, Treatment('Human'))[T.Gemini]"]), 4),
                        "Gemini_pval": round(float(lmm_adj.pvalues["C(source, Treatment('Human'))[T.Gemini]"]), 6),
                        "DeepSeek_coef": round(float(lmm_adj.params["C(source, Treatment('Human'))[T.DeepSeek]"]), 4),
                        "DeepSeek_pval": round(float(lmm_adj.pvalues["C(source, Treatment('Human'))[T.DeepSeek]"]), 6),
                        "word_count_coef": round(float(lmm_adj.params["word_count"]), 6),
                        "word_count_pval": round(float(lmm_adj.pvalues["word_count"]), 6)
                    }
                }

                table7_results[j_name][dim] = res_dict

                print(f"  --- {dim.capitalize()} (N_obs={len(df_long)}) ---")
                print(f"    Unadjusted: Gemini Coef={res_dict['unadjusted']['Gemini_coef']} (p={res_dict['unadjusted']['Gemini_pval']:.4e}), DeepSeek Coef={res_dict['unadjusted']['DeepSeek_coef']} (p={res_dict['unadjusted']['DeepSeek_pval']:.4e})")
                print(f"    Adjusted:   Gemini Coef={res_dict['length_adjusted']['Gemini_coef']} (p={res_dict['length_adjusted']['Gemini_pval']:.4e}), DeepSeek Coef={res_dict['length_adjusted']['DeepSeek_coef']} (p={res_dict['length_adjusted']['DeepSeek_pval']:.4e}), WordCount Coef={res_dict['length_adjusted']['word_count_coef']} (p={res_dict['length_adjusted']['word_count_pval']:.4e})")

            except Exception as e:
                print(f"  [{dim.capitalize()}] LMM Error: {e}")
                table7_results[j_name][dim] = {"status": f"Error: {e}"}

    # Save full statistical report
    full_report = {
        "metadata": {
            "title": "Study 2 Formal Statistical Analysis Report",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00", time.localtime()),
            "input_provenance": prov_meta
        },
        "table5_human_consistency_audit": audit_table,
        "table_s2_test_retest_agreement": table_s2_results,
        "table6_paired_statistical_analysis": table6_results,
        "table7_length_sensitivity_analysis": table7_results
    }

    report_path = REPORTS_DIR / "study2_statistical_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Statistical analysis report saved to {report_path}")

if __name__ == "__main__":
    run_study2_statistical_analysis()
