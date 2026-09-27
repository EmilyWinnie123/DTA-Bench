# Digital Therapeutic Alliance (DTA) Evaluation Framework & Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: DTA-Rubric v1.0](https://img.shields.io/badge/Rubric-DTA--v1.0-green.svg)](rubric/DTA_Rubric.md)

This repository provides the complete open-source implementation, dataset benchmarks, LLM Judge evaluation pipeline, and empirical statistical analysis for evaluating **Digital Therapeutic Alliance (DTA)** in text-based supportive dialogue.

---

## 📌 Project Overview

The **Digital Therapeutic Alliance (DTA)** framework measures the therapeutic relationship between a seeker and a supporter across three independent dimensions:
1. **Bond (情感纽带)**: Emotional understanding, acceptance/non-judgment, situational personalization, and willingness to remain with distress.
2. **Goal (目标对齐)**: Goal elicitation, understanding checks, directional alignment, and persistence toward seeker goals.
3. **Task (任务协同)**: Actionability, rationale, collaborative framing, and timing (with a global Task N/A decision rule for non-intervention/exploration phases).

### Core Research Components

* **Study 1: LLM Judge Calibration & Prompt Ablation**
  * Expert Human Annotation across 190 dialogue samples (20 Calibration, 20 Dev, 150 Official Eval).
  * Prompt Ablation Study comparing 4 prompt architectures (P1 Zero-Shot, P2 Structured, P3 Few-Shot, P4 Dimension-Wise Independent Evaluation).
  * LLM Judge validation across **Claude Sonnet 5** (`[G]claude-sonnet-5`) and **Qwen3.8-Max** (`Qwen3.8-Max`).

* **Study 2: AI vs. Human Response Evaluation & Dual-Judge Blind Audit**
  * Context-preserving response generation from **Gemini 3.8 Flash** (Generator A) and **DeepSeek V4 Pro** (Generator B) across 150 dialogue samples (300 AI responses + 150 Human Gold = 450 total responses).
  * **Dual-Judge Blind Evaluation**: Claude Sonnet 5 and Qwen3.8-Max independently rate all 450 randomized, anonymized responses (`anon_001` – `anon_450`, seed=42; 900 ratings total).
  * Non-parametric statistical analysis (Friedman test, Holm-adjusted Wilcoxon signed-rank test, Cochran's Q test, McNemar test, and Linear Mixed-Effects Model length-sensitivity analysis).

---

## 📁 Repository Structure & File Index

```
dta_project/
├── .env.example                       # Environment variables template (API Keys & Endpoints)
├── .gitignore                          # Git ignore rule file (Excludes secrets & local caches)
├── requirements.txt                    # Project Python package dependencies
├── README.md                           # Main project documentation (This file)
│
├── config/                             # Experiment & Model Configurations
│   └── judge_p4_config.json            # Frozen P4 Prompt & Judge API configuration
│
├── rubric/                             # Clinical Evaluation Standards
│   └── DTA_Rubric.md                   # Frozen DTA-Rubric v1.0 specification
│
├── data/                               # Datasets (Raw & Processed)
│   ├── raw/                            # Original public datasets (ESConv & AnnoMI)
│   │   ├── ESConv.json                 # Emotional Support Conversation dataset
│   │   ├── AnnoMI-simple.csv           # AnnoMI Motivational Interviewing dataset
│   │   └── annomi.parquet              # Parquet version of AnnoMI
│   ├── processed/                      # Preprocessed & split evaluation sets
│   │   ├── dev_set.json                # Development set (20 samples)
│   │   └── eval_set.json               # Main evaluation set (150 samples)
│   ├── study2_pool.json                # Extracted 150 evaluation contexts for Study 2
│   └── study2_task_precheck.json       # Pre-check audit records for Study 2 inputs
│
├── annotation/                         # Human Annotation & Gold Standard Pipelines
│   ├── export_for_annotation.py        # Exports dataset windows to annotator Excel sheets
│   ├── export_main_annotation.py       # Exports 150 main evaluation samples for annotation
│   ├── export_calibration_set.py       # Exports 20 calibration samples for annotator alignment
│   ├── generate_gold_standard.py       # Adjudicates annotator ratings into Gold Standard
│   ├── import_annotation.py            # Imports and parses rated Excel files
│   ├── gold_labels.json                # Consolidated Gold Standard labels (Calibration/Dev/Eval)
│   ├── gold_sample_ids.json            # Official list of sample IDs for benchmark
│   ├── calibration_sample_metadata.json# Metadata for calibration dialogue windows
│   ├── few_shot_exemplars.json         # Few-shot exemplar candidates for prompt ablation
│   ├── round1_agreement_report.json    # Inter-annotator agreement metrics (QWK, ICC, MAE)
│   └── sheets/                         # Annotator Excel sheets & adjudication files
│
├── scripts/                            # Core Pipeline & Execution Scripts
│   ├── prompts.py                      # Prompt templates (P1-P4) & prompt builder helpers
│   ├── stage1_prompt_ablation.py       # Executes Stage 1 prompt ablation (P1, P2, P3, P4)
│   ├── run_live_stage1.py              # Runs Stage 1 live evaluations
│   ├── run_live_stage2_eval.py         # Runs Stage 2 live eval for Primary Judge (Claude Sonnet 5)
│   ├── run_qwen_stage2_eval.py         # Runs Stage 2 live eval for Validation Judge (Qwen3.8-Max)
│   ├── run_deepseek_stage2_eval.py     # Runs Stage 2 live eval for DeepSeek V4 Pro Judge
│   ├── run_study2_generation.py        # Generates AI responses (Gemini 3.8 Flash & DeepSeek V4 Pro)
│   ├── run_study2_dual_judge_eval.py   # Anonymizes 450 items, runs Dual-Judge eval, de-anonymizes
│   ├── run_study2_statistical_analysis.py # Formal statistical pipeline (Friedman, Wilcoxon, LMM)
│   ├── compute_round1_agreement.py     # Calculates human annotator agreement statistics
│   ├── preprocess_data.py              # Data preprocessing and dialogue window extraction
│   ├── check_final_prompt_config.py    # Audits prompt hashes and configuration integrity
│   ├── download_data.py                # Data downloader dispatcher
│   ├── download_real_esconv.py         # Fetches official ESConv dataset from GitHub
│   ├── download_real_annomi.py         # Fetches official AnnoMI dataset
│   ├── download_official_annomi.py     # Alternate official AnnoMI downloader
│   ├── fetch_annomi_parquet.py         # Downloads HuggingFace AnnoMI Parquet dataset
│   ├── scan_annomi.py                  # Scans AnnoMI structure and utterance attributes
│   ├── search_github_annomi.py         # Utility to discover open-source AnnoMI repositories
│   └── test_*.py                       # Unit tests and API connectivity checks
│
├── cache/                              # Experiment Caches & Evaluation Outputs
│   ├── llm_responses.jsonl             # Cached raw LLM API response logs
│   ├── stage1_p*.json                  # Stage 1 prompt ablation prediction outputs
│   ├── stage2_p4_predictions.json      # Study 1 Claude Sonnet 5 P4 evaluation predictions
│   ├── qwen_stage2_p4_predictions.json # Study 1 Qwen3.8-Max P4 evaluation predictions
│   ├── study2_ai_responses.json        # 450 raw responses (Human, Gemini, DeepSeek) + metadata
│   ├── study2_anonymization_map.json   # Randomized anonymization mapping (anon_001-450, seed=42)
│   ├── study2_claude_eval_predictions.json # Study 2 Claude Sonnet 5 predictions (450 items)
│   ├── study2_qwen_eval_predictions.json   # Study 2 Qwen3.8-Max predictions (450 items)
│   ├── study2_dual_judge_summary.json  # Study 2 de-anonymized dual-judge agreement summary
│   └── study2_generation_audit_report.json # Pre-evaluation audit report for Study 2 data
│
└── reports/                            # Generated Statistical Analysis Reports
    └── study2_statistical_analysis_report.json # Full Study 2 statistical report (Tables 5, 6, 7)
```

---

## 🔒 Security, Privacy & Environment Configuration

### 1. Environment Variable Isolation (`.env`)
To ensure security, **no API keys or gateway credentials are hardcoded** in any script. All scripts dynamically load credentials from environment variables using `python-dotenv`.

1. Copy the template configuration file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your model API keys:
   ```env
   # Claude API Credentials
   CLAUDE_API_KEY=your_claude_api_key_here
   CLAUDE_API_URL=https://your-api-endpoint/v1/chat/completions

   # Qwen API Credentials
   QWEN_API_KEY=your_qwen_api_key_here
   QWEN_API_URL=https://your-api-endpoint/v1/chat/completions

   # Gemini API Credentials
   GEMINI_API_KEY=your_gemini_api_key_here

   # DeepSeek API Credentials
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

### 2. Privacy & PII Protection
* All dialogue samples from **ESConv** and **AnnoMI** have been sanitized and anonymized.
* Phone number patterns and email regular expressions have been thoroughly audited across all dataset files, cache files, and logs to ensure no Personally Identifiable Information (PII) is included in the codebase.
* Public crisis hotlines (e.g., `1-800-273-8255`) present in public datasets are preserved as part of supportive dialogue context.

---

## ⚙️ Installation & Usage

### 1. Installation
Clone the repository and install the Python dependencies:
```bash
git clone https://github.com/your-username/dta_project.git
cd dta_project
pip install -r requirements.txt
```

### 2. Running Study 2 Response Generation
To generate 300 AI responses (150 Gemini 3.8 Flash + 150 DeepSeek V4 Pro) using neutral system prompts and frozen context:
```bash
python scripts/run_study2_generation.py
```

### 3. Running Dual-Judge Blind Evaluation
To anonymize the 450 responses (seed=42) and run independent blind evaluations using Claude Sonnet 5 and Qwen3.8-Max:
```bash
python scripts/run_study2_dual_judge_eval.py
```

### 4. Running Formal Statistical Analysis
To compute Study 1 vs Study 2 consistency (Table 5), non-parametric paired tests (Friedman, Holm-adjusted Wilcoxon, Cochran's Q; Table 6), and Linear Mixed-Effects Model length-sensitivity analysis (Table 7):
```bash
python scripts/run_study2_statistical_analysis.py
```

Outputs will be saved in `reports/study2_statistical_analysis_report.json`.

---

## 📊 Benchmark Results Summary

### Dual-Judge Inter-Rater Agreement (Claude Sonnet 5 vs Qwen3.8-Max)
- **Macro ICC(2,1)** = `0.6184` (Good Reliability)
- **Bond ICC(2,1)** = `0.7662` (High Consistency)
- **Goal ICC(2,1)** = `0.5119`
- **Task N/A Agreement** = `87.78%`

### Main Statistical Conclusions (Study 2)
1. **Bond & Goal Superiority**: Both Claude Sonnet 5 and Qwen3.8-Max judges show that **Gemini 3.8 Flash** and **DeepSeek V4 Pro** score significantly higher than Human Gold responses on Bond and Goal dimensions ($p_{\text{adj}} < 0.0001$, Wilcoxon $r > 0.68$).
2. **Length-Sensitivity Check**: Linear Mixed-Effects Models controlling for response English word count confirm that AI model superiority remains strongly significant ($p < 0.001$), demonstrating that alliance quality improvement is substantive rather than a byproduct of response length.

---

## 📜 Citation & License

This project is released under the [MIT License](LICENSE). 

If you use this benchmark, rubric, or code in your research, please cite:
```bibtex
@article{dta_bench_2026,
  title={Evaluating Digital Therapeutic Alliance in Supportive Dialogue: A Dual-Judge Blind Evaluation Benchmark},
  author={DTA Project Team},
  year={2026}
}
```
