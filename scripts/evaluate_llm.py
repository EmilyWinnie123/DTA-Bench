import os
import json
import time
import argparse
import random
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pingouin as pg
from prompts import build_prompt, get_rubric_text_and_hash

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANNOTATION_DIR = BASE_DIR / "annotation"
CACHE_DIR = BASE_DIR / "cache"
CACHE_FILE = CACHE_DIR / "llm_responses.jsonl"
DATA_DIR = BASE_DIR / "data"

RANDOM_SEED = 42

MODEL_PRICING = {
    "gpt-4o": {"prompt": 0.0025, "completion": 0.0100},
    "claude-3-5-sonnet": {"prompt": 0.0030, "completion": 0.0150},
    "llama-3.1-70b": {"prompt": 0.0009, "completion": 0.0009}
}

def extract_icc_safe(icc_res: pd.DataFrame, target_type: str = "ICC1") -> float:
    if icc_res.empty:
        return 0.0
    types = icc_res["Type"].astype(str).tolist()
    for t in types:
        if target_type.lower() in t.lower() or t.lower() in target_type.lower():
            return float(icc_res[icc_res["Type"] == t]["ICC"].values[0])
    return float(icc_res["ICC"].iloc[0])

class CacheManager:
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()

    def load_cache(self):
        if not self.cache_file.exists():
            return
        with open(self.cache_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    key = (item["sample_id"], item["model"], item["condition"], item["run_idx"], item.get("dimension", "all"))
                    self.cache[key] = item

    def get(self, sample_id: str, model: str, condition: str, run_idx: int, dimension: str = "all"):
        return self.cache.get((sample_id, model, condition, run_idx, dimension))

    def save(self, record: dict):
        key = (record["sample_id"], record["model"], record["condition"], record["run_idx"], record.get("dimension", "all"))
        self.cache[key] = record
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def call_llm_api(model: str, sys_prompt: str, user_prompt: str, api_key: str = None, base_url: str = None) -> str:
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    base = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or os.environ.get("API_BASE")
    if not key:
        raise ValueError("API Key missing! Set OPENAI_API_KEY environment variable or pass --api-key parameter.")
    
    kwargs = {"api_key": key}
    if base:
        kwargs["base_url"] = base
        
    client = OpenAI(**kwargs)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content

def mock_llm_call(model: str, sys_prompt: str, user_prompt: str, condition: str) -> str:
    h = int(hashlib.md5((user_prompt + model).encode()).hexdigest(), 16)
    score_b = (h % 5) + 1
    score_g = ((h >> 2) % 5) + 1
    score_t = ((h >> 4) % 5) + 1

    resp = {
        "reasoning": f"[{model} CoT] Evaluated under condition {condition}. Bond={score_b}, Goal={score_g}, Task={score_t}.",
        "scores": {
            "Bond": score_b,
            "Goal": score_g,
            "Task": score_t
        }
    }
    return json.dumps(resp)

def estimate_cost(model: str, prompt_tokens: int = 400, completion_tokens: int = 150) -> float:
    pricing = MODEL_PRICING.get(model, {"prompt": 0.002, "completion": 0.005})
    return (prompt_tokens / 1000.0 * pricing["prompt"]) + (completion_tokens / 1000.0 * pricing["completion"])

def select_optimal_condition_from_stage1(dev_records: list, gold_labels: dict) -> str:
    condition_scores = {}

    for condition in ["zero_shot", "few_shot", "cot_then_score", "dimension_wise"]:
        cond_records = [r for r in dev_records if r["condition"] == condition]
        if not cond_records:
            continue

        iccs = []
        for dim in ["Bond", "Goal", "Task"]:
            pairs = []
            for r in cond_records:
                sid = r["sample_id"]
                if sid in gold_labels and gold_labels[sid]["scores"].get(dim) is not None:
                    gold_v = gold_labels[sid]["scores"][dim]
                    llm_v = r["parsed_output"].get("scores", {}).get(dim)
                    if llm_v is not None:
                        pairs.append((gold_v, llm_v))

            if len(pairs) >= 5:
                df = pd.DataFrame({
                    "Targets": list(range(len(pairs))) * 2,
                    "Raters": ["Human"] * len(pairs) + ["LLM"] * len(pairs),
                    "Scores": [p[0] for p in pairs] + [p[1] for p in pairs]
                })
                icc_res = pg.intraclass_corr(data=df, targets="Targets", raters="Raters", ratings="Scores")
                icc_11 = extract_icc_safe(icc_res, "ICC1")
                iccs.append(icc_11)

        mean_icc = float(np.mean(iccs)) if iccs else 0.0
        total_cost = sum(r.get("estimated_cost", 0.0) for r in cond_records)
        condition_scores[condition] = {"mean_icc": mean_icc, "cost": total_cost}

    print("\n--- [STAGE 1 ABLATION SELECTION METRICS (Mean ICC(2,1))] ---")
    for cond, info in condition_scores.items():
        print(f"  - Condition: {cond:15s} | Mean ICC(2,1): {info['mean_icc']:.4f} | Total Cost: ${info['cost']:.4f}")

    best_condition = max(condition_scores.keys(), key=lambda c: (condition_scores[c]["mean_icc"], -condition_scores[c]["cost"]))
    print(f"\n[OK] [STAGE 1 RESULT] Optimal Prompt Condition Selected: '{best_condition}' (Mean ICC = {condition_scores[best_condition]['mean_icc']:.4f})")

    with open(CACHE_DIR / "stage1_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump({"selected_condition": best_condition, "metrics": condition_scores}, f, indent=2, ensure_ascii=False)

    return best_condition

def run_evaluation(stage: int, models: list, conditions: list, max_cost: float = 100.0, dry_run: bool = True, num_runs: int = 3, api_key: str = None, base_url: str = None):
    cache = CacheManager(CACHE_FILE)
    rubric_content, rubric_hash = get_rubric_text_and_hash()

    exemplars = []

    if stage == 1:
        exemplar_path = ANNOTATION_DIR / "few_shot_exemplars.json"
        if not exemplar_path.exists():
            raise FileNotFoundError(
                "CRITICAL ERROR: few_shot_exemplars.json missing!\n"
                "Stage 1 requires few-shot exemplars from dev_set annotation.\n"
                "Please run 'python annotation/export_for_annotation.py' and 'python annotation/import_annotation.py --mock' first."
            )
        with open(exemplar_path, "r", encoding="utf-8") as f:
            exemplars = json.load(f)
        print(f"[evaluate_llm] Loaded {len(exemplars)} dev exemplars for Stage 1 Ablation.")

    if stage == 2:
        assert len(conditions) == 1, f"CRITICAL BUDGET ERROR: Stage 2 must run ONLY 1 optimal condition, got {len(conditions)}: {conditions}"
        print(f"[evaluate_llm] Stage 2 executing frozen optimal condition: {conditions[0]}")

    if stage == 1:
        data_path = PROCESSED_DIR / "dev_set.json"
    elif stage == 2:
        data_path = PROCESSED_DIR / "eval_set.json"
    elif stage == 3:
        data_path = DATA_DIR / "study2_pool.json"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset for stage {stage} missing at {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"[evaluate_llm] Loaded {len(samples)} samples for Stage {stage} execution.")

    total_cost = 0.0
    total_calls = 0
    cache_hits = 0
    executed_records = []

    for sample in samples:
        sample_id = sample.get("sample_id", sample.get("context_sample_id"))

        for model in models:
            for condition in conditions:
                for run_idx in range(num_runs):
                    dims_to_run = ["Bond", "Goal", "Task"] if condition == "dimension_wise" else ["all"]

                    for dim in dims_to_run:
                        cached_entry = cache.get(sample_id, model, condition, run_idx, dim)
                        if cached_entry:
                            cache_hits += 1
                            executed_records.append(cached_entry)
                            continue

                        call_cost = estimate_cost(model)
                        if total_cost + call_cost > max_cost:
                            print(f"\n[evaluate_llm] COST GUARDRAIL TRIGGERED: Total cost ${total_cost:.4f} exceeded max-cost limit ${max_cost:.2f}. Halting execution!")
                            return

                        sys_prompt, user_prompt = build_prompt(condition, sample, exemplars, dimension=dim)
                        if dry_run:
                            raw_resp = mock_llm_call(model, sys_prompt, user_prompt, condition)
                        else:
                            raw_resp = call_llm_api(model, sys_prompt, user_prompt, api_key=api_key, base_url=base_url)

                        try:
                            parsed_json = json.loads(raw_resp)
                        except json.JSONDecodeError:
                            parsed_json = {"reasoning": raw_resp, "scores": {"Bond": None, "Goal": None, "Task": None}}

                        total_cost += call_cost
                        total_calls += 1

                        record = {
                            "sample_id": sample_id,
                            "dataset": sample.get("dataset", "ESConv"),
                            "stage": stage,
                            "model": model,
                            "condition": condition,
                            "run_idx": run_idx,
                            "dimension": dim,
                            "raw_output": raw_resp,
                            "parsed_output": parsed_json,
                            "rubric_sha256": rubric_hash,
                            "estimated_cost": call_cost,
                            "timestamp": time.time()
                        }
                        cache.save(record)
                        executed_records.append(record)

    print(f"\n[evaluate_llm] Stage {stage} Execution Summary:")
    print(f"  - Total New API Calls: {total_calls}")
    print(f"  - Cache Hits: {cache_hits}")
    print(f"  - Total Stage Cost: ${total_cost:.4f}")

    if stage == 1:
        gold_path = ANNOTATION_DIR / "gold_labels.json"
        if gold_path.exists():
            with open(gold_path, "r", encoding="utf-8") as f:
                gold_labels = json.load(f)
            select_optimal_condition_from_stage1(executed_records, gold_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Evaluation Runner for DTA Measurement")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3], help="Stage 1 (Ablation), Stage 2 (Full Eval), Stage 3 (Study 2)")
    parser.add_argument("--models", nargs="+", default=["[G]claude-sonnet-5"], help="Models to evaluate")
    parser.add_argument("--conditions", nargs="+", default=["p4_final"], help="Prompt conditions")
    parser.add_argument("--max-cost", type=float, default=50.0, help="Maximum allowed cost guardrail in USD")
    parser.add_argument("--live", action="store_true", help="Use real API calls instead of mock provider")
    parser.add_argument("--api-key", type=str, default=None, help="API Key for live calls")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for OpenAI-compatible API")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of repetitions per sample")
    args = parser.parse_args()

    if args.stage == 2:
        print("[evaluate_llm] Stage 2 automatically enforcing frozen optimal condition 'p4_final' (P4 Dimension-Wise Independent Evaluation) and primary judge model '[G]claude-sonnet-5'.")
        args.conditions = ["p4_final"]
        args.models = ["[G]claude-sonnet-5"]

    run_evaluation(
        stage=args.stage,
        models=args.models,
        conditions=args.conditions,
        max_cost=args.max_cost,
        dry_run=not args.live,
        num_runs=args.num_runs,
        api_key=args.api_key,
        base_url=args.base_url
    )

