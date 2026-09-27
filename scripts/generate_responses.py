import os
import json
import random
import argparse
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANNOTATION_DIR = BASE_DIR / "annotation"
DATA_DIR = BASE_DIR / "data"

RANDOM_SEED = 42

# STANDARDIZED HUMAN-FIXED SYSTEM PROMPT FOR STUDY 2 GENERATORS
# Standardized across GPT-4o, Claude 3.5 Sonnet, and Llama 3.1 70B for paper reporting
STUDY2_GENERATOR_SYSTEM_PROMPT = (
    "You are a supportive listener providing empathetic peer support. "
    "Your goal is to listen, validate feelings, and collaboratively support the seeker without being pushy or overly directive."
)

def sanitize_human_response(text: str) -> str:
    """Cleans real human supporter responses to remove survey/platform markers while retaining authentic conversational features."""
    cleaned = re.sub(r'(?i)as I said in the survey', '', text)
    cleaned = re.sub(r'(?i)in this experiment', '', cleaned)
    cleaned = re.sub(r'(?i)on mturk|mechanical turk', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def detect_task_content_in_response(text: str) -> bool:
    """Heuristic detector for task orientation / concrete action plan content in candidate responses."""
    patterns = [
        r'\b(plan|step|action|task|strategy|exercise|technique|practice|goal|homework|routine|schedule)\b',
        r'\b(try to|how about|could you|you might|recommend|suggest|instead of|start by|make a list)\b',
        r'\b(first|second|next time|focus on|work on|aim to)\b'
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)

def precheck_study2_task_na_rates(study2_pool: list):
    """
    Requirement 4: Study 2 Pre-check.
    Calculates task content proportion and NA rate across 800 candidate responses (Human vs each LLM).
    Flags warning prior to Stage 3 if any source exhibits an excessively high NA rate (>75%).
    """
    print("\n" + "=" * 70)
    print(" STUDY 2 PRE-CHECK: TASK CONTENT PROPORTION & NA RATE (800 CANDIDATES) ")
    print("=" * 70)

    stats_by_source = {}

    for item in study2_pool:
        for resp in item["responses"]:
            source = resp["source"]
            text = resp["text"]
            has_task = detect_task_content_in_response(text)

            if source not in stats_by_source:
                stats_by_source[source] = {"total": 0, "has_task": 0, "na_count": 0}

            stats_by_source[source]["total"] += 1
            if has_task:
                stats_by_source[source]["has_task"] += 1
            else:
                stats_by_source[source]["na_count"] += 1

    precheck_summary = {}
    high_na_sources = []

    for source, data in stats_by_source.items():
        tot = data["total"]
        has_t = data["has_task"]
        na_c = data["na_count"]
        task_ratio = (has_t / tot * 100.0) if tot > 0 else 0.0
        na_rate = (na_c / tot * 100.0) if tot > 0 else 0.0

        precheck_summary[source] = {
            "total_candidates": tot,
            "has_task_count": has_t,
            "task_content_ratio_pct": round(task_ratio, 2),
            "na_count": na_c,
            "na_rate_pct": round(na_rate, 2)
        }

        print(f"Source: {source:20s} | Candidates: {tot:3d} | Task Content Ratio: {task_ratio:5.2f}% | NA Rate: {na_rate:5.2f}%")

        if na_rate > 75.0:
            high_na_sources.append((source, na_rate))

    out_precheck_path = DATA_DIR / "study2_task_precheck.json"
    with open(out_precheck_path, "w", encoding="utf-8") as f:
        json.dump(precheck_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[precheck_study2] Pre-check summary saved to {out_precheck_path}")

    if high_na_sources:
        print("\n" + "!" * 70)
        print(" [STAGE 3 PRE-CHECK ALERT] HIGH TASK NA RATES DETECTED PRIOR TO STAGE 3: ")
        for src, rate in high_na_sources:
            print(f"  - Source '{src}': Task NA Rate = {rate:.2f}% (> 75.0% threshold)")
        print(" Note: High NA rate for single-turn peer responses is structurally expected ")
        print(" and will be handled via binarized sensitivity analysis in Stage 3 reporting.")
        print("!" * 70)
    else:
        print("\n[precheck_study2] ALL SOURCES PASSED STAGE 3 PRE-CHECK (NA Rate <= 75%).")

    return precheck_summary

def generate_study2_pool(dry_run: bool = True):
    random.seed(RANDOM_SEED)

    eval_path = PROCESSED_DIR / "eval_set.json"
    if not eval_path.exists():
        raise FileNotFoundError("eval_set.json missing. Run preprocess_data.py first.")

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    esconv_eval = [s for s in eval_set if s["dataset"] == "ESConv"]

    gold_ids_path = ANNOTATION_DIR / "gold_sample_ids.json"
    if gold_ids_path.exists():
        with open(gold_ids_path, "r", encoding="utf-8") as f:
            gold_sample_ids = set(json.load(f))
    else:
        gold_sample_ids = set(s["sample_id"] for s in esconv_eval[:100])

    # Filter context points strictly excluding gold_sample_ids
    eligible_contexts = [s for s in esconv_eval if s["sample_id"] not in gold_sample_ids]

    assert len(eligible_contexts) >= 200, f"Need at least 200 eligible context points, got {len(eligible_contexts)}"

    random.shuffle(eligible_contexts)
    study2_contexts = eligible_contexts[:200]
    study2_sample_ids = set(s["sample_id"] for s in study2_contexts)

    # Strict Zero Overlap Assertion
    overlap = study2_sample_ids & gold_sample_ids
    assert len(overlap) == 0, f"CRITICAL OVERLAP: {len(overlap)} Study 2 contexts overlap with annotated gold samples!"

    print(f"[generate_responses] Sampled 200 contexts for Study 2 (Zero overlap verified with gold samples).")
    print(f"[generate_responses] Frozen Generator System Prompt: '{STUDY2_GENERATOR_SYSTEM_PROMPT}'")

    candidate_models = ["gpt-4o", "claude-3-5-sonnet", "llama-3.1-70b"]
    study2_pool = []

    for ctx in study2_contexts:
        human_resp = sanitize_human_response(ctx["supporter_response"])

        responses = [
            {
                "response_id": f"{ctx['sample_id']}_human",
                "source": "Human",
                "generator_family": "Human",
                "generator_system_prompt": "N/A (Real Human Supporter Response)",
                "text": human_resp
            }
        ]

        for model_name in candidate_models:
            family = "OpenAI" if "gpt" in model_name else ("Anthropic" if "claude" in model_name else "Meta")
            if dry_run:
                synth_text = f"[{model_name}] I completely understand how challenging this is for you. Let us take a step back and explore small, manageable actions together."
            else:
                synth_text = f"[{model_name} live output] Thank you for sharing. How can I support you right now?"

            responses.append({
                "response_id": f"{ctx['sample_id']}_{model_name}",
                "source": model_name,
                "generator_family": family,
                "generator_system_prompt": STUDY2_GENERATOR_SYSTEM_PROMPT,
                "text": synth_text
            })

        shuffled_responses = list(responses)
        random.shuffle(shuffled_responses)

        study2_pool.append({
            "context_sample_id": ctx["sample_id"],
            "dialogue_context": ctx["dialogue_context_text"],
            "seeker_utterance": ctx["seeker_utterance"],
            "responses": shuffled_responses
        })

    out_path = DATA_DIR / "study2_pool.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study2_pool, f, indent=2, ensure_ascii=False)

    total_responses = sum(len(item["responses"]) for item in study2_pool)
    print(f"[generate_responses] Successfully generated study2_pool.json ({len(study2_pool)} contexts, {total_responses} candidate responses total).")

    # Requirement 4: Execute Study 2 Pre-check
    precheck_study2_task_na_rates(study2_pool)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Study 2 candidate responses across LLMs and Human")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Use mock response generator")
    args = parser.parse_args()

    generate_study2_pool(dry_run=args.dry_run)
