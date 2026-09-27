import json
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RUBRIC_PATH = BASE_DIR / "rubric" / "DTA_Rubric.md"
CONFIG_PATH = BASE_DIR / "config" / "judge_p4_config.json"
CACHE_CONFIG_PATH = BASE_DIR / "cache" / "final_prompt_config.json"

def get_rubric_text_and_hash() -> tuple[str, str, str]:
    """Dynamically loads frozen DTA_Rubric.md and computes SHA-256 hashes (raw bytes & text)."""
    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(f"Rubric file missing at {RUBRIC_PATH}")
    
    raw_bytes = RUBRIC_PATH.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    
    text_content = RUBRIC_PATH.read_text(encoding="utf-8")
    text_sha256 = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
    
    return text_content, raw_sha256, text_sha256

def load_judge_p4_config() -> dict:
    """Loads and returns the frozen P4 Judge Prompt configuration."""
    target_config_path = CONFIG_PATH if CONFIG_PATH.exists() else CACHE_CONFIG_PATH
    if not target_config_path.exists():
        raise FileNotFoundError(f"P4 Judge prompt configuration missing at {target_config_path}")
    
    with open(target_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def format_dialogue_window(sample: dict) -> str:
    """Formats dialogue context, seeker utterance, and supporter response into a dialogue window."""
    ctx = str(sample.get("dialogue_context", sample.get("dialogue_context_text", ""))).strip()
    su = str(sample.get("seeker_utterance", "")).strip()
    sr = str(sample.get("supporter_response", sample.get("text", ""))).strip()

    parts = []
    if ctx and ctx.lower() not in ["nan", "none", "null", ""]:
        parts.append(f"--- Dialogue Context ---\n{ctx}")
    if su and su.lower() not in ["nan", "none", "null", ""]:
        parts.append(f"Seeker: {su}")
    if sr and sr.lower() not in ["nan", "none", "null", ""]:
        parts.append(f"--- Target Supporter Response to Evaluate ---\nSupporter: {sr}")
    
    return "\n\n".join(parts) if parts else "[Empty Dialogue Window]"

def build_final_p4_prompt(sample: dict) -> tuple[str, str, dict]:
    """
    Builds the final frozen P4 (Dimension-Wise Independent Evaluation) system & user prompt.
    - Dynamically loads frozen DTA_Rubric.md
    - Verifies rubric SHA-256 hash
    - Replaces {dialogue} with actual dialogue window
    - Excludes Human Gold Standard answers
    - Returns (system_prompt, user_prompt, metadata)
    """
    config = load_judge_p4_config()
    rubric_text, raw_sha, text_sha = get_rubric_text_and_hash()

    # Validate Rubric Integrity
    expected_raw_sha = config["rubric_config"]["rubric_raw_sha256"]
    expected_text_sha = config["rubric_config"]["rubric_text_sha256"]
    if raw_sha != expected_raw_sha and text_sha != expected_text_sha:
        raise ValueError(
            f"FROZEN RUBRIC SHA-256 MISMATCH!\n"
            f"Expected raw SHA-256: {expected_raw_sha}\n"
            f"Actual raw SHA-256:   {raw_sha}\n"
            f"Actual text SHA-256:  {text_sha}"
        )

    common_sys_instruction = config["system_instruction_raw"]
    sys_prompt = f"{common_sys_instruction}\n\n=== FROZEN DTA-RUBRIC V1.0 FULL TEXT ===\n{rubric_text}\n=== END OF RUBRIC ==="

    dialogue_window_str = format_dialogue_window(sample)
    user_template = config["user_prompt_template_raw"]

    if "{dialogue}" not in user_template:
        raise ValueError("P4 user prompt template missing '{dialogue}' placeholder!")

    user_prompt = user_template.replace("{dialogue}", dialogue_window_str)

    meta = {
        "prompt_version": config["prompt_version"],
        "prompt_name": config["prompt_name"],
        "primary_model": config["model_config"]["primary_model"],
        "allow_model_switch": config["model_config"]["allow_model_switch"],
        "api_parameters": config["model_config"]["api_parameters"],
        "rubric_raw_sha256": raw_sha,
        "system_instruction_sha256": hashlib.sha256(common_sys_instruction.encode("utf-8")).hexdigest(),
        "user_prompt_template_sha256": hashlib.sha256(user_template.encode("utf-8")).hexdigest(),
        "user_prompt_rendered_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
    }

    return sys_prompt, user_prompt, meta

SYSTEM_PROMPT = """You are an expert clinical supervisor and psychological researcher evaluating digital therapeutic alliance (DTA) in text-based peer support and AI therapy dialogues.
You must adhere strictly to the provided evaluation rubric and output valid JSON matching the requested schema."""

def build_prompt(condition: str, sample: dict, exemplars: list = None, dimension: str = None) -> tuple[str, str]:
    """Builds system and user prompts for specified condition. Returns (system_prompt, user_prompt)."""
    if condition in ["p4_final", "p4"]:
        sys_p, user_p, _ = build_final_p4_prompt(sample)
        return sys_p, user_p

    rubric_text, _, _ = get_rubric_text_and_hash()

    context_str = sample.get("dialogue_context_text", sample.get("dialogue_context", ""))
    seeker_utt = sample.get("seeker_utterance", "")
    supporter_resp = sample.get("supporter_response", sample.get("text", ""))

    input_text = f"Dialogue History:\n{context_str}\n\nSeeker Utterance:\n{seeker_utt}\n\nSupporter Response:\n{supporter_resp}"

    if condition == "zero_shot":
        user_prompt = f"""Rubric Guidelines:\n{rubric_text}\n\n{input_text}\n\nEvaluate Bond, Goal, and Task scores on a 1-5 scale (or null if N/A). Return JSON strictly matching this schema:
{{
  "reasoning": "Short justification",
  "scores": {{
    "Bond": <int or null>,
    "Goal": <int or null>,
    "Task": <int or null>
  }}
}}"""

    elif condition == "few_shot":
        exemplar_str = ""
        if exemplars:
            exemplar_str = "Gold Reference Examples:\n"
            for idx, ex in enumerate(exemplars, 1):
                exemplar_str += f"--- Example {idx} ---\n{ex.get('dialogue_context', '')}\nSeeker: {ex.get('seeker_utterance', '')}\nSupporter: {ex.get('supporter_response', '')}\nGold Scores: {json.dumps(ex.get('scores', {}))}\n\n"

        user_prompt = f"""Rubric Guidelines:\n{rubric_text}\n\n{exemplar_str}{input_text}\n\nEvaluate Bond, Goal, and Task scores. Return JSON:
{{
  "reasoning": "Short justification",
  "scores": {{
    "Bond": <int or null>,
    "Goal": <int or null>,
    "Task": <int or null>
  }}
}}"""

    elif condition == "cot_then_score":
        user_prompt = f"""Rubric Guidelines:\n{rubric_text}\n\n{input_text}\n\nProvide step-by-step psychological clinical reasoning analyzing emotional warmth (Bond), goal alignment (Goal), and collaborative action (Task). After your reasoning, output final integer scores (1-5 or null).
Return JSON strictly:
{{
  "reasoning": "Detailed step-by-step clinical analysis of Bond, Goal, and Task...",
  "scores": {{
    "Bond": <int or null>,
    "Goal": <int or null>,
    "Task": <int or null>
  }}
}}"""

    elif condition == "dimension_wise":
        dim_target = dimension if dimension else "Bond"
        user_prompt = f"""Rubric Guidelines:\n{rubric_text}\n\n{input_text}\n\nFocus exclusively on evaluating the **{dim_target}** dimension. Return JSON:
{{
  "reasoning": "Justification for {dim_target}",
  "scores": {{
    " {dim_target}": <int or null>
  }}
}}"""

    else:
        raise ValueError(f"Unknown prompt condition: {condition}")

    return SYSTEM_PROMPT, user_prompt

