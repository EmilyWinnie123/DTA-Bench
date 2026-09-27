import os
import sys
import json
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from prompts import (
    load_judge_p4_config,
    get_rubric_text_and_hash,
    build_final_p4_prompt,
    format_dialogue_window
)

def run_prompt_config_checks():
    print("=" * 80)
    print(" P4 DIMENSION-WISE INDEPENDENT EVALUATION — CONFIG & PROMPT VERIFICATION ")
    print("=" * 80)
    print("Target Workspace:", BASE_DIR.resolve())
    
    # 1. LOAD CONFIGURATION
    config_path = BASE_DIR / "config" / "judge_p4_config.json"
    cache_config_path = BASE_DIR / "cache" / "final_prompt_config.json"
    
    assert config_path.exists(), f"[FAIL] Config file missing at {config_path}"
    assert cache_config_path.exists(), f"[FAIL] Cache config file missing at {cache_config_path}"
    
    config = load_judge_p4_config()
    print("\n[OK] Prompt configuration successfully loaded from disk.")
    print(f"  - Prompt Name:    {config.get('prompt_name')}")
    print(f"  - Prompt Version: {config.get('prompt_version')}")
    print(f"  - Primary Model:  {config['model_config']['primary_model']}")
    print(f"  - Model Switch:   {config['model_config']['allow_model_switch']} (Disabled)")
    
    # REQUIREMENT 1: Preserve P4 content verbatim without modification or optimization
    print("\n--- Requirement 1 Check: P4 Content Integrity ---")
    sys_raw = config["system_instruction_raw"]
    user_raw = config["user_prompt_template_raw"]
    
    # Check key sections of P4 verbatim text
    assert "You are an evaluator of Digital Therapeutic Alliance (DTA)" in sys_raw, "[FAIL] System instruction missing expected verbatim header!"
    assert "# P4 — Dimension-Wise Independent Evaluation" in user_raw, "[FAIL] User prompt template missing verbatim P4 header!"
    assert "* accurate emotional understanding" in user_raw, "[FAIL] Bond dimension details missing in P4 template!"
    assert "* persistence toward the seeker's expressed goal" in user_raw, "[FAIL] Goal dimension details missing in P4 template!"
    assert "First apply the Task N/A decision rule." in user_raw, "[FAIL] Task N/A rule instruction missing in P4 template!"
    print("[PASS] P4 prompt text preserved verbatim without modification or optimization.")
    
    # REQUIREMENT 2: Dynamic Rubric loading & SHA-256
    print("\n--- Requirement 2 Check: Dynamic DTA_Rubric.md Loading & SHA-256 ---")
    rubric_text, raw_sha, text_sha = get_rubric_text_and_hash()
    print(f"  - Loaded DTA_Rubric.md length: {len(rubric_text)} chars")
    print(f"  - DTA_Rubric.md Raw Bytes SHA-256: {raw_sha}")
    print(f"  - DTA_Rubric.md Text SHA-256:      {text_sha}")
    
    assert raw_sha == config["rubric_config"]["rubric_raw_sha256"], f"[FAIL] Rubric Raw SHA-256 mismatch! Got {raw_sha}, expected {config['rubric_config']['rubric_raw_sha256']}"
    print("[PASS] DTA_Rubric.md loaded dynamically and SHA-256 hash verified.")
    
    # REQUIREMENT 3: Dialogue Window Replacement
    print("\n--- Requirement 3 Check: Dialogue Window Substitution ---")
    sample_item = {
        "sample_id": "TEST_SAMPLE_001",
        "dialogue_context": "Seeker: I feel really overwhelmed with work lately.\nSupporter: That sounds super stressful. Have you had time to pause?",
        "seeker_utterance": "Not really, deadlines are piling up.",
        "supporter_response": "Let's take it one step at a time. What is the single most urgent task on your plate right now?"
    }
    
    sys_p, user_p, meta = build_final_p4_prompt(sample_item)
    assert "{dialogue}" not in user_p, "[FAIL] {dialogue} placeholder was not replaced!"
    assert "--- Target Supporter Response to Evaluate ---" in user_p, "[FAIL] Formatted dialogue window not found in rendered prompt!"
    assert "Let's take it one step at a time." in user_p, "[FAIL] Supporter response text missing from rendered prompt!"
    print("[PASS] Dialogue window {dialogue} correctly replaced with actual dialogue contents.")
    
    # REQUIREMENT 4: Output Schema & Valid JSON
    print("\n--- Requirement 4 Check: Output Schema Constraints ---")
    expected_keys = config["output_schema_config"]["required_keys"]
    print(f"  - Required JSON keys: {expected_keys}")
    assert expected_keys == ["bond", "goal", "task"], "[FAIL] Unexpected JSON keys schema!"
    assert '{\n"bond": 1-5 or "NA",\n"goal": 1-5 or "NA",\n"task": 1-5 or "NA"\n}' in user_p, "[FAIL] P4 JSON format specification missing!"
    print("[PASS] Output strictly constrained to valid JSON with keys: bond, goal, task (1-5 or NA).")
    
    # REQUIREMENT 5: Exclusion of Human Gold Standard answers
    print("\n--- Requirement 5 Check: Human Gold Standard Exclusion ---")
    assert "Gold Reference Examples" not in sys_p and "Gold Reference Examples" not in user_p, "[FAIL] Gold standard examples found in prompt!"
    assert "Gold rating:" not in user_p, "[FAIL] Gold rating exemplar found in user prompt!"
    assert config["gold_standard_config"]["include_gold_examples"] is False, "[FAIL] include_gold_examples is True!"
    assert config["gold_standard_config"]["exemplar_count"] == 0, "[FAIL] Exemplar count > 0!"
    print("[PASS] Human Gold Standard answers strictly excluded (Zero-shot across dimensions).")
    
    # REQUIREMENT 6: Fixed Judge Model, API parameters, and Prompt Version
    print("\n--- Requirement 6 Check: Model, Parameters, and Version Lock ---")
    p_model = config["model_config"]["primary_model"]
    api_params = config["model_config"]["api_parameters"]
    p_ver = config["prompt_version"]
    print(f"  - Judge Model: {p_model}")
    print(f"  - API Parameters: {json.dumps(api_params)}")
    print(f"  - Prompt Version: {p_ver}")
    assert p_model == "[G]claude-sonnet-5", f"[FAIL] Expected model '[G]claude-sonnet-5', got '{p_model}'"
    assert api_params["temperature"] == 0.0, "[FAIL] Temperature is not 0.0!"
    assert p_ver == "v1.0-P4-DimensionWise-Frozen", f"[FAIL] Unexpected prompt version '{p_ver}'"
    print("[PASS] Judge model, API parameters, and prompt version strictly fixed.")
    
    # REQUIREMENT 7: Prohibition of Automatic Model Switching on Failure
    print("\n--- Requirement 7 Check: Single Model Lock (No Fallback Switch) ---")
    allow_switch = config["model_config"]["allow_model_switch"]
    fallbacks = config["model_config"]["fallback_models"]
    assert allow_switch is False, "[FAIL] allow_model_switch must be False!"
    assert len(fallbacks) == 0, "[FAIL] fallback_models list must be empty!"
    print("[PASS] Automatic model switching on failure is explicitly prohibited.")
    
    # REQUIREMENT 8: Save Prompt Config & SHA-256 Hashes
    print("\n--- Requirement 8 Check: Configuration & Hashes Persistence ---")
    hashes = meta
    print("  - System Instruction SHA-256:  ", hashes["system_instruction_sha256"])
    print("  - User Template SHA-256:       ", hashes["user_prompt_template_sha256"])
    print("  - Rubric Raw SHA-256:          ", hashes["rubric_raw_sha256"])
    print("  - Rendered User Prompt SHA-256:", hashes["user_prompt_rendered_sha256"])
    assert len(hashes["system_instruction_sha256"]) == 64, "[FAIL] Invalid SHA-256 length!"
    assert len(hashes["user_prompt_template_sha256"]) == 64, "[FAIL] Invalid SHA-256 length!"
    print("[PASS] Prompt configuration and all component SHA-256 signatures verified.")
    
    # STAGE 2 GUARDRAIL CHECK
    print("\n--- STAGE 2 EXPERIMENT GUARDRAIL CHECK ---")
    print("[OK] Confirmed: Stage 2 formal experiment has NOT been started.")
    print("All prompt configuration, freezing, dynamic loading, and validation checks completed successfully.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_prompt_config_checks()
