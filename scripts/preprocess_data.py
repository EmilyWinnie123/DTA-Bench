import os
import json
import random
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RANDOM_SEED = 42

def preprocess_data():
    random.seed(RANDOM_SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    esconv_path = RAW_DIR / "ESConv.json"
    annomi_path = RAW_DIR / "AnnoMI-simple.csv"

    if not esconv_path.exists() or not annomi_path.exists():
        raise FileNotFoundError("Raw data files not found. Run scripts/download_data.py first.")

    with open(esconv_path, "r", encoding="utf-8") as f:
        esconv_raw = json.load(f)

    annomi_df = pd.read_csv(annomi_path, encoding="utf-8")

    # 1. Transcript-level split into dev pool (15%) and eval pool (85%)
    esconv_ids = [d.get("conversation_id", f"esconv_{idx}") for idx, d in enumerate(esconv_raw)]
    annomi_ids = sorted(list(annomi_df["transcript_id"].unique()))

    random.shuffle(esconv_ids)
    random.shuffle(annomi_ids)

    n_esconv_dev = max(1, int(len(esconv_ids) * 0.15))
    n_annomi_dev = max(1, int(len(annomi_ids) * 0.15))

    esconv_dev_ids = set(esconv_ids[:n_esconv_dev])
    esconv_eval_ids = set(esconv_ids[n_esconv_dev:])

    annomi_dev_ids = set(annomi_ids[:n_annomi_dev])
    annomi_eval_ids = set(annomi_ids[n_annomi_dev:])

    assert len(esconv_dev_ids & esconv_eval_ids) == 0, "ESConv dev/eval transcript leak!"
    assert len(annomi_dev_ids & annomi_eval_ids) == 0, "AnnoMI dev/eval transcript leak!"

    # 2. Extract window samples from ESConv
    esconv_dev_windows = []
    esconv_eval_windows = []

    for d_idx, d in enumerate(esconv_raw):
        cid = d.get("conversation_id", f"esconv_{d_idx}")
        pool_type = "dev" if cid in esconv_dev_ids else "eval"
        history = []
        dialog_content = d.get("content", d.get("dialog", []))

        for idx, utt in enumerate(dialog_content):
            speaker_str = utt.get("speaker", "")
            text_str = utt.get("text", utt.get("content", ""))
            history.append({
                "speaker": speaker_str,
                "text": text_str
            })

            annotation_obj = utt.get("annotation", {})
            strategy = annotation_obj.get("strategy", "General") if isinstance(annotation_obj, dict) else "General"
            feedback_val = annotation_obj.get("feedback") if isinstance(annotation_obj, dict) else None

            if speaker_str == "supporter":
                context_slice = history[-6:]
                seeker_utt = history[-2]["text"] if len(history) >= 2 and history[-2]["speaker"] == "seeker" else ""

                sample_obj = {
                    "sample_id": f"{cid}_w{idx}",
                    "dataset": "ESConv",
                    "transcript_id": cid,
                    "pool_type": pool_type,
                    "emotion_type": d.get("emotion_type", ""),
                    "strategy": strategy,
                    "context": context_slice,
                    "dialogue_context_text": "\n".join([f"{item['speaker'].capitalize()}: {item['text']}" for item in context_slice[:-1]]),
                    "seeker_utterance": seeker_utt,
                    "supporter_response": text_str,
                    "seeker_feedback": feedback_val
                }

                if pool_type == "dev":
                    esconv_dev_windows.append(sample_obj)
                else:
                    esconv_eval_windows.append(sample_obj)

    # 3. Extract sliding window samples from AnnoMI
    annomi_dev_windows = []
    annomi_eval_windows = []

    for tid, group in annomi_df.groupby("transcript_id"):
        pool_type = "dev" if tid in annomi_dev_ids else "eval"
        group_rows = group.to_dict("records")
        mi_quality = group_rows[0].get("mi_quality", "unknown")

        for i in range(0, len(group_rows) - 5, 4):
            win_rows = group_rows[i:i+6]
            supporter_turn = win_rows[-1]
            if supporter_turn.get("interlocutor") != "therapist":
                continue

            seeker_utt = win_rows[-2]["utterance_text"] if len(win_rows) >= 2 else ""

            sample_obj = {
                "sample_id": f"{tid}_w{i}",
                "dataset": "AnnoMI",
                "transcript_id": tid,
                "pool_type": pool_type,
                "mi_quality": mi_quality,
                "topic": win_rows[0].get("topic", ""),
                "context": win_rows,
                "dialogue_context_text": "\n".join([f"{r['interlocutor'].capitalize()}: {r['utterance_text']}" for r in win_rows[:-1]]),
                "seeker_utterance": seeker_utt,
                "supporter_response": supporter_turn["utterance_text"],
                "seeker_feedback": None
            }

            if pool_type == "dev":
                annomi_dev_windows.append(sample_obj)
            else:
                annomi_eval_windows.append(sample_obj)

    dev_set = esconv_dev_windows + annomi_dev_windows
    eval_set = esconv_eval_windows + annomi_eval_windows

    dev_transcripts = set(s["transcript_id"] for s in dev_set)
    eval_transcripts = set(s["transcript_id"] for s in eval_set)
    overlap = dev_transcripts & eval_transcripts
    assert len(overlap) == 0, f"Critical Leak: {len(overlap)} transcript IDs overlap!"

    with open(PROCESSED_DIR / "dev_set.json", "w", encoding="utf-8") as f:
        json.dump(dev_set, f, indent=2, ensure_ascii=False)

    with open(PROCESSED_DIR / "eval_set.json", "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    print(f"[preprocess_data] Generated dev_set.json ({len(dev_set)} samples) and eval_set.json ({len(eval_set)} samples). Zero leakage verified.")

if __name__ == "__main__":
    preprocess_data()
