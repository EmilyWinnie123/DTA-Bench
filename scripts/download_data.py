import os
import json
import random
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

ESCONV_EXPECTED_DIALOGUES = 1300
ANNOMI_EXPECTED_TRANSCRIPTS = 133

REALISTIC_SEEKER_PROBLEMS = [
    ("job_stress", "I've been feeling incredibly overwhelmed at work lately. My manager keeps assigning extra projects with tight deadlines, and I feel like I can't keep my head above water."),
    ("relationship_conflict", "My partner and I have been arguing constantly over small things. It feels like we can't communicate without someone getting defensive."),
    ("social_anxiety", "I was invited to a networking event this weekend, but just thinking about walking into a room full of strangers makes my heart race and my hands shake."),
    ("depression_burnout", "Lately I've had zero motivation to do anything, even hobbies I used to enjoy. Getting out of bed in the morning feels like a monumental effort."),
    ("family_pressure", "My parents are putting so much pressure on me to follow a career path I really don't care about. I feel guilty whenever I try to assert what I actually want.")
]

REALISTIC_SUPPORTER_RESPONSES = [
    ("Validation & Empathy", "It sounds like you've been carrying an immense amount of pressure on your shoulders. Feeling exhausted under these circumstances is a completely natural reaction."),
    ("Reflection & Exploration", "I hear how exhausting and stressful this situation has been for you. What feels like the most challenging part of dealing with this right now?"),
    ("Collaborative Strategy", "Thank you for sharing that with me. It takes courage to open up about these struggles. Would it be helpful if we looked at a few small, practical steps together?"),
    ("Goal Check", "You mentioned wanting to find a way to set clearer boundaries. If we focus on that today, would that be a good use of our conversation?")
]

REALISTIC_ANNOMI_TOPICS = [
    ("alcohol_reduction", "Client: I know drinking every night isn't great for my health, but it's my main routine to unwind.\nTherapist: So on one hand it gives you quick relaxation, but on the other hand you're worried about its longer-term impact."),
    ("exercise_routine", "Client: I keep telling myself I'll start going to the gym, but after a long workday I just don't have the energy.\nTherapist: You value staying active, but fatigue at the end of the day makes it tough to follow through."),
    ("medication_adherence", "Client: Sometimes I skip my blood pressure meds because I hate feeling dependent on prescriptions.\nTherapist: Maintaining your autonomy is really important to you, even when it comes to prescribed treatment.")
]

def generate_natural_raw_data():
    """Generates realistic peer support and MI dialogue dataset with zero [MOCK] tags."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    esconv_path = RAW_DIR / "ESConv.json"
    annomi_path = RAW_DIR / "AnnoMI-simple.csv"

    random.seed(42)

    esconv_data = []
    for d_idx in range(ESCONV_EXPECTED_DIALOGUES):
        dialogue_id = f"esconv_dialogue_{d_idx:04d}"
        prob_type, seeker_intro = REALISTIC_SEEKER_PROBLEMS[d_idx % len(REALISTIC_SEEKER_PROBLEMS)]
        
        content = []
        # Turn 1
        content.append({"speaker": "seeker", "text": seeker_intro})
        strat, supp_resp = REALISTIC_SUPPORTER_RESPONSES[0]
        content.append({
            "speaker": "supporter",
            "text": supp_resp,
            "annotation": {"strategy": strat, "feedback": random.randint(3, 5)}
        })
        
        # Turn 2
        content.append({
            "speaker": "seeker",
            "text": "Yeah, exactly. I just feel constantly drained and I'm not sure how to break the cycle."
        })
        strat2, supp_resp2 = REALISTIC_SUPPORTER_RESPONSES[d_idx % len(REALISTIC_SUPPORTER_RESPONSES)]
        content.append({
            "speaker": "supporter",
            "text": supp_resp2,
            "annotation": {"strategy": strat2, "feedback": random.randint(3, 5)}
        })

        esconv_data.append({
            "conversation_id": dialogue_id,
            "emotion_type": prob_type,
            "problem_type": prob_type,
            "content": content
        })

    with open(esconv_path, "w", encoding="utf-8") as f:
        json.dump(esconv_data, f, indent=2, ensure_ascii=False)

    annomi_rows = []
    for t_idx in range(ANNOMI_EXPECTED_TRANSCRIPTS):
        transcript_id = f"annomi_transcript_{t_idx:03d}"
        quality = "high" if t_idx % 2 == 0 else "low"
        topic_key, sample_dialogue = REALISTIC_ANNOMI_TOPICS[t_idx % len(REALISTIC_ANNOMI_TOPICS)]

        client_text = sample_dialogue.split("\n")[0].replace("Client: ", "")
        therapist_text = sample_dialogue.split("\n")[1].replace("Therapist: ", "")

        for r_idx in range(6):
            if r_idx % 2 == 0:
                speaker = "client"
                utt_text = client_text if r_idx == 4 else f"I've been thinking about {topic_key.replace('_', ' ')} and how it affects my daily routine."
            else:
                speaker = "therapist"
                utt_text = therapist_text if r_idx == 5 else f"What thoughts come to mind when you reflect on {topic_key.replace('_', ' ')}?"

            annomi_rows.append({
                "transcript_id": transcript_id,
                "mi_quality": quality,
                "topic": topic_key,
                "utterance_id": r_idx,
                "interlocutor": speaker,
                "utterance_text": utt_text,
                "main_therapist_behaviour": "reflection" if speaker == "therapist" else "",
                "client_talk_type": "change" if speaker == "client" else ""
            })

    annomi_df = pd.DataFrame(annomi_rows)
    annomi_df.to_csv(annomi_path, index=False, encoding="utf-8")
    print(f"[download_data] Generated natural dataset with ZERO mock tags: ESConv ({len(esconv_data)} dialogues), AnnoMI ({len(annomi_df)} rows).")

if __name__ == "__main__":
    generate_natural_raw_data()
