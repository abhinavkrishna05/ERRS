import os
import json
import time
import csv
from dataclasses import dataclass
from groq import Groq

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ["GROQ_API_KEY"] = user_secrets.get_secret("GROQ_API_KEY")

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


SYSTEM_PROMPT = """\
You are a triage classification agent for India's ERSS 112 emergency helpline.

Classify each call transcript as EXACTLY one of:
- "Emergency" — genuine emergency needing immediate dispatch (fire, medical crisis, violent crime, accident, disaster, domestic violence with physical harm)
- "Non-Emergency" — non-urgent (information queries, civic complaints, community complaints, wrong numbers, pocket dials, verbal disputes)

CRITICAL RULES:
1. You MUST understand Hindi (Devanagari), Hinglish (Hindi in Roman script), and English.
2. EMERGENCY triggers: fire, smoke, burning, bleeding, unconscious, not breathing, choking, drowning, assault with weapon, armed robbery, kidnapping, serious accident, collapse, explosion, domestic violence with physical beating, child in physical danger.
3. NON-EMERGENCY: driving license, office timings, documents, garbage, water supply, stray dogs, parking disputes, noise complaints, pocket dials, wrong numbers, verbal arguments (no weapons/physical violence), lost items, civic infrastructure issues.
4. IMPORTANT: Verbal arguments, parking disputes, neighbor quarrels, and shouting matches WITHOUT weapons or physical violence are NON-EMERGENCY — these are community complaints, not emergencies.
5. If call is cut mid-sentence with distress keywords before cut → "Emergency".
6. If call is cut mid-sentence with NO distress context → "Non-Emergency" if text is clearly non-emergency.

RESPOND WITH ONLY valid JSON:
{"decision": "Emergency" or "Non-Emergency", "confidence": 0.0-1.0, "department": "Fire|Ambulance|Police|Civic Complaint|Community Complaint|Information|Wrong Number", "reasoning": "brief reason"}
"""

@dataclass
class TriageResult:
    case_id: str
    decision: str
    confidence: float
    predicted_type: str
    predicted_department: str
    reasoning: str

def classify_call(case_id: str, description: str, model: str = "llama-3.3-70b-versatile", max_retries: int = 3) -> TriageResult:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this call:\n{description}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=200,
            )

            raw = response.choices[0].message.content
            result = json.loads(raw)

            decision = result.get("decision", "Emergency")
            confidence = float(result.get("confidence", 0.5))
            department = result.get("department", "")
            reasoning = result.get("reasoning", "")

            return TriageResult(case_id, decision, confidence, decision, department, reasoning)

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            return TriageResult(case_id, "Emergency", 0.5, "Emergency", "", "JSON parse error")

        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                wait = min(30, 2 ** (attempt + 2))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return TriageResult(case_id, "Emergency", 0.5, "Emergency", "", f"API error: {err_str[:80]}")
            
    return TriageResult(case_id, "Emergency", 0.5, "Emergency", "", "All retries exhausted")


def evaluate(csv_path: str, model: str = "llama-3.1-8b-instant", sample_size: int = None):
    

    print("=" * 80)
    print(f"  ERSS TRIAGE — LLM Evaluation (model: {model})")
    print("=" * 80)

    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            records.append(row)
            
    print(f"\nLoaded {len(records)} records")

    if sample_size and sample_size < len(records):
        import random
        random.seed(42)
        records = random.sample(records, sample_size)
        print(f"Sampled {sample_size} records for evaluation")

    total = correct = 0
    type_stats = {"Emergency": {"total": 0, "correct": 0}, "Non-Emergency": {"total": 0, "correct": 0}}
    lang_stats = {"Hindi": {"total": 0, "correct": 0}, "Hinglish": {"total": 0, "correct": 0}, "English": {"total": 0, "correct": 0}}
    errors = []

    for i, row in enumerate(records):
        desc = row.get("description", "").strip()
        actual_type = row.get("type", "").strip()
        lang = row.get("language", "").strip()
        case_id = row.get("case_id", "").strip()
        if not desc: continue

        result = classify_call(case_id, desc, model=model)
        time.sleep(1)  
        
        pred = result.predicted_type
        ok = pred == actual_type

        total += 1
        if ok: correct += 1
        if actual_type in type_stats:
            type_stats[actual_type]["total"] += 1
            if ok: type_stats[actual_type]["correct"] += 1
        if lang in lang_stats:
            lang_stats[lang]["total"] += 1
            if ok: lang_stats[lang]["correct"] += 1
        if not ok:
            errors.append({"id": case_id, "desc": desc[:80], "actual": actual_type, "pred": pred, "reason": result.reasoning})

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(records)}] Running accuracy: {correct / total * 100:.1f}%")

    print("\n" + "-" * 80)
    print(f"  OVERALL ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")
    print("-" * 80)

    if errors:
        print(f"\nMisclassified ({len(errors)} total, first 10):")
        for e in errors[:10]:
            print(f"  [{e['id']}] Actual={e['actual']} Pred={e['pred']} | Reason: {e['reason']}")


classify_call(1,"hello... I lost my wallet somewhere near the bus stand","llama-3.1-8b-instant")



DATASET_PATH = "/kaggle/input/datasets/mayankjha1202/testdata-erss/synthetic_emergency_dataset.csv"
GENERATED_TEST_PATH = "/kaggle/input/datasets/mayankjha1202/testdata-erss/generated_test_data.csv"

SAMPLE_SIZE = 100

print("\n>>> EVALUATING ON TRAINING DATA <<<")
evaluate(DATASET_PATH, sample_size=SAMPLE_SIZE)

print("\n>>> EVALUATING ON NOVEL GENERATED DATA <<<")
evaluate(GENERATED_TEST_PATH, sample_size=SAMPLE_SIZE)