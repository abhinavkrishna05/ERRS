"""
ERSS Triage Classification — LLM-based via Groq (Llama 3)
==========================================================
Uses Groq API with Llama 3 to classify emergency call transcripts.
No hardcoded keyword dictionaries — the LLM generalizes natively
across English, Hindi, and Hinglish.

This module exposes the `TriageClassifier` class for easy integration 
into web backends or other modules.
"""

import csv
import os
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = """\
You are a triage classification agent for India's ERSS 112 emergency helpline.

Classify each call transcript as EXACTLY one of:
- "Emergency" — genuine emergency needing immediate dispatch (fire, medical crisis, violent crime, accident, disaster, domestic violence with physical harm)
- "Non-Emergency" — non-urgent (information queries, civic complaints, community complaints, wrong numbers, pocket dials, verbal disputes)

CRITICAL RULES:
1. You MUST understand Hindi (Devanagari), Hinglish (Hindi in Roman script), and English.
2. Provide an English translation of the native transcript to support downstream processing and human verification.
3. EMERGENCY triggers: fire, smoke, burning, bleeding, unconscious, not breathing, choking, drowning, assault with weapon, armed robbery, kidnapping, serious accident, collapse, explosion, domestic violence with physical beating, child in physical danger.
4. NON-EMERGENCY: driving license, office timings, documents, garbage, water supply, stray dogs, parking disputes, noise complaints, pocket dials, wrong numbers, verbal arguments (no weapons/physical violence), lost items, civic infrastructure issues.
5. IMPORTANT: Verbal arguments, parking disputes, neighbor quarrels, and shouting matches WITHOUT weapons or physical violence are NON-EMERGENCY — these are community complaints, not emergencies.
6. If call is cut mid-sentence with distress keywords before cut → "Emergency".
7. If call is cut mid-sentence with NO distress context → "Non-Emergency" if text is clearly non-emergency.

RESPOND WITH ONLY valid JSON:
{
  "decision": "Emergency" or "Non-Emergency",
  "confidence": 0.0-1.0,
  "department": "Fire|Ambulance|Police|Civic Complaint|Community Complaint|Information|Wrong Number",
  "reasoning": "brief reason",
  "english_translation": "Translated English text of the caller's audio"
}
"""

@dataclass
class TriageResult:
    case_id: str
    decision: str       
    confidence: float
    predicted_type: str   
    predicted_department: str
    reasoning: str
    english_translation: str


class TriageClassifier:
    """
    A reusable classifier module that interfaces with Groq's LLMs 
    for ERSS triage classification.
    """
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set! Please define it in your .env file or environment variables."
            )
        self.client = Groq(api_key=api_key)

    def classify_call(self, case_id: str, description: str, max_retries: int = 3) -> TriageResult:
        """Classify a single transcript using the LLM."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Classify this call:\n{description}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=250,
                )

                raw = response.choices[0].message.content
                result = json.loads(raw)

                decision = result.get("decision", "Emergency")
                
                return TriageResult(
                    case_id=case_id,
                    decision=decision,
                    confidence=float(result.get("confidence", 0.5)),
                    predicted_type=decision,
                    predicted_department=result.get("department", ""),
                    reasoning=result.get("reasoning", ""),
                    english_translation=result.get("english_translation", "")
                )

            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                return TriageResult(case_id, "Emergency", 0.5, "Emergency", "", "JSON parse error", "")

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
                return TriageResult(case_id, "Emergency", 0.5, "Emergency", "", f"API error: {err_str[:80]}", "")

        return TriageResult(case_id, "Emergency", 0.5, "Emergency", "", "All retries exhausted", "")

    def classify_single(self, transcript: str) -> Dict[str, Any]:
        """Convenience method for classifying a single transcript and returning a dict."""
        r = self.classify_call("interactive", transcript)
        return {
            "decision": r.decision,
            "confidence": r.confidence,
            "predicted_type": r.predicted_type,
            "predicted_department": r.predicted_department,
            "reasoning": r.reasoning,
            "english_translation": r.english_translation
        }

def load_dataset(csv_path: str) -> list:
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            records.append(row)
    return records


def evaluate(csv_path: str, model: str = "llama-3.3-70b-versatile", sample_size: int = None):
    """Evaluate the LLM classifier on a CSV dataset."""
    classifier = TriageClassifier(model=model)

    print("=" * 80)
    print(f"  ERSS TRIAGE — LLM Evaluation (model: {model})")
    print(f"  Dataset: {csv_path}")
    print("=" * 80)

    records = load_dataset(csv_path)
    print(f"\nLoaded {len(records)} records")

    if sample_size and sample_size < len(records):
        import random
        random.seed(42)
        records = random.sample(records, sample_size)
        print(f"Sampled {sample_size} records for evaluation")

    total = correct = 0
    type_stats = {"Emergency": {"total": 0, "correct": 0},
                  "Non-Emergency": {"total": 0, "correct": 0}}
    lang_stats = {"Hindi": {"total": 0, "correct": 0},
                  "Hinglish": {"total": 0, "correct": 0},
                  "English": {"total": 0, "correct": 0}}
    errors = []

    for i, row in enumerate(records):
        desc = row.get("description", "").strip()
        actual_type = row.get("type", "").strip()
        lang = row.get("language", "").strip()
        case_id = row.get("case_id", "").strip()
        if not desc:
            continue

        result = classifier.classify_call(case_id, desc)
        time.sleep(1)  
        pred = result.predicted_type
        ok = pred == actual_type

        total += 1
        if ok:
            correct += 1
        if actual_type in type_stats:
            type_stats[actual_type]["total"] += 1
            if ok:
                type_stats[actual_type]["correct"] += 1
        if lang in lang_stats:
            lang_stats[lang]["total"] += 1
            if ok:
                lang_stats[lang]["correct"] += 1
        if not ok:
            errors.append({
                "id": case_id, "desc": desc[:80], "actual": actual_type,
                "pred": pred, "lang": lang, "reason": result.reasoning
            })

        if (i + 1) % 25 == 0:
            acc_so_far = correct / total * 100
            print(f"  [{i+1}/{len(records)}] Running accuracy: {acc_so_far:.1f}%")

    acc = correct / total * 100 if total else 0

    print("\n" + "-" * 80)
    print(f"  OVERALL ACCURACY: {correct}/{total} ({acc:.1f}%)")
    print("-" * 80)

    print(f"\n  {'Type':<18} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    for t, s in type_stats.items():
        a = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {t:<18} {s['correct']:>8} {s['total']:>8} {a:>7.1f}%")

    print(f"\n  {'Language':<18} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    for l, s in lang_stats.items():
        a = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {l:<18} {s['correct']:>8} {s['total']:>8} {a:>7.1f}%")

    if errors:
        print(f"\nMisclassified ({len(errors)} total, first 10):")
        for e in errors[:10]:
            print(f"  [{e['id']}] {e['lang']}: {e['desc']}")
            print(f"    Actual={e['actual']} Pred={e['pred']} | {e['reason']}")

    return acc, errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ERSS Triage Classifier (Groq + Llama)")
    parser.add_argument("--dataset", default="synthetic_emergency_dataset.csv",
                        help="CSV file to evaluate")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq model name")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N rows (saves API calls for testing)")
    parser.add_argument("--text", type=str, default=None,
                        help="Classify a single transcript instead of dataset")
    args = parser.parse_args()

    if args.text:
        classifier = TriageClassifier(model=args.model)
        result = classifier.classify_single(args.text)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n>>> EVALUATING ON:", args.dataset, "<<<")
        acc1, _ = evaluate(args.dataset, model=args.model, sample_size=args.sample)

        gen_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "generated_test_data.csv")
        if os.path.exists(gen_path) and args.dataset != gen_path:
            print(f"\n>>> EVALUATING ON GENERATED TEST DATA <<<")
            acc2, _ = evaluate(gen_path, model=args.model, sample_size=args.sample)
            print("\n" + "=" * 80)
            print(f"  Original: {acc1:.1f}% | Generated: {acc2:.1f}%")
            print("=" * 80)
