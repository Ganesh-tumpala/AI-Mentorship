import os
import sys
import json
import time
import re
import numpy as np
from langdetect import detect, LangDetectException
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional
from sklearn.metrics import classification_report, precision_recall_fscore_support, cohen_kappa_score

F1_THRESHOLD = 0.65
MODEL_NAME = "openai/gpt-oss-120b"

INJECTION_PATTERNS = [
    "ignore your previous instructions",
    "ignore previous instructions",
    "disregard your instructions",
    "you are now",
    "reply with only",
    "system prompt",
]

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

class NewsLabel(BaseModel):
    label: Literal["World", "Sports", "Business", "Sci/Tech"]
    confidence: float = Field(ge=0, le=1)
    reason: str

SYSTEM = """You classify news items into exactly one of four topics: World, Sports, Business, Sci/Tech.
Reply with ONLY a JSON object matching this schema, nothing else:
{"label": "<one of the four>", "confidence": <0 to 1>, "reason": "<one short sentence>"}"""

def classify(text: str) -> Optional[NewsLabel]:
    for attempt in range(2):
        try:
            r = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": text}],
                temperature=0,
            )
            raw = r.choices[0].message.content.strip()
            data = json.loads(raw)
            return NewsLabel(**data)
        except Exception:
            if attempt == 1:
                return None
    return None

def guarded_classify(text: str) -> str:
    if not text or not text.strip():
        return "refuse"
    lowered = text.lower()
    if any(p in lowered for p in INJECTION_PATTERNS):
        return "refuse"
    try:
        if detect(text) != "en":
            return "flag_for_human"
    except LangDetectException:
        return "flag_for_human"
    prediction = classify(text)
    if prediction is None:
        return "refuse"
    if prediction.confidence < 0.6:
        return "flag_for_human"
    return prediction.label

def run_evaluation(golden_set):
    results = []
    latencies = []
    for item in golden_set:
        start = time.perf_counter()
        predicted_label = guarded_classify(item["text"])
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        results.append({
            "id": item["id"],
            "type": item["type"],
            "expected": item["expected_label"],
            "predicted": predicted_label,
            "latency": elapsed,
        })
    return results, latencies

def main():
    with open("golden_set.json") as f:
        golden_set = json.load(f)

    print(f"Running evaluation on {len(golden_set)} items...")
    results, latencies = run_evaluation(golden_set)

    benchmark_results = [r for r in results if r["type"] == "benchmark"]
    y_true = [r["expected"] for r in benchmark_results]
    y_pred = [r["predicted"] for r in benchmark_results]

    print("\n=== Benchmark classification report ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    kappa = cohen_kappa_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"Macro F1: {f1:.3f}")

    adv_results = [r for r in results if r["type"] != "benchmark"]
    adv_correct = sum(1 for r in adv_results if r["predicted"] == r["expected"])
    print(f"\n=== Adversarial cases: {adv_correct}/{len(adv_results)} handled as expected ===")
    for r in adv_results:
        status = "PASS" if r["predicted"] == r["expected"] else "FAIL"
        print(f"  [{status}] {r['id']}: expected={r['expected']}, got={r['predicted']}")

    lat_array = np.array(latencies)
    print(f"\n=== Latency ===")
    print(f"Mean: {lat_array.mean():.2f}s")
    print(f"P95:  {np.percentile(lat_array, 95):.2f}s")

    n_calls = len(benchmark_results)
    est_cost = n_calls * ((150 / 1_000_000 * 0.15) + (40 / 1_000_000 * 0.60))
    print(f"Estimated cost for this run: ${est_cost:.5f}")

    print(f"\n=== Gate check ===")
    print(f"F1 threshold: {F1_THRESHOLD}")
    if f1 < F1_THRESHOLD:
        print(f"FAILED: Macro F1 {f1:.3f} is below threshold {F1_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"PASSED: Macro F1 {f1:.3f} meets threshold {F1_THRESHOLD}")
        sys.exit(0)

if __name__ == "__main__":
    main()
