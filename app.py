import os
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal, Optional
from openai import OpenAI
from langdetect import detect, LangDetectException

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL_NAME = "openai/gpt-oss-120b"

INJECTION_PATTERNS = [
    "ignore your previous instructions",
    "ignore previous instructions",
    "disregard your instructions",
    "you are now",
    "reply with only",
    "system prompt",
]

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    reason: str

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(request: ClassifyRequest):
    text = request.text

    if not text or not text.strip():
        return ClassifyResponse(label="refuse", confidence=0.0, reason="Empty input")

    lowered = text.lower()
    if any(p in lowered for p in INJECTION_PATTERNS):
        return ClassifyResponse(label="refuse", confidence=0.0, reason="Detected possible prompt injection")

    try:
        if detect(text) != "en":
            return ClassifyResponse(label="flag_for_human", confidence=0.0, reason="Non-English input")
    except LangDetectException:
        return ClassifyResponse(label="flag_for_human", confidence=0.0, reason="Could not detect language")

    prediction = classify(text)
    if prediction is None:
        return ClassifyResponse(label="refuse", confidence=0.0, reason="Upstream classifier failed after retry")

    if prediction.confidence < 0.6:
        return ClassifyResponse(label="flag_for_human", confidence=prediction.confidence, reason=prediction.reason)

    return ClassifyResponse(label=prediction.label, confidence=prediction.confidence, reason=prediction.reason)
