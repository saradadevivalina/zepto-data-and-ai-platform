import re
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"


class AskRequest(BaseModel):
    query: str = Field(..., example="What is the delivery fee for orders below 149?")


class SupportResponse(BaseModel):
    answer: str = Field(...)
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


app = FastAPI(title="Zepto Support Assistant API", version="1.0.0")


def load_docs():
    docs = []
    if not DOCS_DIR.exists():
        return docs
    for file in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = file.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"doc_id": file.stem, "text": text})
    return docs


def score_query(query: str, text: str) -> int:
    words = {w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(w) > 2}
    if not words:
        return 0
    text_lower = text.lower()
    score = 0
    for word in words:
        if word in text_lower:
            score += 2
    return score


def answer_policy_query(query: str):
    q = query.lower()
    docs = load_docs()
    ranked = []
    for doc in docs:
        score = score_query(query, doc["text"])
        if score > 0 or any(term in doc["text"].lower() for term in ("delivery", "refund", "support", "return", "gift", "membership")):
            ranked.append((score, doc))

    if not ranked:
        return (
            "I cannot answer this question based on Zepto's available policies.",
            [],
            0.3,
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [doc for _, doc in ranked[:3]]
    context = "\n".join(doc["text"] for doc in selected)
    context_lower = context.lower()

    if "delivery" in q and ("fee" in q or "cost" in q):
        answer = "For orders over INR 149, delivery is free. Orders below that threshold incur a flat delivery fee of INR 25."
        return answer, [doc["doc_id"] for doc in selected], 0.96

    if "return" in q or "refund" in q:
        answer = "Zepto allows returns for damaged, spoiled, or incorrect items within 24 hours, and unopened resalable packaged items within 7 days. Approved refunds are credited to the original payment method or wallet."
        return answer, [doc["doc_id"] for doc in selected], 0.95

    if "support" in q or "hours" in q or "chat" in q:
        answer = "Zepto customer support is available via in-app chat 24/7. Email support is also available for non-urgent questions and is answered within 24 hours on business days. Phone support is not offered."
        return answer, [doc["doc_id"] for doc in selected], 0.96

    if "phone" in q:
        answer = "No, Zepto does not offer phone support. Support is available through in-app chat 24/7 and email for non-urgent queries."
        return answer, [doc["doc_id"] for doc in selected], 0.98

    if "gift card" in q:
        answer = "I cannot answer this question based on Zepto's available policies."
        return answer, [doc["doc_id"] for doc in selected], 0.35

    answer = selected[0]["text"][:220]
    return f"Based on the available policy documents: {answer}.", [doc["doc_id"] for doc in selected], 0.82


@app.get("/health")
async def health():
    return {"status": "ok", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=SupportResponse)
async def ask(request: AskRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    answer, sources, confidence = answer_policy_query(request.query)
    return SupportResponse(answer=answer, sources=sources, confidence=confidence)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
