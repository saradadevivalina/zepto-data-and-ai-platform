# graph.py
# Stage 3 & 4 of the RAG pipeline: RETRIEVAL and GENERATION, orchestrated by LangGraph.
#
# Flow:   START -> classify_intent --(policy_question)--> retrieve_and_answer -> END
#                                  \-(general_question)-> direct_answer -------> END
#
# Only the GENERATION step inside each node branches on MOCK_LLM.
# Retrieval (embedding the query + ChromaDB search) always runs for real.

import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import ValidationError

from app.config import MOCK_LLM, TOP_K
from app.corpus_loader import get_collection
from app.prompts import build_prompt
from app.schemas import SupportResponse

# Keywords from the assignment: any of these in the query -> policy question
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

# Fixed reply for general questions in mock mode
GENERAL_ANSWER = "I can only answer questions about Zepto policies right now."

# Real-LLM settings (only used when MOCK_LLM=0)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_RETRIES = 2  # extra attempts if the LLM's JSON fails validation


# ---------------------------------------------------------------------------
# State: the data passed from node to node
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    query: str                    # the user's question
    intent: str                   # "policy_question" or "general_question"
    retrieved_chunks: list[dict]  # [{"id": "doc_07", "text": "..."}, ...]
    final_response: dict          # a validated SupportResponse, as a dict


# The ChromaDB collection is loaded once and reused for every request.
_collection = None

def _get_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


# ---------------------------------------------------------------------------
# Real-LLM helpers (optional MOCK_LLM=0 extension; never called in mock mode)
# ---------------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    """Send a prompt to Groq and return the raw text reply."""
    from groq import Groq  # imported here so mock mode never needs the package
    client = Groq()        # reads GROQ_API_KEY from the environment
    reply = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,     # deterministic, factual answers
    )
    return reply.choices[0].message.content


def _extract_json(text: str) -> str:
    """LLMs sometimes wrap JSON in extra text or ```json fences; keep only the {...} part."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else text


def generate_validated_response(prompt: str) -> SupportResponse:
    """
    Ask the LLM for a JSON answer and validate it against SupportResponse.
    If validation fails, retry up to MAX_RETRIES more times with a corrective
    instruction; if it still fails, return a clearly marked error response.
    """
    current_prompt = prompt
    for attempt in range(1 + MAX_RETRIES):
        raw = call_llm(current_prompt)
        try:
            return SupportResponse.model_validate_json(_extract_json(raw))
        except ValidationError as error:
            current_prompt = (
                prompt
                + "\n\nYour previous reply was not valid JSON for the required schema."
                + f"\nValidation error: {error}"
                + "\nReply again with ONLY a valid JSON object with keys "
                + '"answer" (string), "sources" (list of strings) and "confidence" (number 0-1).'
            )
    return SupportResponse(
        answer="ERROR: the language model did not return a valid response after retries.",
        sources=[],
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> dict:
    query = state["query"].lower()

    if MOCK_LLM:
        # Mock mode (graded baseline): keyword heuristic, no LLM call
        is_policy = any(keyword in query for keyword in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
    else:
        # Optional real-LLM mode: let the LLM classify
        reply = call_llm(
            "Classify this customer message as exactly one of: policy_question, general_question.\n"
            "policy_question = about Zepto delivery, returns, refunds, membership, tracking, "
            "cancellation, damaged items, gift cards or support hours.\n"
            f"Message: {state['query']}\n"
            "Reply with only the label."
        ).strip().lower()
        intent = "policy_question" if "policy" in reply else "general_question"

    return {"intent": intent}


# Conditional edge: decides which node runs next (does NOT depend on MOCK_LLM)
def route_by_intent(state: GraphState) -> str:
    return state["intent"]


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy questions)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> dict:
    # Retrieval ALWAYS runs for real: embed the query, get the top-3 similar chunks
    results = _get_collection().query(query_texts=[state["query"]], n_results=TOP_K)
    chunks = [
        {"id": doc_id, "text": text}
        for doc_id, text in zip(results["ids"][0], results["documents"][0])
    ]

    if MOCK_LLM:
        # Mock mode: canned template built from the single most similar chunk
        top_chunk_snippet = chunks[0]["text"][:200]
        response = SupportResponse(
            answer=f"Based on the retrieved context: {top_chunk_snippet}",
            sources=[chunk["id"] for chunk in chunks],
            confidence=1.0,
        )
    else:
        # Optional real-LLM mode: grounded answer using the structured prompt template
        response = generate_validated_response(build_prompt(chunks, state["query"]))

    return {"retrieved_chunks": chunks, "final_response": response.model_dump()}


# ---------------------------------------------------------------------------
# Node 3: direct_answer (general questions, no retrieval)
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> dict:
    if MOCK_LLM:
        # Mock mode: fixed canned string, no LLM call
        response = SupportResponse(answer=GENERAL_ANSWER, sources=[], confidence=1.0)
    else:
        # Optional real-LLM mode: ask the LLM directly, no retrieval
        reply = call_llm(f"Answer briefly and politely: {state['query']}")
        response = SupportResponse(answer=reply.strip(), sources=[], confidence=0.5)

    return {"final_response": response.model_dump()}


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


if __name__ == "__main__":
    app_graph = build_graph()
    print(f"MOCK_LLM = {MOCK_LLM}\n")

    test_queries = [
        "Can I combine two gift card balances in one order?",  # policy -> retrieval
        "What is the refund timeline for a returned item?",    # policy -> retrieval
        "What is the capital of France?",                     # general -> direct
    ]
    for q in test_queries:
        result = app_graph.invoke({"query": q})
        print("Query:   ", q)
        print("Intent:  ", result["intent"])
        print("Response:", result["final_response"])
        print()