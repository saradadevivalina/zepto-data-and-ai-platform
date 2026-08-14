import json
from typing import List, Dict, Any, TypedDict
from typing_extensions import Annotated
from pydantic import ValidationError

from langgraph.graph import StateGraph, END
from app.config import MOCK_LLM, GROQ_API_KEY
from app.schemas import SupportResponse
from app.prompts import STRUCTURED_RAG_PROMPT_TEMPLATE

# --- State Definition ---
class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    final_response: SupportResponse

# --- Keywords for Mock Classifier ---
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]

# --- Helper: Optional Real LLM Call (Groq) with Retry ---
def call_real_llm_with_retry(prompt: str, max_retries: int = 2) -> SupportResponse:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    
    current_prompt = prompt
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": current_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        raw_text = response.choices[0].message.content
        try:
            parsed_json = json.loads(raw_text)
            validated = SupportResponse(**parsed_json)
            return validated
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries:
                current_prompt = f"{prompt}\n\n[ERROR IN PREVIOUS OUTPUT]\nYour previous output failed validation: {str(e)}. Please correct your format and return strictly valid JSON matching the schema."
            else:
                return SupportResponse(
                    answer="Error generating structured response from LLM.",
                    sources=[],
                    confidence=0.0
                )

# --- Node 1: Classify Intent ---
def classify_intent_node(state: GraphState) -> Dict[str, Any]:
    query = state["query"]
    
    if MOCK_LLM:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        # Optional MOCK_LLM=0 Real LLM classification
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Classify the following user query as either 'policy_question' or 'general_question'. Return ONLY the classification term.\nQuery: {query}"
            }]
        )
        text = res.choices[0].message.content.strip().lower()
        intent = "policy_question" if "policy" in text else "general_question"
        
    return {"intent": intent}

# --- Node 2: Retrieve and Answer ---
def retrieve_and_answer_node(state: GraphState, collection) -> Dict[str, Any]:
    query = state["query"]
    
    # 1. Retrieval (Always runs for real against ChromaDB)
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved_chunks = []
    sources = []
    if results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        for doc, meta in zip(docs, metas):
            retrieved_chunks.append({"content": doc, "metadata": meta})
            sources.append(meta.get("doc_id"))
    
    # 2. Generation Branching
    if MOCK_LLM:
        top_snippet = retrieved_chunks[0]["content"][:200] if retrieved_chunks else "No relevant context found."
        canned_answer = f"Based on the retrieved context: {top_snippet}..."
        
        response = SupportResponse(
            answer=canned_answer,
            sources=list(set(sources)),
            confidence=1.0
        )
    else:
        # Optional MOCK_LLM=0 Path
        context_str = "\n".join([f"[{c['metadata'].get('doc_id')}]: {c['content']}" for c in retrieved_chunks])
        formatted_prompt = STRUCTURED_RAG_PROMPT_TEMPLATE.format(context=context_str, query=query)
        response = call_real_llm_with_retry(formatted_prompt)
        
    return {"retrieved_chunks": retrieved_chunks, "final_response": response}

# --- Node 3: Direct Answer ---
def direct_answer_node(state: GraphState) -> Dict[str, Any]:
    if MOCK_LLM:
        response = SupportResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
    else:
        # Optional MOCK_LLM=0 Path
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": state["query"]}]
        )
        response = SupportResponse(
            answer=res.choices[0].message.content,
            sources=[],
            confidence=0.8
        )
        
    return {"final_response": response}

# --- Conditional Edge Router ---
def route_intent(state: GraphState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

# --- Graph Assembly Function ---
def build_support_graph(collection):
    workflow = StateGraph(GraphState)
    
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("retrieve_and_answer", lambda state: retrieve_and_answer_node(state, collection))
    workflow.add_node("direct_answer", direct_answer_node)
    
    workflow.set_entry_point("classify_intent")
    
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )
    
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    
    return workflow.compile()