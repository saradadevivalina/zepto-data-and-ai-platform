from fastapi import FastAPI, HTTPException

from app.corpus_loader import initialize_vector_store
from app.graph import build_support_graph
from app.schemas import AskRequest, SupportResponse

app = FastAPI(
    title="Zepto Support Assistant API",
    description="GenAI Support Assistant with LangGraph orchestration and ChromaDB context retrieval.",
    version="1.0.0",
)

collection = initialize_vector_store()
graph = build_support_graph(collection)


@app.post("/ask", response_model=SupportResponse)
async def ask_endpoint(request: AskRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    initial_state = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "final_response": None,
    }

    final_output = graph.invoke(initial_state)
    return final_output["final_response"]
