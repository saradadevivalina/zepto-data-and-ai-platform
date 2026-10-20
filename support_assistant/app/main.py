# main.py
# The API layer: wraps the LangGraph RAG pipeline in a FastAPI web service.
#
#   POST /ask   body: {"query": "..."}  ->  {"answer": ..., "sources": [...], "confidence": ...}
#   GET  /      quick health check
#
# Run locally (from inside support_assistant/):
#   uvicorn app.main:app --host 0.0.0.0 --port 7860

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import MOCK_LLM
from app.graph import build_graph, _get_collection
from app.schemas import AskRequest, SupportResponse

# The compiled LangGraph graph, built once when the server starts
rag_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before the first request):
    loads the embedding model + ChromaDB collection and compiles the graph,
    so individual requests stay fast.
    """
    global rag_graph
    _get_collection()          # embeds the 8 docs on first run, reuses them afterwards
    rag_graph = build_graph()
    yield                      # the app serves requests while paused here


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based policy assistant (LangGraph + ChromaDB + FastAPI)",
    lifespan=lifespan,
)


@app.get("/")
def health():
    """Health check: confirms the service is up and which LLM mode it is in."""
    return {"status": "ok", "mock_llm": MOCK_LLM}


@app.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest) -> SupportResponse:
    """
    Answer a customer question.
    FastAPI validates the incoming JSON against AskRequest, and the
    returned object against SupportResponse (answer / sources / confidence).
    """
    result = rag_graph.invoke({"query": request.query})
    return SupportResponse(**result["final_response"])