# `/support_assistant` — Zepto Data & AI Platform

A GenAI support assistant answering Zepto policy questions, grounded in Zepto's own documents via a LangGraph-orchestrated RAG pipeline, wrapped in a FastAPI service. The entire graded baseline runs fully offline and deterministically — no API key, no signup, no network call to any LLM provider is required.

## Setup

```bash
pip install -r requirements.txt
```

Required packages: `fastapi`, `uvicorn`, `pydantic`, `langgraph`, `chromadb`, `sentence-transformers`. `groq` is only needed for the optional `MOCK_LLM=0` extension.

## How to Run

**Locally:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```
The app initializes the ChromaDB vector store and compiles the LangGraph graph once at startup, then serves `POST /ask`.

**With Docker:**
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

**MOCK_LLM toggle:** left unset, or explicitly set to `MOCK_LLM=1`, the service runs entirely offline in deterministic mock mode — this is the default (also hardcoded as the Docker image's default via `ENV MOCK_LLM=1`) and is what gets graded. Setting `MOCK_LLM=0` switches to the optional, ungraded real-LLM extension via Groq's free tier, requiring a `GROQ_API_KEY` environment variable.

## Architecture

The pipeline runs in four stages — **ingestion → embedding → retrieval → generation** — each handled by a specific component:

**1. Ingestion** (`app/corpus_loader.py`): `initialize_vector_store()` reads all 8 policy documents from `docs/doc_01.txt` through `doc_08.txt`. Each document is treated as a single chunk (a reasonable simplification given their short, single-paragraph length), with its filename-derived `doc_id` (e.g., `"doc_01"`) and source filename stored as metadata alongside it.

**2. Embedding** (`app/corpus_loader.py`, via ChromaDB's built-in embedding function): each document chunk is embedded using the local, open-source `all-MiniLM-L6-v2` sentence-transformer model, wired directly into ChromaDB's `SentenceTransformerEmbeddingFunction`. This means both stored documents and future incoming queries are embedded with the exact same model automatically, with no manual step required to keep them in sync. Embeddings are persisted to disk via `chromadb.PersistentClient`, in a collection named `zepto_policies` configured for cosine similarity (`hnsw:space: cosine`). Ingestion only runs once — a `collection.count() == 0` check prevents re-embedding the corpus on every app restart.

**3. Retrieval** (`app/graph.py`, inside the `retrieve_and_answer` node): when a query is classified as a `policy_question`, the query is embedded and the top-3 most similar chunks are retrieved from the ChromaDB collection via cosine similarity. This retrieval step always runs for real, in both `MOCK_LLM` states, since embedding and ChromaDB querying require no API key or network access.

**4. Generation** (`app/graph.py`, inside `retrieve_and_answer` and `direct_answer`): this is the only stage that branches on `MOCK_LLM`.
- **Mock mode (default, graded baseline):** `retrieve_and_answer` returns a canned string built from the top retrieved chunk (`f"Based on the retrieved context: {top_snippet}..."`), and `direct_answer` returns a fixed string ("I can only answer questions about Zepto policies right now."). No LLM is called in either node; the `SupportResponse` schema (`answer`, `sources`, `confidence`) is populated deterministically by code.
- **Optional `MOCK_LLM=0` extension:** both nodes instead call Groq's LLM API — `retrieve_and_answer` using the structured prompt template from `app/prompts.py` (Role/Context/Task/Format/Length skeleton, negative constraints, and few-shot examples), grounded in the retrieved chunks; `direct_answer` calls the LLM directly with no retrieval. In this mode, the LLM's raw JSON output is validated against the `SupportResponse` schema, with up to 2 retries (each including a corrective instruction) if validation fails before returning a marked error response.

**Orchestration** (`app/graph.py`): a LangGraph `StateGraph` with a `TypedDict` state (`query`, `intent`, `retrieved_chunks`, `final_response`) and 3 nodes — `classify_intent`, `retrieve_and_answer`, `direct_answer`. `classify_intent` uses a keyword heuristic in mock mode (checking for "delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", or "support hours") to decide between `policy_question` and `general_question`, with no LLM call. A conditional edge (`route_intent`) then routes to the correct handler node based on that classification — this routing logic itself does not depend on `MOCK_LLM`, only the generation step inside each destination node does.

**API layer** (`app/main.py`): a FastAPI app exposing `POST /ask`, accepting `{"query": str}` (via the `AskRequest` Pydantic model) and returning the validated `SupportResponse` model. The vector store and compiled graph are initialized once at app startup, not per-request.

## Example Calls

Both run locally with `MOCK_LLM` left at its default.

**Example 1 — a policy-triggering query (routes to `retrieve_and_answer`):**
```
POST /ask
{"query": "..."}
```
Response:
```json
{
  "answer": "...",
  "sources": [...],
  "confidence": ...
}
```


**Example 2 — a general, non-policy query (routes to `direct_answer`):**
```
POST /ask
{"query": "..."}
```
Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```


## Structured Prompt Template

The full Role-Context-Task-Format-Length skeleton, including negative constraints and few-shot examples, is defined in `app/prompts.py` (`STRUCTURED_RAG_PROMPT_TEMPLATE`). This template is used only in the optional `MOCK_LLM=0` extension — it exists as required text regardless of which mode is active, per the task specification.


