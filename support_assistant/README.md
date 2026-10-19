# `/support_assistant` — Zepto Data & AI Platform

A GenAI customer-support assistant that answers questions about Zepto's delivery, returns, membership, tracking, cancellation, damaged-item, gift-card and support-hours policies. Answers are grounded in Zepto's own policy documents through a **LangGraph-orchestrated RAG pipeline** (local `all-MiniLM-L6-v2` embeddings + ChromaDB), returned through a **FastAPI** `POST /ask` endpoint with a **Pydantic-validated** JSON response.

The graded baseline runs **fully offline and deterministically** (`MOCK_LLM` unset or `1`): no API key, no signup and no network call to any LLM provider.

---

## Folder structure

```
support_assistant/
├── docs/                 # 8 Zepto policy documents (doc_01.txt ... doc_08.txt)
├── app/
│   ├── __init__.py       # makes app/ a Python package (needed for `uvicorn app.main:app`)
│   ├── config.py         # paths, model name, collection name, TOP_K, MOCK_LLM toggle
│   ├── corpus_loader.py  # ingestion + embedding into ChromaDB
│   ├── schemas.py        # Pydantic request/response models
│   ├── prompts.py        # structured RAG prompt template (used when MOCK_LLM=0)
│   ├── graph.py          # LangGraph StateGraph: routing, retrieval, generation
│   └── main.py           # FastAPI app exposing POST /ask
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

The first run downloads the open-source `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face and builds the ChromaDB collection in `chroma_db/`. Later runs reuse both.

## How to run

All commands are run from inside `support_assistant/`.

**Locally with uvicorn:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```
Then open `http://127.0.0.1:7860/docs` for the interactive API page, or call `POST /ask` directly (examples below).

**With Docker:**
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

**Quick module checks:**
```bash
python -m app.corpus_loader   # loads 8 docs, builds/loads ChromaDB, runs a test query
python -m app.schemas         # shows a valid response and a rejected one (confidence=1.5)
python -m app.prompts         # prints the filled-in structured prompt
python -m app.graph           # runs 3 example queries through the graph
```

### The `MOCK_LLM` toggle

| `MOCK_LLM` | Behaviour |
|---|---|
| unset or `1` (**default, graded**) | Fully offline. Intent is classified with a keyword heuristic; answers are deterministic templates. No LLM call is made. |
| `0` (optional extension) | Calls a real LLM through Groq's free tier. Requires a `GROQ_API_KEY` environment variable (never committed to the repo). |

---

## Example calls (recorded with `MOCK_LLM` at its default)

Server started with `uvicorn app.main:app --host 0.0.0.0 --port 7860`; requests sent from Windows PowerShell.

### Example 1 — policy question → routed to `retrieve_and_answer`

Request:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:7860/ask -Method Post -ContentType "application/json" -Body '{"query": "Can I combine two gift cards in one order?"}' | ConvertTo-Json
```
Raw JSON response:
```json
{
    "answer":  "Based on the retrieved context: Gift Cards: \"Zepto gift cards are available in fixed denominations of INR 100, INR 250, INR 500, and INR 1000, and are delivered by email or SMS within minutes of purchase. Gift cards are valid for 1 ",
    "sources":  [
                    "doc_07",
                    "doc_01",
                    "doc_02"
                ],
    "confidence":  1.0
}
```
The query contains the keyword `gift card`, so `classify_intent` labels it `policy_question`. Retrieval returns `doc_07` (Gift Cards) as the most similar chunk, and the mock answer is built from its first 200 characters.

### Example 2 — general question → routed to `direct_answer`

Request:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:7860/ask -Method Post -ContentType "application/json" -Body '{"query": "What is the capital of France?"}' | ConvertTo-Json
```
Raw JSON response:
```json
{
    "answer":  "I can only answer questions about Zepto policies right now.",
    "sources":  [

                ],
    "confidence":  1.0
}
```
No policy keyword is present, so the query is routed to `direct_answer`, which returns the fixed canned string with an empty `sources` list. No retrieval and no LLM call happen.

---

## Architecture: the RAG pipeline

```
                 ┌──────────────────────────── ingestion + embedding (startup) ───────────────────────────┐
 docs/*.txt ──► corpus_loader.load_documents() ──► all-MiniLM-L6-v2 ──► ChromaDB collection "zepto_policies"
                                                                                     │
 POST /ask {"query"} ──► main.ask() ──► LangGraph StateGraph                         │
                                          │                                          │
                                   classify_intent                                   │
                               ┌──────────┴───────────┐                              │
                    policy_question            general_question                      │
                               │                      │                              │
                     retrieve_and_answer ◄── top-3 cosine search ────────────────────┘
                               │                      │
                               │               direct_answer
                               └──────────┬───────────┘
                                          ▼
                       SupportResponse {answer, sources, confidence}  (Pydantic-validated JSON)
```

**1. Ingestion** — `app/corpus_loader.py`, `load_documents()` reads the 8 files in `docs/` in sorted order. Each document is a single short paragraph, so each whole file is treated as **one chunk**, with its file stem as the chunk ID (e.g. `doc_07`) and its filename stored as `source` metadata.

**2. Embedding** — `app/corpus_loader.py`, `get_collection()` creates a `chromadb.PersistentClient` saving to `chroma_db/` and a collection named **`zepto_policies`** configured for cosine similarity (`hnsw:space: cosine`). The collection uses ChromaDB's `SentenceTransformerEmbeddingFunction` with the local **`all-MiniLM-L6-v2`** model, so stored documents and incoming queries are always embedded by the same model. Documents are only added when the collection is empty, so the corpus is embedded once and reused on every restart. The FastAPI `lifespan` hook in `app/main.py` loads the collection and compiles the graph once at startup.

**3. Retrieval** — `app/graph.py`, node **`retrieve_and_answer`**: the query is embedded and the **top-3** most similar chunks (`TOP_K` in `config.py`) are returned from `zepto_policies` by cosine similarity. This step always runs for real, in both `MOCK_LLM` modes, since embedding and ChromaDB need no API key or network.

**4. Generation** — `app/graph.py`:
- **Mock mode (default):** `retrieve_and_answer` returns `f"Based on the retrieved context: {top_chunk_snippet}"`, where the snippet is the first 200 characters of the most similar chunk; `sources` is the list of retrieved chunk IDs; `confidence` is fixed at `1.0`. `direct_answer` returns the fixed string *"I can only answer questions about Zepto policies right now."* with empty `sources`.
- **Real-LLM mode (`MOCK_LLM=0`):** `retrieve_and_answer` fills the structured prompt from `app/prompts.py` (`STRUCTURED_RAG_PROMPT_TEMPLATE`, via `build_prompt()`) with the retrieved chunks and sends it to Groq; `direct_answer` sends the question to the LLM directly, without retrieval.

**Orchestration** — `app/graph.py`, `build_graph()`: a LangGraph `StateGraph` over a `TypedDict` state (`query`, `intent`, `retrieved_chunks`, `final_response`) with three nodes: `classify_intent`, `retrieve_and_answer` and `direct_answer`. A **conditional edge** from `classify_intent` (`route_by_intent`) sends `policy_question` to `retrieve_and_answer` and `general_question` to `direct_answer`. In mock mode, `classify_intent` uses the keyword heuristic (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`); with `MOCK_LLM=0` the LLM classifies instead. The routing logic itself does not depend on `MOCK_LLM` — only the generation step inside each node does.

**Structured output** — `app/schemas.py`: `AskRequest` (`query: str`) validates the request; `SupportResponse` enforces `answer: str`, `sources: list[str]` and `confidence: float` between 0 and 1. In mock mode the response is populated deterministically by code. With `MOCK_LLM=0`, `generate_validated_response()` in `app/graph.py` validates the LLM's raw JSON against `SupportResponse` and, on failure, **retries up to 2 more times with a corrective instruction** before returning a clearly marked error response.

### Which stages branch on `MOCK_LLM`

Only **generation** (and the optional LLM-based intent classification). Ingestion, embedding, retrieval and the graph's routing are identical in both modes.

---

## Structured prompt template

Defined as text in `app/prompts.py` (`STRUCTURED_RAG_PROMPT_TEMPLATE`), used only by the optional `MOCK_LLM=0` path. It contains all five skeleton components — **[ROLE]**, **[CONTEXT]**, **[TASK]**, **[FORMAT]**, **[LENGTH]** — plus explicit **negative constraints** (e.g. *"Do not answer using information not present in the provided context"*) and **two few-shot examples** with ideal JSON answers.