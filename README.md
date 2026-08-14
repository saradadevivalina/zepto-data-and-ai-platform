# Zepto Data & AI Platform

**Certificate Program in Artificial Intelligence and Machine Learning — Capstone Project**

An end-to-end AI/ML platform joining Zepto's analytics guild workflow: a data-engineering pipeline that turns raw scraped data into a clean relational store, an analytics pipeline that profiles and models a customer/passenger-style dataset end to end, and a GenAI support assistant that answers policy questions grounded in Zepto's own documents. All three modules live together in this single repository as one coherent submission.

**Total marks: 100** · `/data_pipeline` (25) · `/analytics` (50) · `/support_assistant` (25)

---

## Repository Structure

```
zepto-data-and-ai-platform/
├── data_pipeline/          # Module 1 — scrape, clean, convert, load to SQLite
│   ├── data_pipeline.py
│   ├── books_database.db
│   └── README.md
├── analytics/               # Module 2 — EDA + predictive modeling on Titanic
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── titanic.csv
│   ├── best_titanic_pipeline.joblib
│   └── README.md
├── support_assistant/       # Module 3 — RAG-grounded policy support agent
│   ├── docs/                # 8 Zepto policy documents
│   ├── app/                 # FastAPI + LangGraph application code
│   ├── Dockerfile
│   └── README.md
├── requirements.txt
└── README.md                 # you are here
```

Each module has its own detailed `README.md` covering setup, run instructions, and design decisions specific to that module. This root README gives the overall picture; see each module's own README for full depth.

---

## Setup

Each module has its own dependencies. Install per-module as you work through them:
```bash
pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt
pip install -r support_assistant/requirements.txt
```
(Or use the consolidated root `requirements.txt` covering all three, if provided as a single file — see comments at the top of that file for which approach this repo uses.)

---

## How to Run Each Module

**`/data_pipeline`**
```bash
cd data_pipeline
python data_pipeline.py
```
Scrapes ≥60 books across 4 categories from books.toscrape.com, cleans and converts pricing to INR at a fixed rate, loads into a normalized SQLite database, and runs 5 required SQL queries with a pandas-merge verification. See `data_pipeline/README.md` for the full design rationale.

**`/analytics`**
```bash
cd analytics
jupyter notebook 01_eda.ipynb   # then 02_modeling.ipynb
```
Run in order — `01_eda.ipynb` loads and profiles the Titanic dataset (via `sns.load_dataset`, cached after first run), cleans it, and produces the full exploratory data story, saving `titanic.csv` as a committed offline fallback. `02_modeling.ipynb` continues from that same cleaned data into a full classification + regression pipeline: three classifiers compared, imbalance handling compared, hyperparameters tuned, a regression side-task on fare, and a final saved, reloadable pipeline via `joblib`. See `analytics/README.md` for every task's results and written interpretation.

**`/support_assistant`**
```bash
cd support_assistant
uvicorn app.main:app --host 0.0.0.0 --port 7860
# or, via Docker:
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```
Serves a `POST /ask` endpoint answering Zepto policy questions via a LangGraph-orchestrated RAG pipeline (ChromaDB + local sentence-transformer embeddings), routing between grounded policy answers and a fallback response for unrelated queries. Runs fully offline by default (`MOCK_LLM=1`) — no API key required for full marks. See `support_assistant/README.md` for the full architecture walkthrough and example call transcripts.

---

## Design Decisions Summary

*(Full justification for each lives in the respective module README — this is a quick index.)*

- **`/data_pipeline`:** SQLite chosen for zero-config, file-based portability; drop-vs-median-impute strategy applied based on measured missing-value percentages per column; two-table normalized schema (categories/books) to avoid update anomalies; fixed currency rate (1 GBP = 105.50 INR) chosen for reproducibility over live-API fragility.
- **`/analytics`:** Threshold-based missing-value handling (drop <5%, impute 5-30%, drop/flag column above that) applied and justified per column with exact percentages; `ColumnTransformer`/`Pipeline` used throughout to structurally enforce fit-on-train-only preprocessing; three imbalance-handling strategies compared with a recall-prioritized recommendation; final model recommendation grounded in specific baseline and tuned metrics rather than a single headline number.
- **`/support_assistant`:** Fully offline, deterministic mock mode as the required graded baseline (no API key needed); ChromaDB's built-in embedding function used to keep document and query embeddings automatically in sync; LangGraph's conditional routing kept independent of the `MOCK_LLM` toggle, with only the generation step inside each node branching.

---

## Git Workflow

Per the project's submission requirements, this repository's commit history includes at least one feature branch, committed to at least twice, and merged back into `main` — checked once across the whole repository's history (visible via `git log --graph --all`), not required separately inside each module folder.

---

## Academic Integrity

The code, analysis, and written interpretations throughout this repository are my own work. Standard library and framework documentation was consulted during development, but all reasoning, implementation, and written conclusions are original.