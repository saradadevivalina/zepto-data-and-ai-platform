# config.py
# Central settings for the Zepto support assistant.
# Every other file imports from here, so paths and names live in one place.

import os
from pathlib import Path

# --- Paths ---
# __file__ is this file (app/config.py).
# .parent -> app/, .parent.parent -> support_assistant/
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder holding the 8 Zepto policy documents (doc_01.txt ... doc_08.txt)
DOCS_DIR = BASE_DIR / "docs"

# Folder where ChromaDB saves the embeddings on disk (ignored by Git)
CHROMA_DIR = BASE_DIR / "chroma_db"

# --- Vector store settings ---
# Name of the ChromaDB collection that stores the policy chunks
COLLECTION_NAME = "zepto_policies"

# Local sentence-transformers model used to embed documents and queries
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of most-similar chunks to retrieve for each question
TOP_K = 3

# --- LLM mode toggle ---
# Environment variables are always text, so we compare strings.
# Unset -> default "1" -> mock mode ON (the graded, offline baseline).
# Only MOCK_LLM=0 switches to the optional real-LLM (Groq) path.
MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"