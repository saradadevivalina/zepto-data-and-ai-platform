import os

MOCK_LLM: bool = os.getenv("MOCK_LLM", "1").lower() in ("1", "true", "yes")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCS_DIR: str = os.getenv("DOCS_DIR", "./docs")