# corpus_loader.py
# Loads the Zepto policy documents from disk and ready them for embedding and retrieval.
# Returns a list of text chunks, each with a unique ID, text content, and source document name.

from app.config import DOCS_DIR

def load_documents():
    """
    Load the Zepto policy documents from disk.
    Each document is short (one paragraph), so each whole file 
    is treated as a single chunk.

    Returns:
        List[dict]: A list of dictionaries, each representing a text chunk with keys:
            - 'id': Unique identifier for the chunk (e.g., "doc_01").
            - 'text': The text content of the chunk.
            - 'source': The name of the source document (e.g., "doc_01.txt").
    """
    documents = []
    for doc_file in sorted(DOCS_DIR.glob("doc_*.txt")):
        with open(doc_file, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({
                "id": doc_file.stem,  # e.g., "doc_01"
                "text": text,
                "source": doc_file.name  # e.g., "doc_01.txt"
            })
    return documents

if __name__ == "__main__":
    # Example usage: Load documents and print their IDs and sources
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}")
    for doc in docs:
        print(f"Loaded document ID: {doc['id']}, Source: {doc['source']}, Text: {doc['text'][:60]}...")