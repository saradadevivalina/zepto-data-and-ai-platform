# corpus_loader.py
# Stage 1 & 2 of the RAG pipeline: INGESTION and EMBEDDING.
#   - load_documents(): reads the 8 Zepto policy files from docs/
#   - get_collection(): embeds them with all-MiniLM-L6-v2 and stores
#     the vectors in a persistent ChromaDB collection for retrieval.

import chromadb
from chromadb.utils import embedding_functions

from app.config import DOCS_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def load_documents():
    """
    Load the Zepto policy documents from disk.

    Each document is short (one paragraph), so each whole file
    is treated as a single chunk.

    Returns:
        list[dict]: one dictionary per document, with keys:
            - 'id':     unique chunk ID, e.g. "doc_01"
            - 'text':   the full text of the document
            - 'source': the file name, e.g. "doc_01.txt"
    """
    documents = []

    # sorted() keeps the order predictable: doc_01, doc_02, ... doc_08
    for doc_file in sorted(DOCS_DIR.glob("doc_*.txt")):
        # 'with' closes the file automatically when the block ends
        with open(doc_file, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({
                "id": doc_file.stem,       # file name without .txt -> "doc_01"
                "text": text,
                "source": doc_file.name,   # full file name -> "doc_01.txt"
            })

    return documents


def get_collection():
    """
    Get the ChromaDB collection holding the embedded policy documents.

    On the first run the collection is empty, so the documents are
    loaded, embedded and stored. On later runs the saved collection
    is reused from disk, so nothing is embedded twice.

    Returns:
        chromadb.Collection: the collection, ready to be queried.
    """
    # PersistentClient saves the database to disk (in chroma_db/),
    # so embeddings survive between runs. Chroma expects a plain
    # string path, so the Path object is wrapped in str().
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # The real all-MiniLM-L6-v2 model turns text into 384-number vectors
    # that capture meaning. Passing it to the collection means Chroma
    # embeds both the stored documents AND incoming queries with the
    # same model automatically, so they are always comparable.
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Reuse the collection if it exists, otherwise create it.
    # "hnsw:space": "cosine" -> similarity is measured by cosine distance.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    # Only add documents when the collection is empty, to avoid
    # re-embedding (and duplicate IDs) on every run.
    if collection.count() == 0:
        docs = load_documents()
        # Chroma embeds the 'documents' text using the embedding function above
        collection.add(
            documents=[doc["text"] for doc in docs],
            ids=[doc["id"] for doc in docs],
            metadatas=[{"source": doc["source"]} for doc in docs],
        )

    # Outside the 'if': return the collection whether it was just
    # built or loaded from disk.
    return collection


# Runs only with: python -m app.corpus_loader
# (not when another file imports this module)
if __name__ == "__main__":
    # Check 1: all 8 documents load from disk
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCS_DIR}")
    for doc in docs:
        print(f"Loaded document ID: {doc['id']}, Source: {doc['source']}, Text: {doc['text'][:60]}...")

    # Check 2: all 8 documents are stored in ChromaDB
    collection = get_collection()
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} documents.")

    # Check 3: retrieval is truly semantic. A gift card question
    # should return doc_07 (Gift Cards) as the top result.
    results = collection.query(
        query_texts=["Can I combine two gift cards?"],
        n_results=3,
    )
    # query() returns lists of lists (one inner list per question),
    # so [0] picks the results for our single question.
    print("Retrieved IDs:", results["ids"][0])