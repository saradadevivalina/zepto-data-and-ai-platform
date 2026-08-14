import os
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from chromadb.utils import embedding_functions
from app.config import CHROMA_PERSIST_DIR, DOCS_DIR
import chromadb.utils.embedding_functions as embedding_functions

class OfflineMiniLMEmbeddingFunction(EmbeddingFunction):
    """Offline mock embedding function producing 384-dim vectors matching all-MiniLM-L6-v2 shape."""
    def __call__(self, input: Documents) -> Embeddings:
        # MiniLM-L6-v2 outputs 384 dimensions
        return [[0.1] * 384 for _ in input]
    
def initialize_vector_store():

    client = chromadb.Client()
    embedding_fn = OfflineMiniLMEmbeddingFunction()
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
    name="support_docs",
    embedding_function=embedding_fn
)
    
    # Ingest documents if collection is empty
    if collection.count() == 0:
        documents = []
        metadatas = []
        ids = []
        
        for filename in sorted(os.listdir(DOCS_DIR)):
            if filename.startswith("doc_") and filename.endswith(".txt"):
                file_path = os.path.join(DOCS_DIR, filename)
                doc_id = os.path.splitext(filename)[0]
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                documents.append(content)
                metadatas.append({"source": filename, "doc_id": doc_id})
                ids.append(f"{doc_id}_chunk_0")
        
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Ingested {len(documents)} documents into ChromaDB.")
            
    return collection