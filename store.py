"""
ChromaDB storage layer.
Uses ChromaDB's built-in default embedding function (ONNX MiniLM) —
deliberately NOT sentence-transformers/torch, to keep the install light.
"""

import chromadb
import hashlib
from typing import List, Dict

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "course_knowledge"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def _make_id(text: str, metadata: Dict) -> str:
    """Deterministic ID so re-running ingestion on the same file doesn't duplicate chunks."""
    key = f"{metadata.get('source_name')}|{metadata.get('page_number', '')}|{metadata.get('timestamp', '')}|{text[:50]}"
    return hashlib.sha256(key.encode()).hexdigest()


def add_records(records: List[Dict]):
    """records: list of {text, metadata} as produced by ingest_pdf / ingest_audio."""
    if not records:
        print("No records to add.")
        return
    ids = [_make_id(r["text"], r["metadata"]) for r in records]
    documents = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Stored {len(records)} chunks in ChromaDB (collection: {COLLECTION_NAME})")


def query(question: str, n_results: int = 4) -> List[Dict]:
    """Returns top-n matching chunks with their citation metadata."""
    results = collection.query(query_texts=[question], n_results=n_results)
    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        out.append({"text": doc, "metadata": meta, "distance": dist})
    return out


def get_all_chunks() -> List[Dict]:
    """Returns every stored chunk with its id, text, and metadata (for tagging passes)."""
    result = collection.get()
    return [
        {"id": i, "text": d, "metadata": m}
        for i, d, m in zip(result["ids"], result["documents"], result["metadatas"])
    ]


def update_metadata(chunk_id: str, metadata_updates: Dict):
    """Merges new fields into an existing chunk's metadata (e.g. adding topic/concepts tags)."""
    existing = collection.get(ids=[chunk_id])
    if not existing["ids"]:
        return
    merged = {**existing["metadatas"][0], **metadata_updates}
    collection.update(ids=[chunk_id], metadatas=[merged])


def count() -> int:
    return collection.count()


if __name__ == "__main__":
    print(f"Chunks currently stored: {count()}")
