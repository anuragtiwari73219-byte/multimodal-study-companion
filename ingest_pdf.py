"""
PDF / slides ingestion.
Extracts text page-by-page and produces chunks tagged with
source metadata (file name, page number) so every chunk can be cited later.
"""

import pymupdf as fitz  # PyMuPDF (new import name; fitz is deprecated)
import re
from typing import List, Dict


def extract_pages(pdf_path: str) -> List[Dict]:
    """Returns a list of {page_number, text} for each page in the PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text: str, max_words: int = 180, overlap_words: int = 30) -> List[str]:
    """
    Splits text into overlapping word-based chunks.
    Overlap keeps context from being cut mid-idea at chunk boundaries.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def ingest_pdf(pdf_path: str, source_name: str) -> List[Dict]:
    """
    Full pipeline for one PDF: extract pages -> chunk -> attach citation metadata.
    Returns a list of dicts ready to insert into ChromaDB:
      {text, metadata: {source_type, source_name, page_number}}
    """
    pages = extract_pages(pdf_path)
    records = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            records.append({
                "text": chunk,
                "metadata": {
                    "source_type": "pdf",
                    "source_name": source_name,
                    "page_number": page["page_number"],
                }
            })
    return records


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf.py <path_to_pdf>")
        sys.exit(1)
    path = sys.argv[1]
    recs = ingest_pdf(path, source_name=path.split("/")[-1])
    print(f"Extracted {len(recs)} chunks from {path}")
    if recs:
        print("--- Sample chunk ---")
        print(recs[0]["metadata"])
        print(recs[0]["text"][:300])
