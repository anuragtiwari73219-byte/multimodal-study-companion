"""
Day 2: RAG Question Answering with citations.

Flow:
  question -> retrieve top-k chunks from ChromaDB -> build a prompt that
  includes each chunk's source -> ask Groq's LLM to answer USING ONLY
  those chunks -> return the answer plus the exact sources used.

Design choice: the citation shown to the user comes from OUR metadata
(page_number / timestamp), not from the LLM's own claim. The LLM is only
asked to say which chunk number(s) it used; we map that back to the real
metadata ourselves. This avoids the LLM hallucinating a page/timestamp
that doesn't exist.
"""

import os
import json
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
from store import query as retrieve_chunks

load_dotenv()
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to .env first.")
        _client = Groq(api_key=api_key)
    return _client


def _format_source(metadata: Dict) -> str:
    """Turns raw metadata into a human-readable citation string."""
    if metadata.get("source_type") == "pdf":
        return f"{metadata['source_name']}, page {metadata['page_number']}"
    elif metadata.get("source_type") == "video":
        return f"{metadata['source_name']}, {metadata['timestamp']}"
    elif metadata.get("source_type") == "image":
        pg = metadata.get("page_number", metadata.get("page", "?"))
        return f"{metadata['source_name']}, page {pg} (diagram)"
    return metadata.get("source_name", "unknown source")


def _source_link(metadata: Dict) -> Dict:
    """Label plus file and URL fragment so the UI can open the exact page or timestamp."""
    st = metadata.get("source_type")
    frag = ""
    if st in ("pdf", "image"):
        pg = metadata.get("page_number", metadata.get("page"))
        if pg is not None:
            frag = f"page={pg}"
    elif st == "video":
        frag = f"t={int(metadata.get('timestamp_seconds', 0) or 0)}"
    return {"label": _format_source(metadata), "file": metadata.get("source_name", ""), "fragment": frag}


DISTANCE_THRESHOLD = 1.6  # ChromaDB L2 distance; above this, treat retrieval as "not covered"
# Calibrated empirically on this project's own ingested content (Newton's
# laws video + Python notes PDF): in-scope best-match distances were
# 1.423-1.457, out-of-scope (unrelated questions) were 1.711-1.898.
# If you ingest very different content, re-run this same check — the gap
# may shift. See eval.py's refusal test cases for the calibration method.


def answer_question(question: str, n_results: int = 4) -> Dict:
    """
    Returns:
      {
        "answer": str,
        "sources": [str, ...],       # human-readable citations actually used
        "used_chunk_indices": [int], # which retrieved chunks the LLM says it used
        "no_answer_found": bool      # True if the KB has nothing relevant
      }
    """
    chunks = retrieve_chunks(question, n_results=n_results)

    if not chunks:
        return {
            "answer": "I don't have any material on this yet. Try ingesting more content first.",
            "sources": [],
            "used_chunk_indices": [],
            "no_answer_found": True,
        }

    # Out-of-scope refusal: if even the best-matching chunk is too dissimilar,
    # the question is likely not covered by the ingested material at all.
    # This is a distance-based heuristic, not semantic understanding — it
    # catches clearly unrelated questions, not subtly-out-of-scope ones.
    best_distance = min(c["distance"] for c in chunks)
    if best_distance > DISTANCE_THRESHOLD:
        return {
            "answer": "This doesn't appear to be covered in the ingested course material.",
            "sources": [],
            "used_chunk_indices": [],
            "no_answer_found": True,
        }

    # Build numbered context blocks so the LLM can reference them by number
    context_blocks = []
    for i, c in enumerate(chunks):
        source_label = _format_source(c["metadata"])
        context_blocks.append(f"[Chunk {i}] (Source: {source_label})\n{c['text']}")
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a study assistant. Answer the student's question using ONLY "
        "the numbered chunks provided below. Do not use outside knowledge. "
        "If the chunks don't contain enough information to answer, say so clearly "
        "instead of guessing.\n\n"
        "Respond ONLY with valid JSON in this exact format, nothing else:\n"
        '{"answer": "<your answer text>", "used_chunks": [<chunk numbers you actually used>], "sufficient": <true or false>}'
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  # deterministic: same question -> same answer, no random tool-choice-style flakiness
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if the model wraps JSON in markdown fences despite instructions
        cleaned = raw.strip("`").replace("json\n", "", 1).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "answer": raw,
                "sources": [_format_source(c["metadata"]) for c in chunks],
                "used_chunk_indices": [],
                "no_answer_found": False,
                "warning": "Model did not return valid JSON; showing raw output and all retrieved sources.",
            }

    used_indices = [i for i in parsed.get("used_chunks", []) if 0 <= i < len(chunks)]
    sources = [_format_source(chunks[i]["metadata"]) for i in used_indices]
    images = [
        chunks[i]["metadata"]["image_path"]
        for i in used_indices
        if chunks[i]["metadata"].get("source_type") == "image"
        and chunks[i]["metadata"].get("image_path")
    ]

    return {
        "answer": parsed.get("answer", ""),
        "sources": sources,
        "images": images,
        "links": [_source_link(chunks[i]["metadata"]) for i in used_indices],
        "used_chunk_indices": used_indices,
        "no_answer_found": not parsed.get("sufficient", True),
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is Newton's second law?"
    result = answer_question(q)
    print(f"\nQ: {q}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s}")
    else:
        print("(no sources cited)")
    if result.get("warning"):
        print(f"\nWARNING: {result['warning']}")
