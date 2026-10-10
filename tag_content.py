"""
Day 6: Topic/concept tagging.

Post-processing pass over already-ingested chunks: asks the LLM to assign
each chunk a topic, subtopic, and list of key concepts, then writes these
back into ChromaDB metadata. Run this AFTER run_ingestion.py.

This satisfies: "Organize content into topics, concepts, and prerequisites,
with every unit linked to its origin" and "Identify major topics, subtopics,
and key concepts, and tag every content unit to them."

NOTE: prerequisites (concept A requires concept B) are NOT built — that
needs cross-chunk relationship inference, which is a bigger scope item.
Documented as a known gap, not silently skipped.
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
from store import get_all_chunks, update_metadata

load_dotenv()
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set.")
        _client = Groq(api_key=api_key)
    return _client


def tag_chunk(text: str) -> dict:
    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": (
                "Given this piece of course material, identify its main topic, "
                "a more specific subtopic, and 2-4 key concepts it covers. "
                'Respond ONLY with JSON: {"topic": "...", "subtopic": "...", "concepts": ["...", "..."]}'
            )},
            {"role": "user", "content": text[:1500]},  # cap to keep calls fast/cheap
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip("`").replace("json\n", "", 1).strip()
        return json.loads(cleaned)


def tag_all_chunks():
    chunks = get_all_chunks()
    print(f"Tagging {len(chunks)} chunks...")
    failed = []

    for i, chunk in enumerate(chunks):
        # Skip chunks already tagged (so re-running this script is cheap/idempotent)
        if chunk["metadata"].get("topic"):
            continue
        try:
            tags = tag_chunk(chunk["text"])
            # ChromaDB metadata values must be str/int/float/bool — concepts list
            # gets joined into a comma-separated string, not stored as a list.
            update_metadata(chunk["id"], {
                "topic": tags.get("topic", ""),
                "subtopic": tags.get("subtopic", ""),
                "concepts": ", ".join(tags.get("concepts", [])),
            })
            print(f"  [{i+1}/{len(chunks)}] {chunk['metadata'].get('source_name')} -> {tags.get('topic')} / {tags.get('subtopic')}")
        except Exception as e:
            failed.append((chunk["id"], str(e)))
            print(f"  [{i+1}/{len(chunks)}] FAILED: {e}")

    print(f"\nDone. Tagged {len(chunks) - len(failed)}/{len(chunks)}. Failed: {len(failed)}")
    if failed:
        print("Failed chunk IDs:", [f[0] for f in failed])


if __name__ == "__main__":
    tag_all_chunks()
