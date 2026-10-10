"""
Lecture video / audio ingestion.
Uses Groq's hosted Whisper (whisper-large-v3) for transcription with
word/segment-level timestamps, then chunks the transcript while keeping
a timestamp on every chunk so answers can cite "video, 12:40".

Why Groq's API instead of local openai-whisper:
- No multi-GB torch install needed
- Faster (runs on Groq's hardware)
- Already using Groq elsewhere in this project -> one API key, one dependency
"""

import os
import subprocess
import tempfile
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_client = None


def _get_client() -> Groq:
    """Lazy client init so importing this module doesn't crash when no
    video/audio is being processed yet (e.g. PDF-only ingestion runs)."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to a .env file (see .env.example) "
                "before ingesting video/audio."
            )
        _client = Groq(api_key=api_key)
    return _client


def extract_audio(video_path: str) -> str:
    """
    Extracts a mono 16kHz WAV from a video file using ffmpeg.
    Whisper doesn't need the video track, and a smaller audio file
    uploads/transcribes faster.
    """
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-vn",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
    return out_path


def transcribe(audio_path: str) -> List[Dict]:
    """
    Sends audio to Groq Whisper and returns a list of
    {start, end, text} segments with timestamps in seconds.
    """
    client = _get_client()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    segments = getattr(response, "segments", None) or response.get("segments", [])
    return [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in segments if s["text"].strip()
    ]


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def group_segments(segments: List[Dict], max_words: int = 150) -> List[Dict]:
    """
    Groups small Whisper segments into larger chunks (~max_words) for
    retrieval, while keeping the start timestamp of the FIRST segment
    in each group as the citation anchor.
    """
    groups = []
    current_words = []
    current_start = None

    for seg in segments:
        if current_start is None:
            current_start = seg["start"]
        current_words.extend(seg["text"].split())
        if len(current_words) >= max_words:
            groups.append({"start": current_start, "text": " ".join(current_words)})
            current_words = []
            current_start = None

    if current_words:
        groups.append({"start": current_start, "text": " ".join(current_words)})

    return groups


def ingest_video(video_path: str, source_name: str) -> List[Dict]:
    """
    Full pipeline for one video: extract audio -> transcribe -> group into
    chunks -> attach citation metadata (timestamp).
    Returns records ready for ChromaDB insertion.
    """
    audio_path = extract_audio(video_path)
    try:
        segments = transcribe(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    groups = group_segments(segments)
    records = []
    for g in groups:
        records.append({
            "text": g["text"],
            "metadata": {
                "source_type": "video",
                "source_name": source_name,
                "timestamp": format_timestamp(g["start"]),
                "timestamp_seconds": g["start"],
            }
        })
    return records


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest_audio.py <path_to_video_or_audio>")
        sys.exit(1)
    path = sys.argv[1]
    recs = ingest_video(path, source_name=path.split("/")[-1])
    print(f"Extracted {len(recs)} chunks from {path}")
    if recs:
        print("--- Sample chunk ---")
        print(recs[0]["metadata"])
        print(recs[0]["text"][:300])
