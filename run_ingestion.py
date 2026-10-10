"""
Day-1 entry point.
Point this at a folder of course materials (PDFs, mp4/mp3 lecture files)
and it ingests everything into ChromaDB with citation metadata.

Usage:
    python run_ingestion.py ./data
"""

import sys
import os
from ingest_pdf import ingest_pdf
from ingest_audio import ingest_video
from store import add_records, count

PDF_EXTS = (".pdf",)
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".mp3", ".wav", ".m4a")


def ingest_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        print(f"ERROR: {folder_path} is not a folder.")
        sys.exit(1)

    files = sorted(os.listdir(folder_path))
    if not files:
        print(f"No files found in {folder_path}.")
        return

    total_before = count()

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        ext = os.path.splitext(fname)[1].lower()

        if ext in PDF_EXTS:
            print(f"\n[PDF] Ingesting {fname} ...")
            records = ingest_pdf(fpath, source_name=fname)
            add_records(records)

        elif ext in VIDEO_EXTS:
            print(f"\n[VIDEO/AUDIO] Ingesting {fname} ...")
            try:
                records = ingest_video(fpath, source_name=fname)
                add_records(records)
            except Exception as e:
                print(f"  FAILED on {fname}: {e}")

        else:
            print(f"\n[SKIP] {fname} (unsupported extension: {ext})")

    total_after = count()
    print(f"\nDone. Chunks before: {total_before}, after: {total_after}, added: {total_after - total_before}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./data"
    ingest_folder(folder)
