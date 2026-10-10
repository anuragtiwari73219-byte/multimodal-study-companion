# AI Study Companion (multimodal tutor agent)

A study assistant built for the Multimodal AI Hackathon 2026. It ingests course material in three forms (PDF text, lecture video, and diagrams extracted from PDFs), answers questions with citations, refuses out-of-scope questions, and runs an adaptive quiz with a per-topic mastery score.

Everything here was built and tested on a small corpus: one PDF (Python notes, 86 text chunks plus 4 diagram chunks) and one lecture video (Newton's laws, 4 chunks). Results below should be read with that size in mind.

## What it does

- **Ingestion (PDF + video):** PDF text is chunked with page numbers. Video is converted to audio with ffmpeg and transcribed with Whisper (`whisper-large-v3` on Groq), keeping a timestamp on every chunk. Everything is stored in ChromaDB with source metadata.
- **Topic tagging:** each chunk gets a topic, subtopic and list of key concepts from an LLM, written back into the chunk metadata. The script can be re-run safely (already-tagged chunks are skipped).
- **Diagram extraction:** PyMuPDF pulls embedded raster images out of the PDF. A vision model (`qwen/qwen3.8-27b` on Groq) writes a short description of each, and that description is indexed as an `image` chunk so diagrams can be retrieved and shown next to an answer.
- **Question answering with citations:** retrieval from ChromaDB, then an LLM (`openai/gpt-oss-120b` on Groq) answers using only the retrieved chunks. The citation shown to the user (for example `file.pdf, page 5` or `video.mp4, 00:58`) is built from our stored metadata, not from the LLM's claim. The LLM only reports which chunk numbers it used.
- **Out-of-scope refusal:** two gates. First, a distance threshold on the best retrieval match. Second, the LLM must report that the chunks were sufficient. If either fails, the app says the material doesn't cover the question.
- **Adaptive quiz:** the LLM only writes question and evaluation content. Difficulty (easy, medium, hard) is chosen by a deterministic rule from recent correctness. Mastery per topic is a simple transparent score (+10 for correct, -7 for incorrect, clamped to 0-100), not a real Bayesian Knowledge Tracing or IRT model. Correct answers stay server-side until graded. A report is generated at the end.
- **Simulated learners and evaluation scripts:** `simulate.py` runs weak, average and strong simulated students through the adaptive logic. `eval.py` scores the Q&A system with our own LLM-as-judge implementation of faithfulness, answer relevancy, context precision and context recall, plus a refusal test.

## Prerequisites

- Python 3.11 (tested on 3.11.9, Windows)
- ffmpeg available on your PATH (used to extract audio from video)
- A Groq API key
- Your own course material: one or more PDFs and/or lecture videos. The repository does not include the PDF or video we tested with (they are third-party content), so you need to supply your own.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# open .env and set GROQ_API_KEY=<your key>
```

(macOS/Linux: `source venv/bin/activate` and `cp .env.example .env`.)

## Run

1. Put your PDFs and/or videos in a `data/` folder.

2. Ingest, then tag:

   ```powershell
   python run_ingestion.py ./data
   python tag_content.py
   ```

3. Optional: diagram pipeline (needs a PDF that contains embedded images).

   ```powershell
   python extract_images.py "data/<your file>.pdf"
   python describe_images.py
   python ingest_images.py
   ```

   `ingest_images.py` has a `SKIP` set near the top. In our run it lists two university-seal images that were not diagrams. Review `extracted_images/` for your own PDF and edit that set before ingesting.

4. Start the app:

   ```powershell
   python app.py
   ```

   Open http://127.0.0.1:5000. Use "Ask a Question" for cited answers (a diagram appears under the answer when one was used), or "Adaptive Quiz" with a topic from your material.

5. Optional evaluation:

   ```powershell
   python eval.py
   python simulate.py
   ```

   Results are written to `eval_results.json` and `simulation_results.json`. The test questions in `eval.py` are about the material we ingested, so edit them for your own content.

## Results we measured

All numbers below come from the small runs described above.

**Simulated learners (8 questions per profile).** The strong profile reached mastery 79 and the hard difficulty level. The weak and average profiles both ended with the same mastery gain (-5.0) and both stayed on easy. With 8 questions per profile we cannot tell weak from average. Correctness in `simulate.py` is sampled from a hand-picked probability table, not graded from simulated answers, so this tests the adaptive logic, not realistic student behaviour. Question repetition was 0.0% in every profile, but that check is exact text match only.

**Q&A evaluation (2 in-scope questions, 2 out-of-scope questions).**

| Metric | Score |
|---|---|
| Faithfulness | 1.0 |
| Answer relevancy | 1.0 |
| Context precision | 0.5 |
| Context recall | 1.0 |
| Refusal accuracy | 2/2 |

Two in-scope questions is far too few to treat 1.0 as meaningful. Context precision of 0.5 means only 2 of 4 retrieved chunks were judged relevant, so retrieving 4 chunks is probably more than needed.

## Known limitations

- **Tiny evaluation sets.** n=8 per simulated profile, and 2 + 2 questions in the Q&A evaluation.
- **LLM judge, same model family.** `eval.py` uses `openai/gpt-oss-120b` both to answer and to judge. Judges tend to favour their own outputs, and we did not check this. RAGAS could not be imported in our environment because of an upstream dependency break, so we implemented the same four metrics ourselves. These are not RAGAS scores.
- **The refusal distance threshold is not reliable on its own.** The threshold (1.6) was calibrated on the original two documents. After adding diagram chunks, some out-of-scope questions had best-match distances below it (for example "capital of India" 1.553, "photosynthesis" 1.519, "water cycle diagram" 1.262). In an end-to-end test of five questions the LLM sufficiency check refused those correctly, but we tested only five questions.
- **Repetition check is exact match.** Semantically duplicate quiz questions with different wording are not detected.
- **Diagram support is narrow.** It was tested on one PDF with 4 usable diagrams. PyMuPDF extracts embedded raster images only, so diagrams drawn as vector graphics are missed. Image descriptions are model-generated and we did not manually verify all of them. The vision model is a preview-tier model on Groq and may be rotated or removed.
- **Quiz grounding is not checked.** Generated questions come from retrieved chunks, but we have no automated check that every number or fact in a question appears in the source.
- **Prerequisites are not built.** Topics, subtopics and concepts are tagged. Linking concepts by prerequisite needs cross-chunk reasoning that we did not implement.
- **Mastery is a simple heuristic**, not BKT or IRT.
- **Single user, local only.** Quiz state lives in the Flask session. The app runs in debug mode on localhost and is not hardened for deployment.

## Repository layout

| File | Purpose |
|---|---|
| `run_ingestion.py` | Entry point: ingests a folder of PDFs and videos |
| `ingest_pdf.py`, `ingest_audio.py` | PDF text and video transcription ingestion |
| `store.py` | ChromaDB access |
| `tag_content.py` | Topic, subtopic and concept tagging |
| `extract_images.py`, `describe_images.py`, `ingest_images.py` | Diagram pipeline |
| `qa.py` | Cited question answering and refusal logic |
| `tutor.py` | Question generation, grading, difficulty rule, mastery, report |
| `app.py`, `templates/index.html` | Flask web app |
| `simulate.py`, `eval.py` | Simulated learners and Q&A evaluation |
| `eval_results.json`, `simulation_results.json` | Saved results |

## Models used

All through the Groq API: `openai/gpt-oss-120b` (answers, tagging, quiz generation, evaluation judge), `whisper-large-v3` (transcription), `qwen/qwen3.8-27b` (diagram descriptions).
