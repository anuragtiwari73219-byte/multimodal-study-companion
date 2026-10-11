# AI Study Companion (multimodal tutor agent)

A study assistant built for the Multimodal AI Hackathon 2026. It ingests course material in three forms (PDF text, lecture video, and diagrams extracted from PDFs), answers questions with citations, refuses out-of-scope questions, and runs an adaptive quiz with a per-topic mastery score.

Everything here was built and tested on a small corpus: one PDF (Python notes, 86 text chunks plus 4 diagram chunks) and one lecture video (Newton's laws, 4 chunks). Results below should be read with that size in mind.

## What it does

- **Ingestion (PDF + video):** PDF text is chunked with page numbers. Video is converted to audio with ffmpeg and transcribed with Whisper (`whisper-large-v3` on Groq), keeping a timestamp on every chunk. Everything is stored in ChromaDB with source metadata. Ingestion runs from a folder with `run_ingestion.py`; the web app has no upload page.
- **Topic tagging:** each chunk gets a topic, subtopic and list of key concepts from an LLM, written back into the chunk metadata. The script is safe to re-run.
- **Diagram extraction:** PyMuPDF pulls embedded raster images out of the PDF. A vision model (`qwen/qwen3.8-27b` on Groq) describes each one, and the description is indexed as an `image` chunk so diagrams can be retrieved and shown next to an answer.
- **Question answering with citations:** retrieval from ChromaDB, then an LLM (`openai/gpt-oss-120b` on Groq) answers using only the retrieved chunks. The citation (for example `file.pdf, page 5` or `video.mp4, 00:58`) is built from our stored metadata, not from the LLM's claim. Each citation is also a link that opens the source at the exact PDF page or video second (this works when the original files are in `data/`).
- **Out-of-scope refusal:** two gates. First, a distance threshold on the best retrieval match. Second, the LLM must report that the chunks were sufficient.
- **Adaptive quiz:** question types are MCQ, short answer and numerical. Each question carries a difficulty and a source location (page or timestamp). A student with no history gets a fixed diagnostic start: easy, medium, hard for the first three questions, after which the adaptive rule continues from the highest level answered correctly. Difficulty is chosen by a deterministic rule from recent correctness, not by the LLM. Mastery per topic is a simple transparent score (+10 correct, -7 incorrect, clamped to 0-100), not Bayesian Knowledge Tracing or IRT.
- **Question quality checks:** each answer key is cross-checked by a second model (`qwen/qwen3.8-27b`); if it disagrees, the question is regenerated (up to 3 attempts), and if the verifier is unavailable the question is served and shown without the cross-check flag. A new question that shares 70% or more of its words with an earlier one in the same quiz is regenerated.
- **Feedback and report:** after each answer the student sees the expected answer with its source. At the end an LLM-written report gives accuracy and the likely misconception.
- **Simulated learners and evaluation:** `simulate_multi.py` runs weak, average and strong simulated students over several sessions. `eval_full.py` scores the Q&A system with our own LLM-as-judge implementation of faithfulness, answer relevancy, context precision and context recall, plus a refusal test and a retrieval hit@4 check.

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

Run the evaluation with `python eval_full.py` (writes `eval_results_full.json`) and the simulation with `python simulate_multi.py` (writes `simulation_multi_results.json`). The numbers below come from those runs on the small corpus described at the top.

**Q&A evaluation (10 in-scope questions, 6 off-material questions).**

| Metric | Score |
|---|---|
| Faithfulness | 0.90 |
| Answer relevancy | 0.80 |
| Context precision | 0.45 |
| Context recall | 0.967 |
| Refusal accuracy | 6/6 |
| Retrieval hit@4 | 10/10 |

- Only 4 of the 10 in-scope questions have a gold page. The other 6 only check that the right file was retrieved, and with one PDF and one video that is easy to satisfy. The page-level hits are the stronger evidence.
- Context precision of 0.45 means fewer than half of the retrieved chunks were judged relevant, so retrieving 4 chunks is more than needed.
- The reference answers were drafted with AI assistance and have not been exhaustively checked against the source material.
- The judge is an LLM from the same family that answers the questions.

**Simulated learners (3 sessions of 8 questions per profile, one fixed seed per profile).**

| Profile | Mastery after sessions 1, 2, 3 | Total gain | Difficulty at end |
|---|---|---|---|
| Weak | 45, 57, 52 | +2 | medium |
| Average | 62, 57, 69 | +19 | easy |
| Strong | 62, 86, 89 | +39 | medium |

Question repetition was 0.0 (exact) and 0.0 (word overlap) across 24 questions per profile, measured with the repeat-avoidance check active, so it shows the check works, not that the model never repeats.

- Each profile was run once, so there is no confidence interval.
- Correctness is sampled from a hand-picked probability table, not graded from simulated answers. The ordering of the profiles mostly shows that the mastery formula and difficulty rule respond to that table, not that real students learn this way. With +10 and -7, mastery rises on average only when the chance of a correct answer is above about 41%.
- Final difficulty does not follow mastery: the average profile ended on easy at mastery 69, because the rule only looks at recent results.
- An earlier single-session run (8 questions, seed not reproducible) could not separate weak from average; this multi-session run replaces it.

## Known limitations

- **Small evaluation sets.** 10 in-scope and 6 off-material questions, and one simulated run per profile. Reference answers were AI-drafted and not exhaustively verified.
- **LLM judge, same model family.** `eval_full.py` uses `openai/gpt-oss-120b` both to answer and to judge. RAGAS could not be imported in our environment because of an upstream dependency break, so we implemented the same four metrics ourselves. These are not RAGAS scores, and we did not try DeepEval or TruLens.
- **Context precision is low (0.45).** Retrieval finds the right material but also returns chunks that are not needed.
- **The refusal distance threshold is not reliable on its own.** It was calibrated on the original two documents. After adding diagram chunks, some out-of-scope questions had best-match distances below it (for example "capital of India" 1.553, "photosynthesis" 1.519, "water cycle diagram" 1.262). The LLM sufficiency check refused them, and all 6 off-material test questions were refused, but this is a small test.
- **Answer verification is another LLM.** The cross-check uses a preview-tier model, runs once per question, and can be unavailable. It lowers the chance of a wrong answer key but does not guarantee correctness. Quiz questions can also contain details that the source does not state.
- **Repeat check is word overlap.** Questions with the same meaning but different wording can still repeat.
- **Diagnostic start is a fixed three-question ladder,** not a calibrated placement test.
- **Mastery is a simple heuristic,** not BKT or IRT, and the difficulty rule only looks at recent answers.
- **Diagram support is narrow.** It was tested on one PDF with 4 usable diagrams. PyMuPDF extracts embedded raster images only, so vector diagrams are missed. Descriptions are model-generated and not all were manually verified.
- **Prerequisites between concepts are not built.** Topics, subtopics and concepts are tagged, but not linked by prerequisite.
- **Ingestion is command line only.** There is no upload page in the web app.
- **Free-tier rate limits.** Long runs (evaluation, simulation) hit Groq rate limits and need retries.
- **Single user, local only.** Quiz state lives in the Flask session. The app runs in debug mode on localhost and is not hardened for deployment. Source links need the original files in `data/`, which are not in the repository.

## Repository layout

| File | Purpose |
|---|---|
| `run_ingestion.py` | Entry point: ingests a folder of PDFs and videos |
| `ingest_pdf.py`, `ingest_audio.py` | PDF text and video transcription ingestion |
| `store.py` | ChromaDB access |
| `tag_content.py` | Topic, subtopic and concept tagging |
| `extract_images.py`, `describe_images.py`, `ingest_images.py` | Diagram pipeline |
| `qa.py` | Cited question answering, refusal logic, source links |
| `tutor.py` | Question generation, answer verification, grading, difficulty rule, mastery, report |
| `app.py`, `templates/index.html` | Flask web app |
| `testset.py`, `eval_full.py` | Test set and Q&A evaluation with retrieval hit@4 |
| `simulate.py`, `simulate_multi.py` | Simulated learners (single session, multi-session) |
| `eval.py` | Metric implementations used by `eval_full.py` |
| `eval_results.json`, `eval_results_full.json`, `simulation_results.json`, `simulation_multi_results.json` | Saved results |

## Models used

All through the Groq API: `openai/gpt-oss-120b` (answers, tagging, quiz generation, evaluation judge), `whisper-large-v3` (transcription), `qwen/qwen3.8-27b` (diagram descriptions).

## AI assistance disclosure

AI tools used: Claude (Anthropic), as a chat-based coding assistant.

Claude was used throughout this project for writing and debugging code, for planning, and for drafting documentation. Parts that were AI-assisted include: the diagram pipeline (`extract_images.py`, `describe_images.py`, `ingest_images.py`), changes to `qa.py` (citation format and returning image paths), the Flask route and front-end change that shows diagrams in `app.py` and `templates/index.html`, this README, and the Devpost description and demo-video script. The other ingestion, tagging, tutoring, simulation and evaluation code was also written with AI assistance.

I ran, tested and integrated all code myself, chose what to build, and the numbers reported above come from running the scripts in this repository.
