"""
Day 3: Adaptive tutoring.

Pattern reused from the email-triage project: don't let the LLM decide
difficulty (that's the flaky, non-deterministic part). LLM only generates
question/evaluation CONTENT. A deterministic Python rule decides the next
difficulty level based on recent correctness.
"""

import os
import json
from typing import Dict, List
from groq import Groq
from dotenv import load_dotenv
from store import query as retrieve_chunks

load_dotenv()
_client = None

LEVELS = ["easy", "medium", "hard"]
QUESTION_TYPES = ["mcq", "short_answer", "numerical"]


def update_mastery(current_mastery: float, correct: bool) -> float:
    """
    Simple per-topic mastery score (0-100), NOT a real Bayesian Knowledge
    Tracing or IRT model (those need calibrated parameters we don't have
    time to fit). This is a transparent, explainable substitute:
    +10 on correct, -7 on incorrect, clamped to [0, 100].
    Documented honestly as a simplification, not claimed as BKT/IRT.
    """
    delta = 10 if correct else -7
    return max(0.0, min(100.0, current_mastery + delta))


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to .env first.")
        _client = Groq(api_key=api_key)
    return _client


def next_difficulty(current_level: str, recent_results: List[bool]) -> str:
    """
    Deterministic rule (NOT LLM-decided):
      - last 2 correct  -> level up
      - last 2 incorrect -> level down
      - mixed -> stay same
    recent_results: list of bool, most recent last. Only last 2 matter.
    """
    idx = LEVELS.index(current_level)
    last_two = recent_results[-2:]

    if len(last_two) == 2 and all(last_two):
        idx = min(idx + 1, len(LEVELS) - 1)
    elif len(last_two) == 2 and not any(last_two):
        idx = max(idx - 1, 0)
    # mixed or fewer than 2 results -> no change

    return LEVELS[idx]


def _location(m):
    st = m.get("source_type")
    name = m.get("source_name", "unknown")
    if st in ("pdf", "image"):
        pg = m.get("page_number", m.get("page"))
        if pg is not None:
            return f"{name}, page {pg}"
    elif st == "video":
        return f"{name}, {m.get('timestamp', '00:00')}"
    return name


def _generate_question_once(topic: str, difficulty: str, question_type: str = None) -> Dict:
    """
    Retrieves relevant chunks for the topic, asks the LLM to write ONE
    question grounded in that content, at the given difficulty and type
    (mcq / short_answer / numerical — randomly chosen if not specified).
    Returns {question, correct_answer, source, question_type, options}
    — correct_answer is kept server-side, not shown to the student.
    For MCQ, options (list of 4 strings) is included; empty list otherwise.
    """
    import random
    question_type = question_type or random.choice(QUESTION_TYPES)

    chunks = retrieve_chunks(topic, n_results=3)
    if not chunks:
        return {"error": f"No material found on '{topic}'. Ingest more content first."}

    context = "\n\n".join(f"[Chunk {i}]\n{c['text']}" for i, c in enumerate(chunks))
    source = chunks[0]["metadata"].get("source_name", "unknown")

    type_instruction = {
        "mcq": 'Write a multiple-choice question with exactly 4 options. JSON: '
               '{"question": "...", "options": ["A", "B", "C", "D"], "correct_answer": "<one of the 4 options, exact text>"}',
        "short_answer": 'Write a short-answer conceptual question. JSON: '
                         '{"question": "...", "correct_answer": "<concise correct answer>"}',
        "numerical": 'Write a numerical problem requiring calculation. JSON: '
                      '{"question": "...", "correct_answer": "<the numeric answer with units>"}',
    }[question_type]

    system_prompt = (
        f"You are a tutor writing a {difficulty} difficulty question based ONLY on "
        f"the given course material. {type_instruction}\n"
        "Write the question, any answer options and the correct answer in English only, even if the course material or topic is in another language. Do not include Hindi or transliterated words.\n"
        "Also add a \"source_chunk\" field to the JSON: the integer N of the [Chunk N] the question is based on.\n"
        "Respond ONLY with valid JSON, no other text."
    )
    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Course material:\n{context}\n\nTopic: {topic}"},
        ],
        temperature=0.3,  # slight variety in question phrasing is fine here (not a decision step)
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip("`").replace("json\n", "", 1).strip()
        parsed = json.loads(cleaned)  # let it raise if still broken — we want to see real failures

    try:
        _i = int(parsed.get("source_chunk", 0))
    except (TypeError, ValueError):
        _i = 0
    if not 0 <= _i < len(chunks):
        _i = 0
    _m = chunks[_i]["metadata"]
    source = _m.get("source_name", source)

    return {
        "question": parsed["question"],
        "correct_answer": parsed["correct_answer"],
        "options": parsed.get("options", []),
        "question_type": question_type,
        "source": source,
        "difficulty": difficulty,
        "source_label": _location(_m),
        "_context": context,
    }


VERIFY_MODEL = "qwen/qwen3.8-27b"


def verify_question(question, correct_answer, context):
    """Cross-model check: a different model judges the answer key.
    Returns True / False, or None if the verifier was unavailable."""
    import re
    try:
        resp = _get_client().chat.completions.create(
            model=VERIFY_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You verify quiz answer keys. Use the course material for any facts needed, "
                    "solve the question yourself (recompute any numbers), then say whether the proposed answer is correct. "
                    'Respond ONLY with JSON: {"valid": true or false}')},
                {"role": "user", "content": f"Course material:\n{context}\n\nQuestion: {question}\nProposed answer: {correct_answer}"},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        m = re.search(r"\{.*?\}", raw, flags=re.S)
        return bool(json.loads(m.group(0))["valid"]) if m else None
    except Exception:
        return None


def generate_question(topic, difficulty, question_type=None, verify=True, max_tries=3):
    q = None
    for _ in range(max_tries):
        q = _generate_question_once(topic, difficulty, question_type)
        if "error" in q:
            return q
        ctx = q.pop("_context", "")
        if not verify:
            q["verified"] = None
            return q
        ok = verify_question(q["question"], q["correct_answer"], ctx)
        q["verified"] = ok
        if ok is not False:
            return q
    return q


def check_answer(question: str, correct_answer: str, student_answer: str) -> bool:
    """
    Uses the LLM only to JUDGE correctness (semantic match), not to decide
    difficulty. Returns True/False. Deterministic temperature=0.
    """
    system_prompt = (
        "You grade a student's answer against the correct answer. "
        "Accept semantically correct answers even if worded differently. "
        'Respond ONLY with valid JSON: {"correct": true or false}'
    )
    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Question: {question}\n"
                f"Correct answer: {correct_answer}\n"
                f"Student answer: {student_answer}"
            )},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip("`").replace("json\n", "", 1).strip()
        parsed = json.loads(cleaned)
    return bool(parsed["correct"])


def generate_report(topic: str, question_log: List[Dict], final_mastery: float) -> Dict:
    """
    Post-assessment report: accuracy breakdown + LLM-identified misconception
    pattern from the student's incorrect answers.
    question_log: list of {question, student_answer, correct_answer, correct, difficulty}
    """
    total = len(question_log)
    correct_count = sum(1 for q in question_log if q["correct"])
    wrong = [q for q in question_log if not q["correct"]]

    misconception = ""
    if wrong:
        wrong_text = "\n".join(
            f"Q: {w['question']}\nStudent answered: {w['student_answer']}\nCorrect: {w['correct_answer']}"
            for w in wrong
        )
        client = _get_client()
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": (
                    "Given these incorrect student answers on one topic, identify the likely "
                    "underlying misconception in 1-2 sentences. Be specific, not generic."
                )},
                {"role": "user", "content": wrong_text},
            ],
            temperature=0,
        )
        misconception = response.choices[0].message.content.strip()

    return {
        "topic": topic,
        "total_questions": total,
        "correct": correct_count,
        "accuracy": round(correct_count / total, 2) if total else 0,
        "final_mastery": final_mastery,
        "likely_misconception": misconception or "No clear pattern (too few mistakes or all correct).",
    }


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "Newton's laws"

    level = "medium"
    history = []

    q = generate_question(topic, level)
    if "error" in q:
        print(q["error"])
        sys.exit(1)

    print(f"\n[{q['difficulty'].upper()}] {q['question']}")
    print(f"(source: {q['source']})")
    student_answer = input("\nYour answer: ")

    correct = check_answer(q["question"], q["correct_answer"], student_answer)
    history.append(correct)
    print("Correct!" if correct else f"Incorrect. Expected: {q['correct_answer']}")

    new_level = next_difficulty(level, history)
    print(f"\nNext question difficulty: {new_level} (was: {level})")
