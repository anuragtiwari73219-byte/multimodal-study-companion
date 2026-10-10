"""
Day 3-4: System evaluation.

RAGAS 0.4.3's import is broken in this environment (depends on a
langchain_community.chat_models.vertexai submodule that no longer exists —
a real upstream dependency break, not something we can fix by waiting).
Track D's requirements explicitly allow "RAGAS, DeepEval, TruLens, or
similar" — so this implements the same three core metrics ourselves,
using our own Groq LLM as judge. This is a deliberate substitution,
documented honestly (not silently passed off as RAGAS).

Metrics (0-1 scale, higher is better):
  - Faithfulness: is the generated answer actually supported by the
    retrieved context, or does it contain unsupported claims?
  - Answer relevancy: does the answer actually address the question asked?
  - Context precision: of the chunks retrieved, how many were relevant?
  - Context recall: did the retrieved chunks contain what was needed to
    answer correctly (requires a reference/ground-truth answer)?

Test set: a small set of question/reference_answer pairs covering ingested
material, PLUS a few deliberately off-material questions to test refusal
(out-of-scope handling, a separate pass/fail, not a 0-1 score).
"""

import os
import json
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
from qa import answer_question
from store import query as retrieve_chunks

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


def _llm_score(system_prompt: str, user_prompt: str) -> float:
    """Generic helper: ask the LLM to return {"score": 0.0-1.0}."""
    client = _get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt + '\nRespond ONLY with JSON: {"score": <0.0 to 1.0>, "reason": "<one sentence>"}'},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip("`").replace("json\n", "", 1).strip()
        parsed = json.loads(cleaned)
    return float(parsed["score"]), parsed.get("reason", "")


def faithfulness(answer: str, context: str) -> tuple:
    return _llm_score(
        "Judge whether the ANSWER's claims are all supported by the CONTEXT. "
        "Score 1.0 if fully supported, 0.0 if the answer contains claims not "
        "found in the context (hallucination), partial scores in between.",
        f"Context:\n{context}\n\nAnswer:\n{answer}"
    )


def answer_relevancy(question: str, answer: str) -> tuple:
    return _llm_score(
        "Judge whether the ANSWER actually addresses the QUESTION asked "
        "(not whether it's factually correct — just whether it's on-topic "
        "and responsive). 1.0 = directly answers it, 0.0 = off-topic/evasive.",
        f"Question: {question}\n\nAnswer:\n{answer}"
    )


def context_precision(question: str, chunks: List[Dict]) -> tuple:
    """Fraction of retrieved chunks the LLM judges as relevant to the question."""
    if not chunks:
        return 0.0, "no chunks retrieved"
    relevant = 0
    for c in chunks:
        score, _ = _llm_score(
            "Judge whether this CHUNK is relevant to answering the QUESTION. "
            "Score 1.0 if clearly relevant, 0.0 if not.",
            f"Question: {question}\n\nChunk:\n{c['text']}"
        )
        if score >= 0.5:
            relevant += 1
    return relevant / len(chunks), f"{relevant}/{len(chunks)} chunks judged relevant"


def context_recall(reference_answer: str, chunks: List[Dict]) -> tuple:
    """Does the retrieved context contain enough info to produce the reference answer?"""
    context = "\n\n".join(c["text"] for c in chunks)
    return _llm_score(
        "Judge whether the CONTEXT contains enough information to arrive at "
        "the REFERENCE ANSWER. 1.0 = fully supported, 0.0 = missing key info.",
        f"Reference answer: {reference_answer}\n\nContext:\n{context}"
    )


def evaluate_test_set(test_set: List[Dict]) -> Dict:
    """
    test_set: list of {"question": str, "reference_answer": str, "in_scope": bool}
    Returns per-question results plus averaged metrics.
    """
    results = []
    for item in test_set:
        q = item["question"]
        ref = item.get("reference_answer", "")
        in_scope = item.get("in_scope", True)

        qa_result = answer_question(q)
        chunks = retrieve_chunks(q, n_results=4)

        if not in_scope:
            # Refusal test: correct behavior is no_answer_found == True
            results.append({
                "question": q,
                "type": "out_of_scope_refusal",
                "correctly_refused": qa_result.get("no_answer_found", False),
                "answer_given": qa_result["answer"],
            })
            continue

        faith_score, faith_reason = faithfulness(qa_result["answer"], "\n\n".join(c["text"] for c in chunks))
        rel_score, rel_reason = answer_relevancy(q, qa_result["answer"])
        prec_score, prec_reason = context_precision(q, chunks)
        recall_score, recall_reason = context_recall(ref, chunks) if ref else (None, "no reference answer provided")

        results.append({
            "question": q,
            "type": "in_scope",
            "answer": qa_result["answer"],
            "faithfulness": faith_score,
            "answer_relevancy": rel_score,
            "context_precision": prec_score,
            "context_recall": recall_score,
            "notes": {"faithfulness": faith_reason, "relevancy": rel_reason, "precision": prec_reason, "recall": recall_reason},
        })

    in_scope_results = [r for r in results if r["type"] == "in_scope"]
    refusal_results = [r for r in results if r["type"] == "out_of_scope_refusal"]

    def avg(key):
        vals = [r[key] for r in in_scope_results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_in_scope": len(in_scope_results),
        "n_refusal_tests": len(refusal_results),
        "avg_faithfulness": avg("faithfulness"),
        "avg_answer_relevancy": avg("answer_relevancy"),
        "avg_context_precision": avg("context_precision"),
        "avg_context_recall": avg("context_recall"),
        "refusal_accuracy": (
            sum(r["correctly_refused"] for r in refusal_results) / len(refusal_results)
            if refusal_results else None
        ),
    }

    return {"results": results, "summary": summary}


if __name__ == "__main__":
    # Edit this test set to match YOUR ingested content before running.
    # These are placeholder questions based on the Newton's laws video +
    # Python notes PDF already ingested — adjust reference answers if needed.
    test_set = [
        {
            "question": "What is Newton's second law?",
            "reference_answer": "Force equals mass times acceleration (F = ma)",
            "in_scope": True,
        },
        {
            "question": "What is Python used for?",
            "reference_answer": "Web development, software development, mathematics, and system scripting",
            "in_scope": True,
        },
        {
            "question": "Who is the current president of France?",
            "in_scope": False,
        },
        {
            "question": "What is the boiling point of mercury?",
            "in_scope": False,
        },
    ]

    print("Running evaluation... (this calls the LLM multiple times per question, may take a minute)\n")
    result = evaluate_test_set(test_set)

    print("=== Per-question results ===")
    for r in result["results"]:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        print("---")

    print("\n=== Summary ===")
    print(json.dumps(result["summary"], indent=2))

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nSaved to eval_results.json")
