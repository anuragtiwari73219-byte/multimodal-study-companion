"""
Day 4: Flask app tying ingestion (already run via CLI) + RAG Q&A + adaptive
tutoring into one web interface.

Quiz state (current level, question history, correct_answer) is kept
server-side in the Flask session so the correct answer is never sent to
the browser before grading.
"""

import os
import re
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from dotenv import load_dotenv
from qa import answer_question
from tutor import generate_question, check_answer, next_difficulty, update_mastery, generate_report

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/images/<path:filename>")
def serve_image(filename):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extracted_images")
    return send_from_directory(folder, filename)


@app.route("/source/<path:filename>")
def serve_source(filename):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    return send_from_directory(folder, filename)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is empty"}), 400
    result = answer_question(question)
    return jsonify(result)


def _words(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def _too_similar(q, seen, threshold=0.7):
    a = _words(q)
    for s in seen:
        b = _words(s)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


def _fresh_question(topic, level, seen, tries=3):
    q = generate_question(topic, level)
    for _ in range(tries - 1):
        if "error" in q or not _too_similar(q["question"], seen):
            break
        q = generate_question(topic, level)
    return q


def _src_text(q):
    t = q.get("source_label", q.get("source", "unknown"))
    v = q.get("verified")
    if v is True:
        t += " [answer key cross-checked]"
    elif v is False:
        t += " [answer key not confirmed]"
    return t


@app.route("/quiz/start", methods=["POST"])
def quiz_start():
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic is empty"}), 400

    level = "easy"  # diagnostic phase starts easy and climbs
    q = generate_question(topic, level)
    if "error" in q:
        return jsonify(q), 404

    # Server-side state: correct_answer never leaves the server until graded
    session["quiz"] = {
        "topic": topic,
        "level": level,
        "history": [],
        "question_log": [],  # full record for the post-assessment report
        "mastery": 50.0,  # start every topic at neutral 50/100
        "current_question": q["question"],
        "current_answer": q["correct_answer"],
        "current_source": _src_text(q),
        "current_loc": q.get("source_label", q["source"]),
    }

    return jsonify({
        "question": q["question"],
        "difficulty": q["difficulty"],
        "source": _src_text(q),
        "question_type": q["question_type"],
        "options": q["options"],
        "mastery": 50.0,
    })


@app.route("/quiz/answer", methods=["POST"])
def quiz_answer():
    quiz = session.get("quiz")
    if not quiz:
        return jsonify({"error": "No active quiz. Start one first."}), 400

    student_answer = request.json.get("answer", "").strip()
    correct = check_answer(quiz["current_question"], quiz["current_answer"], student_answer)
    expected_answer = quiz["current_answer"]  # save before overwriting with next question
    answered_loc = quiz.get("current_loc")

    quiz["question_log"].append({
        "question": quiz["current_question"],
        "student_answer": student_answer,
        "correct_answer": expected_answer,
        "correct": correct,
        "difficulty": quiz["level"],
    })
    quiz["history"].append(correct)
    n_done = len(quiz["history"])
    ladder = ["easy", "medium", "hard"]
    if n_done < 3:
        # diagnostic phase for a student with no history: fixed easy -> medium -> hard
        new_level = ladder[n_done]
    elif n_done == 3:
        right = [lv for lv, ok in zip(ladder, quiz["history"][:3]) if ok]
        new_level = right[-1] if right else "easy"
    else:
        new_level = next_difficulty(quiz["level"], quiz["history"])
    new_mastery = update_mastery(quiz.get("mastery", 50.0), correct)

    seen = [x["question"] for x in quiz["question_log"]]
    next_q = _fresh_question(quiz["topic"], new_level, seen)
    if "error" in next_q:
        session["quiz"] = None
        return jsonify({
            "correct": correct,
            "expected_answer": expected_answer,
            "mastery": new_mastery,
            "error": next_q["error"],
        })

    quiz["level"] = new_level
    quiz["mastery"] = new_mastery
    quiz["current_question"] = next_q["question"]
    quiz["current_answer"] = next_q["correct_answer"]
    quiz["current_source"] = _src_text(next_q)
    quiz["current_loc"] = next_q.get("source_label", next_q["source"])
    session["quiz"] = quiz

    return jsonify({
        "correct": correct,
        "expected_answer": f"{expected_answer}  [Source: {answered_loc}]",
        "mastery": new_mastery,
        "next_question": next_q["question"],
        "next_difficulty": new_level,
        "next_source": _src_text(next_q),
        "next_question_type": next_q["question_type"],
        "next_options": next_q["options"],
    })


@app.route("/quiz/finish", methods=["POST"])
def quiz_finish():
    quiz = session.get("quiz")
    if not quiz or not quiz.get("question_log"):
        return jsonify({"error": "No quiz data to report on."}), 400

    report = generate_report(quiz["topic"], quiz["question_log"], quiz["mastery"])
    session["quiz"] = None
    return jsonify(report)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
