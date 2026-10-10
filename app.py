"""
Day 4: Flask app tying ingestion (already run via CLI) + RAG Q&A + adaptive
tutoring into one web interface.

Quiz state (current level, question history, correct_answer) is kept
server-side in the Flask session so the correct answer is never sent to
the browser before grading.
"""

import os
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


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is empty"}), 400
    result = answer_question(question)
    return jsonify(result)


@app.route("/quiz/start", methods=["POST"])
def quiz_start():
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "Topic is empty"}), 400

    level = "medium"
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
    }

    return jsonify({
        "question": q["question"],
        "difficulty": q["difficulty"],
        "source": q["source"],
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

    quiz["question_log"].append({
        "question": quiz["current_question"],
        "student_answer": student_answer,
        "correct_answer": expected_answer,
        "correct": correct,
        "difficulty": quiz["level"],
    })
    quiz["history"].append(correct)
    new_level = next_difficulty(quiz["level"], quiz["history"])
    new_mastery = update_mastery(quiz.get("mastery", 50.0), correct)

    next_q = generate_question(quiz["topic"], new_level)
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
    session["quiz"] = quiz

    return jsonify({
        "correct": correct,
        "expected_answer": expected_answer,
        "mastery": new_mastery,
        "next_question": next_q["question"],
        "next_difficulty": new_level,
        "next_source": next_q["source"],
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
