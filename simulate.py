"""
Day 5 (cont.): Simulated student runs.

Required by Track D's evaluation criteria: "Personalization Effectiveness"
(measurable mastery gains across simulated runs) and "System Evaluation"
(question repetition rate).

Design choice: correctness is SAMPLED from a per-profile probability table
(not graded via LLM on fake answer text). This is a deliberate
simplification — simulating a believable wrong-answer string and grading
it would cost 2x the LLM calls per step for no extra signal here, since
what we're measuring is whether the ADAPTIVE LOGIC (difficulty + mastery
update) responds sensibly to a stream of right/wrong outcomes. Documented
honestly as a simplification, not presented as simulating realistic
student reasoning.
"""

import random
from typing import Dict, List
from tutor import generate_question, next_difficulty, update_mastery

# P(correct) by profile and difficulty — rough, hand-picked, not fit to data.
PROFILES = {
    "weak":    {"easy": 0.6, "medium": 0.3, "hard": 0.1},
    "average": {"easy": 0.8, "medium": 0.5, "hard": 0.3},
    "strong":  {"easy": 0.95, "medium": 0.8, "hard": 0.6},
}


def run_simulated_session(profile_name: str, topic: str, n_questions: int = 8, seed: int = None) -> Dict:
    if seed is not None:
        random.seed(seed)

    p_table = PROFILES[profile_name]
    level = "medium"
    mastery = 50.0
    history = []
    mastery_trajectory = [mastery]
    questions_asked = []

    for _ in range(n_questions):
        q = generate_question(topic, level, verify=False)
        if "error" in q:
            break
        questions_asked.append(q["question"])

        correct = random.random() < p_table[level]
        history.append(correct)
        mastery = update_mastery(mastery, correct)
        mastery_trajectory.append(mastery)
        level = next_difficulty(level, history)

    n_unique = len(set(questions_asked))
    repetition_rate = 1 - (n_unique / len(questions_asked)) if questions_asked else 0

    return {
        "profile": profile_name,
        "topic": topic,
        "n_questions": len(questions_asked),
        "mastery_trajectory": mastery_trajectory,
        "final_mastery": mastery,
        "mastery_gain": mastery_trajectory[-1] - mastery_trajectory[0],
        "repetition_rate": round(repetition_rate, 3),
        "final_difficulty": level,
    }


def run_all_profiles(topic: str, n_questions: int = 8) -> List[Dict]:
    results = []
    for profile_name in PROFILES:
        print(f"Simulating '{profile_name}' student on '{topic}' ({n_questions} questions)...")
        result = run_simulated_session(profile_name, topic, n_questions, seed=hash(profile_name) % 1000)
        results.append(result)
    return results


if __name__ == "__main__":
    import sys
    import json
    topic = " ".join(sys.argv[1:]) or "Newton's laws"

    results = run_all_profiles(topic, n_questions=8)

    print("\n=== Simulated Student Results ===")
    for r in results:
        print(f"\nProfile: {r['profile']}")
        print(f"  Mastery trajectory: {[round(m,1) for m in r['mastery_trajectory']]}")
        print(f"  Final mastery: {r['final_mastery']:.1f} (gain: {r['mastery_gain']:+.1f})")
        print(f"  Final difficulty reached: {r['final_difficulty']}")
        print(f"  Question repetition rate: {r['repetition_rate']*100:.1f}%")

    with open("simulation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved to simulation_results.json")
