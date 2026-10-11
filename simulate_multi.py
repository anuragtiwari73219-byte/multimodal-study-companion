import json, random, re, sys
from simulate import PROFILES
from tutor import generate_question, next_difficulty, update_mastery


def words(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def similar(a, b, th=0.7):
    A, B = words(a), words(b)
    return bool(A and B) and len(A & B) / len(A | B) >= th


import time


def safe_gen(topic, level, retries=10):
    last = None
    for attempt in range(retries):
        try:
            return generate_question(topic, level, verify=False)
        except Exception as e:  # rate limit (429) or a malformed model reply
            last = e
            wait = 15 if "429" in str(e) or "rate" in str(e).lower() else 3
            print(f"  retry {attempt + 1}/{retries} after error: {str(e)[:80]}", flush=True)
            time.sleep(wait)
    raise last


def fresh(topic, level, seen, tries=4):
    q = safe_gen(topic, level)
    for _ in range(tries - 1):
        if "error" in q or not any(similar(q["question"], s) for s in seen):
            break
        q = safe_gen(topic, level)
    return q


def run_profile(profile, topic, sessions=3, n_questions=8, seed=0):
    rnd = random.Random(seed)
    p = PROFILES[profile]
    mastery, level, history, seen, out = 50.0, "medium", [], [], []
    for s in range(1, sessions + 1):
        start = mastery
        for _ in range(n_questions):
            q = fresh(topic, level, seen)
            if "error" in q:
                break
            seen.append(q["question"])
            ok = rnd.random() < p[level]
            history.append(ok)
            mastery = update_mastery(mastery, ok)
            level = next_difficulty(level, history)
        out.append({"session": s, "start_mastery": round(start, 1), "end_mastery": round(mastery, 1),
                    "gain": round(mastery - start, 1), "difficulty_after": level})
    n = len(seen)
    exact = 1 - len(set(seen)) / n if n else 0
    sem = sum(1 for i, q in enumerate(seen) if any(similar(q, s) for s in seen[:i])) / n if n else 0
    return {"profile": profile, "topic": topic, "sessions": out, "total_gain": round(mastery - 50.0, 1),
            "n_questions": n, "exact_repetition_rate": round(exact, 3), "semantic_repetition_rate": round(sem, 3)}


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "Newton's laws"
    seeds = {"weak": 11, "average": 22, "strong": 33}
    results = []
    for prof in PROFILES:
        print(f"Running {prof} (3 sessions x 8 questions)...", flush=True)
        results.append(run_profile(prof, topic, seed=seeds[prof]))
    json.dump(results, open("simulation_multi_results.json", "w"), indent=2)
    print("\n=== MULTI-SESSION RESULTS ===")
    for r in results:
        print(r["profile"], "| mastery per session:", [s["end_mastery"] for s in r["sessions"]],
              "| total gain:", r["total_gain"], "| final level:", r["sessions"][-1]["difficulty_after"],
              "| repetition exact/semantic:", r["exact_repetition_rate"], "/", r["semantic_repetition_rate"])
