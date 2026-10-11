import json
from eval import evaluate_test_set
from store import query
from testset import TEST_SET


def retrieval_hit(item, k=4):
    for c in query(item["question"], n_results=k):
        m = c["metadata"]
        if m.get("source_name") != item["gold_file"]:
            continue
        if "gold_page" not in item:
            return True
        if str(m.get("page_number", m.get("page"))) == str(item["gold_page"]):
            return True
    return False


located = [t for t in TEST_SET if t.get("in_scope", True) and t.get("gold_file")]
hits = [{"question": t["question"], "hit": retrieval_hit(t)} for t in located]

result = evaluate_test_set(TEST_SET)
result["retrieval_hit"] = {
    "n": len(hits),
    "hit_at_4": sum(h["hit"] for h in hits) / len(hits),
    "details": hits,
}
json.dump(result, open("eval_results_full.json", "w"), indent=2, default=str)

print("=== SUMMARY ===")
print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2, default=str))
print("Retrieval misses:", [h["question"] for h in hits if not h["hit"]])
