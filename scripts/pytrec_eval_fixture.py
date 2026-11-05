"""Write tests/data/pytrec_eval.json: a small run and qrels plus pytrec_eval's nDCG@10 and
Recall@100 for them, so the Rust metrics can be checked against trec_eval's own code.

    uv run --python 3.12 --with pytrec-eval-terrier scripts/pytrec_eval_fixture.py
"""

import json

import pytrec_eval

GRADES = [2, None, 1, -1, 0, None, 3, 1, None, 0, 1]  # None = unjudged

# q1: graded, scores in groups of 4 ties (so ties straddle ranks 10 and 100), doc ids where
# string order differs from number order, and judged docs that were never retrieved.
q1_run = {f"d{i}": float((150 - i) // 4) for i in range(150)}
q1_qrels = {f"d{i}": GRADES[i % 11] for i in range(150) if GRADES[i % 11] is not None}
q1_qrels.update({f"missing{j}": 1 + j % 2 for j in range(6)})

# q2: binary, fewer than 10 results, all tied.
q2_run = {"b1": 1.5, "b10": 1.5, "b2": 1.5, "b3": 0.5}
q2_qrels = {"b2": 1, "b10": 1, "b7": 1}

qrels = {
    "q1": q1_qrels,
    "q2": q2_qrels,
    "q3": {"c1": 2, "c2": 1},  # judged, never searched: 0 under trec_eval -c
    "q4": {"e1": 0, "e2": 0},  # nothing relevant
}
run = {
    "q1": q1_run,
    "q2": q2_run,
    "q4": {"e1": 3.0, "e3": 2.0},
    "q5": {"z": 1.0},  # searched, not judged: ignored
}

evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
per_query = evaluator.evaluate(run)
expected = {q: {"ndcg_cut_10": m["ndcg_cut_10"], "recall_100": m["recall_100"]} for q, m in per_query.items()}

with open("tests/data/pytrec_eval.json", "w") as f:
    json.dump({"qrels": qrels, "run": run, "expected": expected}, f, indent=1, sort_keys=True)
print(json.dumps(expected, indent=1, sort_keys=True))
