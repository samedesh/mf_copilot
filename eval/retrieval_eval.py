"""Retrieval quality measured SEPARATELY from answer quality: recall@k and MRR for bm25 / dense / hybrid.
Rebuild with a different chunk size first to compare chunking:  python -m app.rag.ingest --chunk-size 800
Usage: python -m eval.retrieval_eval"""
import json
from eval.run_eval import load
from app.rag.retriever import get_retriever

rows = [r for r in load("eval/eval_set.jsonl") if r.get("gold_doc")]
ret, out = get_retriever(), {}
for mode in ("bm25", "dense", "hybrid"):
    rec = {1: 0, 3: 0, 5: 0}
    mrr = 0.0
    for r in rows:
        docs = [h["doc_id"] for h in ret.search(r["question"], k=5, mode=mode)]
        for k in rec:
            rec[k] += r["gold_doc"] in docs[:k]
        mrr += next((1 / (i + 1) for i, d in enumerate(docs) if d == r["gold_doc"]), 0)
    n = len(rows)
    out[mode] = {**{f"recall@{k}": round(v / n, 3) for k, v in rec.items()}, "mrr": round(mrr / n, 3)}
print(json.dumps(out, indent=2))
