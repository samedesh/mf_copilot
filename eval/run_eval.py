"""End-to-end evaluation: accuracy, refusal accuracy, completion rate, avg tool calls, latency, tokens,
and a failure taxonomy (retrieval_miss / reasoning_error / tool_call_error / hallucination).
Usage: python -m eval.run_eval [--set eval/eval_set.jsonl] [--strategy structured]"""
from __future__ import annotations
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from app.agent.graph import run_agent


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def classify_failure(row: dict, res) -> str:
    if any(not o.ok for o in res.observations):
        return "tool_call_error"
    if res.unsupported:
        return "hallucination"
    gold = row.get("gold_doc")
    if gold and gold not in {c.doc_id for c in res.citations}:
        return "retrieval_miss"
    return "reasoning_error"


def evaluate(rows: list[dict], strategy: str | None = None) -> dict:
    details, fails = [], Counter()
    for r in rows:
        res = run_agent(r["question"], f"eval-{r['id']}", strategy=strategy, use_memory=False)
        if r["expect_refusal"]:
            ok = res.status == "refused"
        else:
            ok = res.status == "answered" and all(s.lower() in res.answer.lower() for s in r["expected_contains"])
        fail = None if ok else classify_failure(r, res)
        if fail:
            fails[fail] += 1
        details.append({"id": r["id"], "type": r["type"], "correct": ok, "status": res.status, "failure": fail,
                        "tool_calls": res.tool_calls, "latency_ms": res.latency_ms, "tokens": res.tokens})
    n = len(rows)
    summary = {
        "n": n,
        "accuracy": round(sum(d["correct"] for d in details) / n, 3),
        "completion_rate": round(sum(d["status"] in ("answered", "refused") for d in details) / n, 3),
        "avg_tool_calls": round(statistics.mean(d["tool_calls"] for d in details), 2),
        "avg_latency_ms": round(statistics.mean(d["latency_ms"] for d in details), 1),
        "avg_tokens": round(statistics.mean(d["tokens"] for d in details), 1),
        "failure_taxonomy": dict(fails),
    }
    return {"summary": summary, "details": details}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval/eval_set.jsonl")
    ap.add_argument("--strategy")
    a = ap.parse_args()
    out = evaluate(load(a.set), a.strategy)
    Path("eval/results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
