"""End-to-end evaluation: accuracy, refusal accuracy, completion rate, avg tool calls, latency, tokens,
and a failure taxonomy (retrieval_miss / reasoning_error / tool_call_error / hallucination).
Usage: python -m eval.run_eval [--set eval/eval_set.jsonl] [--strategy structured] [--workers 3] [--limit N]"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.agent.graph import run_agent
from app.config import settings
from app.observability import init_tracing
from app.rag.retriever import get_retriever


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


def _eval_one(idx: int, r: dict, strategy: str | None) -> tuple[int, dict]:
    res = run_agent(r["question"], f"eval-{r['id']}", strategy=strategy, use_memory=False)
    if r["expect_refusal"]:
        ok = res.status == "refused"
    else:
        ok = res.status == "answered" and all(s.lower() in res.answer.lower() for s in r["expected_contains"])
    fail = None if ok else classify_failure(r, res)
    detail = {
        "id": r["id"],
        "type": r["type"],
        "correct": ok,
        "status": res.status,
        "failure": fail,
        "tool_calls": res.tool_calls,
        "latency_ms": res.latency_ms,
        "tokens": res.tokens,
    }
    return idx, detail


def evaluate(
    rows: list[dict],
    strategy: str | None = None,
    workers: int = 3,
    label: str | None = None,
) -> dict:
    init_tracing()
    try:
        get_retriever().load()
    except Exception:
        pass

    tag = label or strategy or settings.planner_strategy
    n = len(rows)
    t0 = time.time()
    workers = max(1, workers)
    print(f"=== Running eval [{tag}] on {n} questions (workers={workers}) ===", flush=True)

    ordered: list[dict | None] = [None] * n
    done_count = 0
    ok_count = 0

    def _report(detail: dict) -> None:
        nonlocal done_count, ok_count
        done_count += 1
        if detail["correct"]:
            ok_count += 1
        acc = (ok_count / done_count) * 100
        status_str = "OK" if detail["correct"] else f"FAIL ({detail['failure']})"
        elapsed = time.time() - t0
        print(
            f"[{tag} {done_count:2d}/{n}] {detail['id']} -> {status_str:<24} "
            f"| acc={acc:5.1f}% | {detail['latency_ms']/1000:4.1f}s (elapsed {elapsed:5.1f}s)",
            flush=True,
        )

    if workers == 1:
        for i, r in enumerate(rows):
            _, detail = _eval_one(i, r, strategy)
            ordered[i] = detail
            _report(detail)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_eval_one, i, r, strategy) for i, r in enumerate(rows)]
            for fut in as_completed(futures):
                idx, detail = fut.result()
                ordered[idx] = detail
                _report(detail)

    details = [d for d in ordered if d is not None]
    fails = Counter(d["failure"] for d in details if d["failure"])
    summary = {
        "n": n,
        "accuracy": round(sum(d["correct"] for d in details) / n, 3) if n else 0.0,
        "completion_rate": round(sum(d["status"] in ("answered", "refused") for d in details) / n, 3) if n else 0.0,
        "avg_tool_calls": round(statistics.mean(d["tool_calls"] for d in details), 2) if n else 0.0,
        "avg_latency_ms": round(statistics.mean(d["latency_ms"] for d in details), 1) if n else 0.0,
        "avg_tokens": round(statistics.mean(d["tokens"] for d in details), 1) if n else 0.0,
        "wall_time_s": round(time.time() - t0, 1),
        "failure_taxonomy": dict(fails),
    }
    return {"summary": summary, "details": details}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run end-to-end agent evaluation with live progress.")
    ap.add_argument("--set", default="eval/eval_set.jsonl")
    ap.add_argument("--strategy")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel worker threads (default: 1 to avoid API 429 throttling).")
    ap.add_argument("--limit", type=int, help="Optional limit on number of questions to evaluate.")
    a = ap.parse_args()
    rows = load(a.set)
    if a.limit:
        rows = rows[:a.limit]
    out = evaluate(rows, a.strategy, workers=a.workers)
    Path("eval/results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSummary:", flush=True)
    print(json.dumps(out["summary"], indent=2), flush=True)
