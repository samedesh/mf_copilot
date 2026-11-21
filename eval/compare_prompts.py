"""Compare the three planner prompting strategies on the eval set (Track B item II).
Only meaningful with LLM_MODE=api (mock mode ignores the strategy).
Usage: python -m eval.compare_prompts [--workers 3] [--limit N] [--set eval/eval_set.jsonl]"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from eval.run_eval import evaluate, load

STRATEGIES = ("structured", "cot", "decomposition")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare structured, cot, and decomposition planner strategies.")
    ap.add_argument("--set", default="eval/eval_set.jsonl")
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker threads per strategy (default: 1 to avoid API 429 throttling).")
    ap.add_argument("--limit", type=int, help="Optional limit on number of questions per strategy.")
    ap.add_argument("--out", default="eval/results_prompts.json")
    a = ap.parse_args()

    rows = load(a.set)
    if a.limit:
        rows = rows[:a.limit]

    out_path = Path(a.out)
    table: dict[str, dict] = {}
    for s in STRATEGIES:
        res = evaluate(rows, strategy=s, workers=a.workers, label=s)
        table[s] = res["summary"]
        # Checkpoint after each strategy completes
        out_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"Saved checkpoint for [{s}] -> {out_path}\n", flush=True)

    print("=== Final Comparison ===", flush=True)
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
