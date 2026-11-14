"""Compare the three planner prompting strategies on the eval set (Track B item II).
Only meaningful with LLM_MODE=api (mock mode ignores the strategy).
Usage: python -m eval.compare_prompts"""
import json
from eval.run_eval import evaluate, load

rows = load("eval/eval_set.jsonl")
table = {s: evaluate(rows, s)["summary"] for s in ("structured", "cot", "decomposition")}
print(json.dumps(table, indent=2))
open("eval/results_prompts.json", "w").write(json.dumps(table, indent=2))
