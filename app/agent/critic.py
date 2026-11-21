"""Critic: deterministic verification of the draft answer against tool evidence."""
from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass, field
from typing import Literal

from app.agent.executor import Observation
from app.schemas import Citation

NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


@dataclass
class Verdict:
    decision: Literal["approve", "retry", "flag_for_human"]
    feedback: str = ""
    unsupported: list[str] = field(default_factory=list)


def _numbers(text: str) -> list[float]:
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def review(question: str, obs: list[Observation], answer: str, cites: list[Citation], retries_left: bool) -> Verdict:
    good = [o for o in obs if o.ok]
    retry_or_flag = "retry" if retries_left else "flag_for_human"
    if not good:
        errs = "; ".join(o.error or "" for o in obs)[:200] or "no evidence gathered"
        return Verdict(retry_or_flag, f"no successful tool results ({errs})")

    # 1. Plan/option consistency (wrong plan is the most common small-model failure)
    q = question.lower()
    for o in good:
        if o.tool == "get_fund_metric":
            if "regular" in q and o.result["plan"] != "Regular":
                return Verdict(retry_or_flag, "question asks for the Regular plan but Direct was looked up")
            if "direct" in q and o.result["plan"] != "Direct":
                return Verdict(retry_or_flag, "question asks for the Direct plan but Regular was looked up")

    # 2. Every number in the answer must be grounded in evidence or in the question
    body = re.sub(r"\[[^\]]*\]", "", answer)
    body = re.sub(r"\b(?:page|pages|pg\.?|tier|section|part)\s+\d+(?:\s*[-–]\s*\d+)?\b", "", body, flags=re.I)
    allowed = _numbers(json.dumps([o.to_dict() for o in good], default=str)) + _numbers(question)
    bad = [str(n) for n in _numbers(body)
           if not any(math.isclose(n, a, rel_tol=1e-3, abs_tol=0.01) for a in allowed)]
    if bad:
        return Verdict(retry_or_flag, f"numbers not found in sources: {bad}", unsupported=bad)

    # 3. Source-based answers must carry a citation
    if any(o.tool in ("get_fund_metric", "search_scheme_docs") for o in good) and not cites:
        return Verdict("flag_for_human", "answer has no citation")
    return Verdict("approve")
