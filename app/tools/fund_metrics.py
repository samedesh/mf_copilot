"""Structured lookup over a verified metrics table (bypasses fuzzy retrieval for key numbers)."""
from __future__ import annotations
import csv
import difflib
import re
from functools import lru_cache

from app.config import settings
from app.errors import ToolError

ALIASES = {"exit load": "exit_load", "expense ratio": "expense_ratio", "ter": "expense_ratio",
           "benchmark": "benchmark", "riskometer": "riskometer", "aum": "aum",
           "min sip": "min_sip", "minimum sip": "min_sip"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@lru_cache(maxsize=1)
def _rows() -> list[dict]:
    if not settings.metrics_csv.exists():
        raise ToolError(f"metrics table not found at {settings.metrics_csv}")
    with open(settings.metrics_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def schemes() -> list[str]:
    try:
        return sorted({r["scheme"] for r in _rows()})
    except ToolError:
        return []


def _resolve_scheme(scheme: str) -> str | None:
    target = _norm(scheme)
    if not target:
        return None
    known = schemes()
    norm_map = {_norm(s): s for s in known}
    if target in norm_map:
        return norm_map[target]
    close = difflib.get_close_matches(target, list(norm_map.keys()), n=1, cutoff=0.85)
    return norm_map[close[0]] if close else None


def run(scheme: str, metric: str, plan: str = "Direct", option: str = "Growth", month: str | None = None) -> dict:
    metric = ALIASES.get(metric.lower().strip(), metric.lower().strip())
    resolved = _resolve_scheme(scheme)
    rows = [
        r for r in _rows()
        if resolved is not None and r["scheme"] == resolved and r["metric"] == metric
        and r["plan"].lower() == plan.lower() and r["option"].lower() == option.lower()
        and (month is None or r["month"] == month)
    ]
    if not rows:
        known = schemes()
        suggestions = difflib.get_close_matches(scheme, known, n=5, cutoff=0.4) or known[:5]
        raise ToolError(
            f"no '{metric}' for {scheme} ({plan}, {option}, {month or 'any month'}). "
            f"Closest schemes in metrics table: {suggestions}. Try search_scheme_docs instead."
        )
    r = max(rows, key=lambda x: x["month"])  # latest month if none given
    try:
        numeric = float(r["value"])
    except ValueError:
        numeric = None
    return {"scheme": r["scheme"], "plan": r["plan"], "option": r["option"], "month": r["month"],
            "metric": metric, "value": r["value"], "numeric": numeric, "unit": r["unit"], "source": r["source"]}
