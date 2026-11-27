"""Rule-based planner used when LLM_MODE=mock (offline dev, tests, demos without an endpoint)."""
from __future__ import annotations
import re

from app.schemas import Plan, PlanStep
from app.tools import fund_metrics

METRIC_WORDS = [("exit load", "exit_load"), ("expense ratio", "expense_ratio"), ("expense", "expense_ratio"),
                ("benchmark", "benchmark"), ("riskometer", "riskometer"), ("risk level", "riskometer"),
                ("aum", "aum"), ("minimum sip", "min_sip"), ("min sip", "min_sip")]


AMC_NAMES = sorted([
    "Axis", "Bandhan", "Bank of India", "BOI AXA", "Baroda BNP Paribas", "Baroda Pioneer", "Baroda",
    "Birla Sun Life", "BNP Paribas", "Canara Robeco", "Canara", "Daiwa", "DHFL Pramerica",
    "DSP", "DWS", "ICICI Prudential", "JPMorgan", "Nippon India", "Nippon", "Tata"
], key=lambda x: -len(x))


def find_scheme(text: str) -> str | None:
    t = text.lower()
    sample = next((s for s in fund_metrics.schemes() if s.lower() in t), None)
    if sample:
        return sample
    return next((amc for amc in AMC_NAMES if amc.lower() in t), None)


def _amount(q: str) -> float | None:
    m = re.search(r"(?:₹|rs\.?\s*|inr\s*)([\d,]+(?:\.\d+)?)\s*(lakhs?|crores?)?|([\d,]+(?:\.\d+)?)\s*(lakhs?|crores?)", q)
    if not m:
        return None
    num = float((m.group(1) or m.group(3)).replace(",", ""))
    unit = (m.group(2) or m.group(4) or "").lower()
    return num * (100000 if unit.startswith("lakh") else 10000000 if unit.startswith("crore") else 1)


def mock_plan(question: str, history: list[dict]) -> Plan:
    q = question.lower()
    scheme = find_scheme(question) or next(
        (find_scheme(h["content"]) for h in reversed(history) if h["role"] == "user" and find_scheme(h["content"])), None)
    plan = "Regular" if "regular" in q else "Direct"
    option = "IDCW" if ("idcw" in q or "dividend" in q) else "Growth"
    metric = next((m for w, m in METRIC_WORDS if w in q), None)
    years, amt = re.search(r"(\d+)\s*years?", q), _amount(q)
    gross = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:return|p\.?a)", q)

    is_sample = scheme in fund_metrics.schemes() if scheme else False

    if is_sample and years and amt and ("cost" in q or "expense" in q):
        return Plan(steps=[
            PlanStep(tool="get_fund_metric", args={"scheme": scheme, "metric": "expense_ratio", "plan": plan, "option": option}),
            PlanStep(tool="calculator", args={"op": "expense_cost", "params": {
                "amount": amt, "expense_pct": "$1.numeric",
                "gross_return_pct": float(gross.group(1)) if gross else 10.0, "years": int(years.group(1))}})])
    if is_sample and metric:
        return Plan(steps=[PlanStep(tool="get_fund_metric", args={
            "scheme": scheme, "metric": metric, "plan": plan, "option": option})])
    return Plan(steps=[PlanStep(tool="search_scheme_docs", args={"query": question, "scheme": scheme, "top_k": 4})])
