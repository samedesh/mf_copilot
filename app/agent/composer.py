"""Builds the draft answer + citations from tool observations."""
from __future__ import annotations
import json

from app.agent.executor import Observation
from app.config import settings
from app.llm import LLMClient
from app.schemas import Citation

NOT_FOUND = ("I couldn't find this in the indexed documents, so I won't guess. "
             "Please check the official factsheet or Scheme Information Document.")


def citations_from(obs: list[Observation]) -> list[Citation]:
    out, seen = [], set()
    for o in obs:
        if not o.ok or not o.result:
            continue
        if o.tool == "get_fund_metric":
            doc, _, sec = o.result["source"].partition("#")
            c = Citation(doc_id=doc, section=sec or None, snippet=f"{o.result['metric']}: {o.result['value']}")
            if (c.doc_id, c.section) not in seen:
                seen.add((c.doc_id, c.section)); out.append(c)
        elif o.tool == "search_scheme_docs":
            for h in o.result["hits"][:3]:
                if (h["doc_id"], h["section"]) not in seen:
                    seen.add((h["doc_id"], h["section"]))
                    out.append(Citation(doc_id=h["doc_id"], section=h["section"], snippet=h["snippet"][:200]))
    return out


def _mock_answer(obs: list[Observation]) -> str:
    parts = []
    for o in obs:
        if not o.ok:
            continue
        r = o.result
        if o.tool == "get_fund_metric":
            parts.append(f"{r['metric'].replace('_', ' ')} of {r['scheme']} ({r['plan']}, {r['option']}) "
                         f"as of {r['month']}: {r['value']} {'' if r['unit'] == 'text' else r['unit']}".strip() + ".")
        elif o.tool == "calculator":
            parts.append(f"Calculated {r['op']}: {r['value']:,.2f} {r['unit']} (formula: {r['formula']}).".replace("  ", " "))
        elif o.tool == "search_scheme_docs":
            for h in r["hits"][:3]:
                parts.append(f"From {h['doc_id']} ({h['section']}): {h['snippet'][:250]}")
    return " ".join(parts) if parts else NOT_FOUND


def compose(question: str, obs: list[Observation], client: LLMClient) -> str:
    if not any(o.ok for o in obs):
        return NOT_FOUND
    if settings.llm_mode == "mock":
        return _mock_answer(obs)
    evidence = json.dumps([o.to_dict() for o in obs if o.ok], default=str)[:6000]
    msgs = [{"role": "system", "content": "Answer the question using ONLY the evidence. State figures with unit, "
             "plan/option and period. Show calculation steps for derived numbers. Cite sources as [doc_id]. "
             "If the evidence is insufficient, say so. Never use outside knowledge. Never give advice."},
            {"role": "user", "content": f"Question: {question}\nEvidence: {evidence}"}]
    return client.chat(msgs, max_tokens=500).strip()
