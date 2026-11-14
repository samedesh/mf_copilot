from __future__ import annotations
from app.errors import ToolError
from app.guardrails import sanitize_context
from app.rag.retriever import get_retriever


def run(query: str, scheme: str | None = None, top_k: int = 5) -> dict:
    hits = get_retriever().search(query, k=top_k, scheme=scheme)
    if not hits:
        raise ToolError("no matching passages found")
    out, removed = [], 0
    for h in hits:
        text, n = sanitize_context(h["text"])
        removed += n
        out.append({"doc_id": h["doc_id"], "section": h["section"], "scheme": h["scheme"],
                    "month": h["month"], "snippet": text[:600], "score": h["score"]})
    return {"hits": out, "injection_lines_removed": removed}
