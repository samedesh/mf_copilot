"""Hybrid retrieval: BM25 + dense (reciprocal rank fusion), optional cross-encoder rerank."""
from __future__ import annotations
import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings
from app.errors import ToolError
from app.llm import embed

RRF_K = 60


def tokenize(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


class Retriever:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.emb = None
        self.bm25 = None
        self._reranker = None

    def load(self) -> None:
        cp, ep = settings.index_dir / "chunks.json", settings.index_dir / "embeddings.npy"
        if not cp.exists() or not ep.exists():
            raise ToolError("search index not built; run: python -m app.rag.ingest")
        self.chunks = json.loads(cp.read_text(encoding="utf-8"))
        self.emb = np.load(ep)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def _rerank(self, query: str, cands: list[dict], k: int) -> list[dict]:
        if settings.reranker == "none":
            return cands[:k]
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # optional heavy dependency
            self._reranker = CrossEncoder(settings.reranker)
        scores = self._reranker.predict([(query, c["text"]) for c in cands])
        return [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])][:k]

    def search(self, query: str, k: int | None = None, scheme: str | None = None, mode: str = "hybrid") -> list[dict]:
        if self.bm25 is None:
            self.load()
        k = k or settings.top_k

        def _match(c: dict) -> bool:
            if not scheme:
                return True
            s = scheme.lower()
            if c.get("scheme") and s in c["scheme"].lower():
                return True
            doc_words = set(c.get("doc_id", "").lower().replace("_", " ").split())
            query_words = set(s.split()) - {"fund", "india", "mutual", "scheme"}
            if doc_words & query_words:
                return True
            return False

        idx = np.array([i for i, c in enumerate(self.chunks) if _match(c)], dtype=int)
        if idx.size == 0:
            idx = np.arange(len(self.chunks), dtype=int)
        fused: dict[int, float] = {}
        if mode in ("hybrid", "bm25"):
            s = np.array(self.bm25.get_scores(tokenize(query)))[idx]
            for r, j in enumerate(np.argsort(-s)[:20]):
                fused[int(idx[j])] = fused.get(int(idx[j]), 0) + 1 / (RRF_K + r + 1)
        if mode in ("hybrid", "dense"):
            s = (self.emb @ embed([query])[0])[idx]
            for r, j in enumerate(np.argsort(-s)[:20]):
                fused[int(idx[j])] = fused.get(int(idx[j]), 0) + 1 / (RRF_K + r + 1)
        order = sorted(fused, key=lambda i: -fused[i])[:20]
        cands = [{**self.chunks[i], "score": round(fused[i], 5)} for i in order]
        return self._rerank(query, cands, k)


_instance: Retriever | None = None


def get_retriever(reload: bool = False) -> Retriever:
    global _instance
    if _instance is None or reload:
        _instance = Retriever()
    return _instance
