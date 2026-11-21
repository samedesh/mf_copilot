"""Hybrid retrieval: SQLite FTS5 BM25 + memory-mapped dense vectors (RRF), optional cross-encoder rerank."""
from __future__ import annotations
import json
import re
import sqlite3
import threading

import numpy as np

from app.config import settings
from app.errors import ToolError
from app.llm import embed

RRF_K = 60
W_BM25 = 0.70
W_DENSE = 0.30
_TOC_DOTS = re.compile(r"\.{4,}")
_PAGE_NUM = re.compile(r"^page\s+(\d+)", re.I)
_SECTION_INTENTS: list[tuple[re.Pattern, set[str]]] = [
    (re.compile(r"\b(exit load|entry load|load structure|redemption load)\b", re.I), {"load structure", "highlights"}),
    (re.compile(r"\b(benchmark|index|tri)\b", re.I), {"benchmark", "highlights"}),
    (re.compile(r"\b(investment objective|objective)\b", re.I), {"investment objective", "highlights"}),
    (re.compile(r"\b(minimum|min sip|application amount|lump\s*sum)\b", re.I), {"minimum investment", "highlights"}),
    (re.compile(r"\b(risk|riskometer|risk factor)\b", re.I), {"risk factors", "highlights"}),
    (re.compile(r"\b(asset allocation|equity|debt allocation)\b", re.I), {"asset allocation", "highlights"}),
    (re.compile(r"\b(expense|ter|fee|recurring)\b", re.I), {"fees and expenses", "highlights"}),
    (re.compile(r"\b(tax|taxation|capital gain)\b", re.I), {"taxation"}),
    (re.compile(r"\b(lock[\s\-]*in|direct|regular|idcw|growth|option)\b", re.I), {"plans and options", "highlights"}),
]
_SCHEME_STOPWORDS = {
    "fund", "funds", "india", "mutual", "scheme", "schemes", "plan", "plans",
    "series", "regular", "direct", "growth", "idcw", "dividend", "option",
    "the", "of", "and", "for", "in", "a", "an", "mf",
}


def tokenize(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def _norm_str(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _scheme_tokens(t: str) -> set[str]:
    return {w for w in tokenize(t) if w not in _SCHEME_STOPWORDS and len(w) > 1}


class Retriever:
    def __init__(self) -> None:
        self.db_path = None
        self._local = threading.local()
        self._load_lock = threading.Lock()
        self._doc_catalog: list[tuple[str, str, str, set[str]]] = []  # (doc_id, norm_scheme, norm_doc, tokens)
        self.emb = None
        # Legacy fallback fields (only used if chunks.db doesn't exist yet)
        self.chunks: list[dict] = []
        self.bm25 = None
        self._reranker = None

    def _get_conn(self) -> sqlite3.Connection | None:
        if self.db_path is None:
            return None
        conn = getattr(self._local, "conn", None)
        if conn is None or getattr(self._local, "db_path", None) != self.db_path:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            try:
                uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro&immutable=1"
                conn = sqlite3.connect(uri, uri=True)
            except sqlite3.OperationalError:
                conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            self._local.db_path = self.db_path
        return conn

    def load(self) -> None:
        with self._load_lock:
            dbp = settings.index_dir / "chunks.db"
            ep = settings.index_dir / "embeddings.npy"
            cp = settings.index_dir / "chunks.json"

            if dbp.exists() and ep.exists():
                self.db_path = dbp
                self.emb = np.load(ep, mmap_mode="r")
                conn = self._get_conn()
                rows = conn.execute("SELECT DISTINCT doc_id, scheme FROM chunks").fetchall()
                self._doc_catalog = [
                    (
                        r["doc_id"],
                        _norm_str(r["scheme"] or ""),
                        _norm_str(r["doc_id"].replace("_", " ")),
                        _scheme_tokens(f"{r['scheme'] or ''} {r['doc_id'].replace('_', ' ')}"),
                    )
                    for r in rows
                ]
                self.bm25 = None
                self.chunks = []
                return

            # Legacy JSON fallback if user hasn't re-indexed to SQLite yet
            if cp.exists() and ep.exists():
                from rank_bm25 import BM25Okapi
                self.db_path = None
                self.chunks = json.loads(cp.read_text(encoding="utf-8"))
                self.emb = np.load(ep, mmap_mode="r")
                self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
                return

            raise ToolError("search index not built; run: python -m app.rag.ingest")

    def _rerank(self, query: str, cands: list[dict], k: int) -> list[dict]:
        if settings.reranker == "none" or not cands:
            return cands[:k]
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # optional heavy dependency
            self._reranker = CrossEncoder(settings.reranker)
        scores = self._reranker.predict([(query, c["text"]) for c in cands])
        return [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])][:k]

    def _match_doc_ids(self, scheme: str | None) -> list[str] | None:
        """Strict scheme resolver: exact/substring match first, then high token overlap (>=75%)."""
        if not scheme or not self._doc_catalog:
            return None
        s_norm = _norm_str(scheme)
        if not s_norm:
            return None

        # Tier 1: Exact or substring match against normalized scheme / doc_id
        exact = [
            doc_id for doc_id, norm_s, norm_d, _ in self._doc_catalog
            if (norm_s and (s_norm == norm_s or s_norm in norm_s or (len(norm_s) >= 8 and norm_s in s_norm)))
            or s_norm in norm_d
        ]
        if exact:
            return exact

        # Tier 2: High-precision token overlap (prevents single-word AMC leakage across 500+ SIDs)
        q_toks = _scheme_tokens(scheme)
        if not q_toks:
            return None

        scored: list[tuple[float, float, str]] = []
        for doc_id, _, _, doc_toks in self._doc_catalog:
            if not doc_toks:
                continue
            inter = len(q_toks & doc_toks)
            if inter == 0:
                continue
            recall = inter / len(q_toks)
            jaccard = inter / len(q_toks | doc_toks)
            scored.append((recall, jaccard, doc_id))

        if not scored:
            return None

        scored.sort(key=lambda x: (-x[0], -x[1]))
        best_recall = scored[0][0]
        if best_recall >= 0.75:
            return [doc_id for rec, _, doc_id in scored if rec >= max(0.75, best_recall - 0.05)]
        if best_recall >= 0.60:
            return [doc_id for rec, _, doc_id in scored if rec >= best_recall - 0.01][:10]
        return None

    def _fts5_bm25(self, query: str, doc_ids: list[str] | None, limit: int = 25) -> list[int]:
        toks = [t for t in dict.fromkeys(tokenize(query)) if len(t) > 1][:12]
        if not toks:
            toks = tokenize(query)[:12]
        conn = self._get_conn()
        if not toks or conn is None:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in toks)
        try:
            if doc_ids:
                placeholders = ",".join("?" for _ in doc_ids)
                sql = (
                    f"SELECT c.row_idx FROM chunks_fts "
                    f"JOIN chunks c ON c.row_idx = chunks_fts.rowid "
                    f"WHERE chunks_fts MATCH ? AND c.doc_id IN ({placeholders}) "
                    f"ORDER BY bm25(chunks_fts) LIMIT ?"
                )
                rows = conn.execute(sql, [fts_query, *doc_ids, limit]).fetchall()
            else:
                sql = (
                    "SELECT rowid AS row_idx FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?"
                )
                rows = conn.execute(sql, [fts_query, limit]).fetchall()
            return [int(r[0]) for r in rows]
        except sqlite3.OperationalError:
            return []

    def _dense_top(self, query: str, doc_ids: list[str] | None, limit: int = 25) -> list[int]:
        conn = self._get_conn()
        if self.emb is None or conn is None:
            return []
        q_vec = embed([query])[0]
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            rows = conn.execute(
                f"SELECT row_idx FROM chunks WHERE doc_id IN ({placeholders})", doc_ids
            ).fetchall()
            if not rows:
                return []
            idx = np.array([r[0] for r in rows], dtype=int)
            scores = self.emb[idx] @ q_vec
            top = np.argsort(-scores)[:limit]
            return [int(idx[j]) for j in top]

        # Full-corpus memory-mapped scan in blocks to keep peak RAM low
        n = self.emb.shape[0]
        block = 50_000
        if n <= block:
            scores = self.emb @ q_vec
            return [int(j) for j in np.argsort(-scores)[:limit]]

        top_scores: list[tuple[float, int]] = []
        for start in range(0, n, block):
            end = min(start + block, n)
            s = self.emb[start:end] @ q_vec
            k_blk = min(limit, end - start)
            part = np.argpartition(-s, k_blk - 1)[:k_blk]
            for j in part:
                top_scores.append((float(s[j]), start + int(j)))
        top_scores.sort(key=lambda x: -x[0])
        return [row_idx for _, row_idx in top_scores[:limit]]

    def _fetch_chunks(self, row_ids: list[int], stitch_neighbor: bool = True) -> dict[int, dict]:
        conn = self._get_conn()
        if not row_ids or conn is None:
            return {}
        fetch_ids = list(dict.fromkeys(row_ids + ([r + 1 for r in row_ids] if stitch_neighbor else [])))
        placeholders = ",".join("?" for _ in fetch_ids)
        rows = conn.execute(
            f"SELECT row_idx, id, doc_id, section, scheme, month, text FROM chunks WHERE row_idx IN ({placeholders})",
            fetch_ids,
        ).fetchall()
        by_id = {int(r["row_idx"]): dict(r) for r in rows}
        out: dict[int, dict] = {}
        for rid in row_ids:
            c = by_id.get(rid)
            if not c:
                continue
            c = dict(c)
            if stitch_neighbor:
                nxt = by_id.get(rid + 1)
                if nxt and nxt["doc_id"] == c["doc_id"] and nxt["section"] == c["section"] and len(c["text"]) < 1050:
                    # Append non-overlapping continuation tail from the next chunk on the same page
                    nxt_words = nxt["text"].split()
                    tail = " ".join(nxt_words[20:]) if len(nxt_words) > 25 else nxt["text"]
                    if tail:
                        c["text"] = (c["text"] + " " + tail)[:1200]
            out[rid] = c
        return out

    def _domain_rerank(self, query: str, cands: list[dict]) -> list[dict]:
        """Lightweight SID domain reranker: demote TOC noise, boost matching SEBI sections & Highlights."""
        wanted_secs: set[str] = set()
        for pat, secs in _SECTION_INTENTS:
            if pat.search(query):
                wanted_secs |= secs

        rescored: list[dict] = []
        for c in cands:
            score = float(c["score"])
            txt_low = c.get("text", "").lower()
            sec = c.get("section", "")

            # Demote Table of Contents / Index chunks that match keywords without containing answers
            is_toc = "table of contents" in txt_low or len(_TOC_DOTS.findall(c.get("text", ""))) >= 3
            if is_toc:
                score *= 0.45
            else:
                # Boost SEBI section tag matching query intent
                if wanted_secs and " | " in sec:
                    topic = sec.split(" | ", 1)[1].strip().lower()
                    if topic in wanted_secs:
                        score *= 1.35
                # Slight boost for early SID Highlights/Summary pages (pages 1-12)
                m_pg = _PAGE_NUM.match(sec)
                if m_pg and int(m_pg.group(1)) <= 12:
                    score *= 1.12

            rescored.append({**c, "score": round(score, 5)})
        rescored.sort(key=lambda x: -x["score"])
        return rescored

    def _search_legacy(self, query: str, k: int, scheme: str | None, mode: str) -> list[dict]:
        """Fallback search for legacy chunks.json indices."""
        s_norm = _norm_str(scheme) if scheme else ""
        q_toks = _scheme_tokens(scheme) if scheme else set()

        def _match(c: dict) -> bool:
            if not s_norm:
                return True
            cs = _norm_str(c.get("scheme", ""))
            cd = _norm_str(c.get("doc_id", "").replace("_", " "))
            if (cs and s_norm in cs) or s_norm in cd:
                return True
            doc_toks = _scheme_tokens(f"{cs} {cd}")
            return bool(q_toks) and (len(q_toks & doc_toks) / len(q_toks)) >= 0.75

        idx = np.array([i for i, c in enumerate(self.chunks) if _match(c)], dtype=int)
        if idx.size == 0:
            idx = np.arange(len(self.chunks), dtype=int)
        fused: dict[int, float] = {}
        if mode in ("hybrid", "bm25"):
            s = np.array(self.bm25.get_scores(tokenize(query)))[idx]
            for r, j in enumerate(np.argsort(-s)[:20]):
                fused[int(idx[j])] = fused.get(int(idx[j]), 0) + 1 / (RRF_K + r + 1)
        if mode in ("hybrid", "dense"):
            s = (self.emb[idx] @ embed([query])[0])
            for r, j in enumerate(np.argsort(-s)[:20]):
                fused[int(idx[j])] = fused.get(int(idx[j]), 0) + 1 / (RRF_K + r + 1)
        order = sorted(fused, key=lambda i: -fused[i])[:20]
        cands = [{**self.chunks[i], "score": round(fused[i], 5)} for i in order]
        return self._rerank(query, cands, k)

    def search(self, query: str, k: int | None = None, scheme: str | None = None, mode: str = "hybrid") -> list[dict]:
        if self.db_path is None and self.bm25 is None:
            self.load()
        k = k or settings.top_k

        if self.db_path is None:
            return self._search_legacy(query, k, scheme, mode)

        doc_ids = self._match_doc_ids(scheme)
        fused: dict[int, float] = {}
        w_bm25 = W_BM25 if mode == "hybrid" else 1.0
        w_dense = W_DENSE if mode == "hybrid" else 1.0

        if mode in ("hybrid", "bm25"):
            for r, row_idx in enumerate(self._fts5_bm25(query, doc_ids, limit=30)):
                fused[row_idx] = fused.get(row_idx, 0.0) + w_bm25 / (RRF_K + r + 1)

        if mode in ("hybrid", "dense"):
            for r, row_idx in enumerate(self._dense_top(query, doc_ids, limit=30)):
                fused[row_idx] = fused.get(row_idx, 0.0) + w_dense / (RRF_K + r + 1)

        if not fused:
            return []

        order = sorted(fused, key=lambda i: -fused[i])[:25]
        fetched = self._fetch_chunks(order)
        cands = []
        for row_idx in order:
            c = fetched.get(row_idx)
            if c:
                c.pop("row_idx", None)
                cands.append({**c, "score": round(fused[row_idx], 5)})
        cands = self._domain_rerank(query, cands)
        return self._rerank(query, cands, k)


_instance: Retriever | None = None


def get_retriever(reload: bool = False) -> Retriever:
    global _instance
    if _instance is None or reload:
        _instance = Retriever()
    return _instance
