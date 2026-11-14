"""Offline ingestion: parse -> split into sections -> chunk -> embed -> index.
Run: python -m app.rag.ingest [--chunk-size 400] [--overlap 50]"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np

from app.config import settings
from app.llm import embed


def _sections_from_text(text: str) -> tuple[dict, list[tuple[str, str]]]:
    meta, sections, name, buf = {}, [], "Overview", []
    for line in text.splitlines():
        m = re.match(r"^(scheme|month|doc_type):\s*(.+)$", line.strip(), re.I)
        if m and not sections and not buf:
            meta[m.group(1).lower()] = m.group(2).strip()
            continue
        if line.startswith("## "):
            if "".join(buf).strip():
                sections.append((name, "\n".join(buf).strip()))
            name, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((name, "\n".join(buf).strip()))
    return meta, sections


def _sections_from_pdf(path: Path) -> tuple[dict, list[tuple[str, str]]]:
    import pdfplumber  # lazy: only needed for PDFs
    out = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            t = (page.extract_text() or "").strip()
            if t:
                out.append((f"page {i}", t))
    return {}, out


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def build_index(chunk_size: int | None = None, overlap: int | None = None) -> int:
    size, ov = chunk_size or settings.chunk_size, overlap if overlap is not None else settings.chunk_overlap
    chunks = []
    for p in sorted(settings.docs_dir.glob("*")):
        if p.suffix.lower() in (".txt", ".md"):
            meta, secs = _sections_from_text(p.read_text(encoding="utf-8"))
        elif p.suffix.lower() == ".pdf":
            meta, secs = _sections_from_pdf(p)
        else:
            continue
        month = meta.get("month") or (re.search(r"\d{4}-\d{2}", p.stem) or [""])[0]
        for sec, body in secs:
            for j, c in enumerate(chunk_words(body, size, ov)):
                chunks.append({"id": f"{p.stem}::{sec}::{j}", "doc_id": p.stem, "section": sec,
                               "scheme": meta.get("scheme", ""), "month": month, "text": c})
    if not chunks:
        raise RuntimeError(f"no documents found in {settings.docs_dir}")
    BATCH_SIZE = 16
    vecs = np.vstack([embed([c["text"] for c in chunks[i:i + BATCH_SIZE]]) for i in range(0, len(chunks), BATCH_SIZE)])
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    (settings.index_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    np.save(settings.index_dir / "embeddings.npy", vecs)
    (settings.index_dir / "meta.json").write_text(json.dumps({"chunk_size": size, "overlap": ov, "n": len(chunks)}))
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--overlap", type=int)
    a = ap.parse_args()
    print("chunks indexed:", build_index(a.chunk_size, a.overlap))
