"""Offline ingestion for Scheme Information Documents (SIDs):
parse (pypdf + multi-core) -> section tag -> chunk -> embed -> checkpoint -> SQLite FTS5 + embeddings.npy.

Run: python -m app.rag.ingest [--chunk-size 150] [--overlap 20] [--rebuild] [--workers 4]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from app.config import settings
from app.llm import embed

BATCH_SIZE = 16

# Major SEBI-mandated SID section patterns (checked against top lines of each PDF page)
SID_SECTIONS: list[tuple[str, re.Pattern]] = [
    ("Load Structure", re.compile(r"\b(load structure|exit load|entry load)\b", re.I)),
    ("Investment Objective", re.compile(r"\binvestment objective\b", re.I)),
    ("Asset Allocation", re.compile(r"\basset allocation\b", re.I)),
    ("Risk Factors", re.compile(r"\b(scheme specific risk factors|risk factors)\b", re.I)),
    ("Fees and Expenses", re.compile(r"\b(fees and expenses|annual scheme recurring expenses|total expense ratio)\b", re.I)),
    ("Benchmark", re.compile(r"\b(benchmark index|tier[\s\-]*1 benchmark|benchmark)\b", re.I)),
    ("Highlights", re.compile(r"\b(highlights\s*/?\s*summary of the scheme|summary of the scheme)\b", re.I)),
    ("Plans and Options", re.compile(r"\b(plans\s*(and|&)\s*options|direct plan.*regular plan)\b", re.I)),
    ("Minimum Investment", re.compile(r"\b(minimum application amount|minimum purchase|minimum sip)\b", re.I)),
    ("Taxation", re.compile(r"\b(taxation|tax treatment|capital gains tax)\b", re.I)),
]


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(65536):
            h.update(buf)
    return h.hexdigest()


def scheme_from_filename(stem: str) -> str:
    """Derive clean scheme name from '{Scheme Name}_{doc_id}' PDF stems."""
    cleaned = re.sub(r"_\d+$", "", stem).replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_")
    return cleaned


def _detect_sid_section(text: str) -> str:
    """Detect primary SEBI SID section heading from top portion of a page."""
    head = "\n".join(text.splitlines()[:12])
    for label, pat in SID_SECTIONS:
        if pat.search(head):
            return label
    return ""


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
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    from pypdf import PdfReader

    out: list[tuple[str, str]] = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages, 1):
        try:
            t = (page.extract_text() or "").strip()
        except Exception:
            continue
        if not t:
            continue
        topic = _detect_sid_section(t)
        sec = f"page {i} | {topic}" if topic else f"page {i}"
        out.append((sec, t))
    return {"scheme": scheme_from_filename(path.stem)}, out


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def _parse_file_worker(task: tuple[str, int, int]) -> tuple[str, list[dict], str | None]:
    """Worker-safe function for parallel PDF/text parsing."""
    path_str, size, ov = task
    p = Path(path_str)
    try:
        if p.suffix.lower() in (".txt", ".md"):
            meta, secs = _sections_from_text(p.read_text(encoding="utf-8"))
        elif p.suffix.lower() == ".pdf":
            meta, secs = _sections_from_pdf(p)
        else:
            return p.name, [], None

        month = meta.get("month") or (re.search(r"\d{4}-\d{2}", p.stem) or [""])[0]
        scheme = meta.get("scheme", "")
        chunks: list[dict] = []
        for sec, body in secs:
            for j, c in enumerate(chunk_words(body, size, ov)):
                chunks.append({
                    "id": f"{p.stem}::{sec}::{j}",
                    "doc_id": p.stem,
                    "section": sec,
                    "scheme": scheme,
                    "month": month,
                    "text": c,
                })
        return p.name, chunks, None
    except Exception as e:
        return p.name, [], f"{type(e).__name__}: {e}"


def _shard_key(filename: str) -> str:
    return hashlib.md5(filename.encode("utf-8")).hexdigest()


def _write_sqlite_and_vectors(index_dir: Path, shard_dir: Path, manifest: dict) -> int:
    """Consolidate all document shards into SQLite FTS5 (chunks.db) and embeddings.npy."""
    db_path = index_dir / "chunks.db"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE chunks (
            row_idx INTEGER PRIMARY KEY,
            id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            section TEXT NOT NULL,
            scheme TEXT NOT NULL,
            month TEXT NOT NULL,
            text TEXT NOT NULL
        );
        CREATE INDEX idx_chunks_scheme ON chunks(scheme);
        CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text, scheme, doc_id, section,
            content='chunks', content_rowid='row_idx',
            tokenize='porter unicode61'
        );
        """
    )

    vec_parts: list[np.ndarray] = []
    row_idx = 0

    for fname in sorted(manifest["files"]):
        entry = manifest["files"][fname]
        if entry["n_chunks"] == 0:
            continue
        key = entry["shard"]
        c_path = shard_dir / f"{key}.chunks.json"
        v_path = shard_dir / f"{key}.npy"
        if not c_path.exists() or not v_path.exists():
            continue
        doc_chunks = json.loads(c_path.read_text(encoding="utf-8"))
        doc_vecs = np.load(v_path)
        if len(doc_chunks) != doc_vecs.shape[0]:
            continue

        rows = []
        for c in doc_chunks:
            rows.append((row_idx, c["id"], c["doc_id"], c["section"], c["scheme"], c["month"], c["text"]))
            row_idx += 1

        conn.executemany(
            "INSERT INTO chunks(row_idx, id, doc_id, section, scheme, month, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        vec_parts.append(doc_vecs)

    if row_idx == 0:
        conn.close()
        raise RuntimeError(f"no indexable chunks produced from {settings.docs_dir}")

    # Populate external-content FTS5 index in one fast bulk pass
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()
    # Switch from WAL back to DELETE so read-only Docker Desktop bind mounts have no -shm/-wal lock issues
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()

    all_vecs = np.vstack(vec_parts).astype(np.float32, copy=False)
    np.save(index_dir / "embeddings.npy", all_vecs)
    return row_idx


def build_index(
    chunk_size: int | None = None,
    overlap: int | None = None,
    rebuild: bool = False,
    workers: int | None = None,
) -> int:
    size = chunk_size or settings.chunk_size
    ov = overlap if overlap is not None else settings.chunk_overlap

    index_dir = settings.index_dir
    shard_dir = index_dir / "shards"
    manifest_path = index_dir / "manifest.json"
    index_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    doc_files = [
        p for p in sorted(settings.docs_dir.glob("*"))
        if p.is_file() and p.suffix.lower() in (".pdf", ".txt", ".md")
    ]
    if not doc_files:
        raise RuntimeError(f"no documents found in {settings.docs_dir}")

    # Load existing manifest if compatible with current chunking & embedding config
    manifest: dict = {}
    if not rebuild and manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                loaded.get("chunk_size") == size
                and loaded.get("overlap") == ov
                and loaded.get("embed_model") == settings.embed_model
                and loaded.get("llm_mode") == settings.llm_mode
            ):
                manifest = loaded
        except Exception:
            manifest = {}

    if not manifest:
        manifest = {
            "chunk_size": size,
            "overlap": ov,
            "embed_model": settings.embed_model,
            "llm_mode": settings.llm_mode,
            "files": {},
        }

    # Drop deleted files from manifest
    current_names = {p.name for p in doc_files}
    for stale in list(manifest["files"]):
        if stale not in current_names:
            manifest["files"].pop(stale, None)

    # Determine which files need parsing + embedding
    pending: list[tuple[Path, str]] = []
    skipped = 0
    for p in doc_files:
        fhash = _file_hash(p)
        entry = manifest["files"].get(p.name)
        key = _shard_key(p.name)
        if (
            entry
            and entry.get("hash") == fhash
            and (entry.get("n_chunks", 0) == 0 or (
                (shard_dir / f"{key}.chunks.json").exists() and (shard_dir / f"{key}.npy").exists()
            ))
        ):
            skipped += 1
        else:
            pending.append((p, fhash))

    total = len(doc_files)
    print(f"Found {total} documents ({skipped} cached, {len(pending)} to index) | chunk_size={size}, overlap={ov}")

    t0 = time.time()
    errors: list[tuple[str, str]] = []

    if pending:
        hash_map = {p.name: h for p, h in pending}
        tasks = [(str(p), size, ov) for p, _ in pending]
        max_workers = workers or min(4, os.cpu_count() or 1)

        if len(tasks) > 3 and max_workers > 1:
            pool = ProcessPoolExecutor(max_workers=max_workers)
            iterator = pool.map(_parse_file_worker, tasks)
        else:
            pool = None
            iterator = map(_parse_file_worker, tasks)

        try:
            for idx, (fname, doc_chunks, err) in enumerate(iterator, 1):
                done_total = skipped + idx
                if err:
                    errors.append((fname, err))
                    print(f"[{done_total}/{total}] SKIP {fname} (parse error: {err})")
                    continue

                key = _shard_key(fname)
                if not doc_chunks:
                    manifest["files"][fname] = {"hash": hash_map[fname], "n_chunks": 0, "shard": key}
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    print(f"[{done_total}/{total}] SKIP {fname} (0 extractable text chunks)")
                    continue

                try:
                    batch_vecs = [
                        embed([c["text"] for c in doc_chunks[i:i + BATCH_SIZE]])
                        for i in range(0, len(doc_chunks), BATCH_SIZE)
                    ]
                    doc_vecs = np.vstack(batch_vecs).astype(np.float32, copy=False)
                except Exception as e:
                    errors.append((fname, f"embed error: {e}"))
                    print(f"[{done_total}/{total}] ERROR {fname} (embed failed: {e})")
                    continue

                # Checkpoint document shard & manifest immediately
                (shard_dir / f"{key}.chunks.json").write_text(json.dumps(doc_chunks), encoding="utf-8")
                np.save(shard_dir / f"{key}.npy", doc_vecs)
                manifest["files"][fname] = {"hash": hash_map[fname], "n_chunks": len(doc_chunks), "shard": key}
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                elapsed = time.time() - t0
                if idx <= 5 or idx % 10 == 0 or idx == len(pending):
                    print(f"[{done_total}/{total}] {fname} -> {len(doc_chunks)} chunks ({elapsed:.1f}s)")
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    # Consolidate into SQLite FTS5 + embeddings.npy
    total_chunks = _write_sqlite_and_vectors(index_dir, shard_dir, manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (index_dir / "meta.json").write_text(
        json.dumps({
            "chunk_size": size,
            "overlap": ov,
            "n": total_chunks,
            "docs": len(manifest["files"]),
            "errors": len(errors),
            "backend": "sqlite_fts5",
        }),
        encoding="utf-8",
    )

    if errors:
        (index_dir / "ingest_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Completed with {len(errors)} skipped/failed documents (see {index_dir / 'ingest_errors.json'}).")

    print(f"Consolidated {total_chunks} chunks across {len(manifest['files'])} documents into chunks.db + embeddings.npy.")
    return total_chunks


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build incremental SQLite FTS5 + dense vector index for SIDs.")
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--rebuild", action="store_true", help="Ignore manifest cache and re-index all documents.")
    ap.add_argument("--workers", type=int, help="Number of CPU worker processes for PDF parsing.")
    a = ap.parse_args()
    print("chunks indexed:", build_index(a.chunk_size, a.overlap, rebuild=a.rebuild, workers=a.workers))
