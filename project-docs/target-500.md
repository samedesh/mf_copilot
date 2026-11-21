# Scaling Strategy for 500+ Scheme Information Documents (Target-500 SID)

## 1. Executive Summary & Scale Estimates

Scaling the **Mutual Fund Scheme Information Document (SID) Q&A Copilot (`mf_copilot`)** from a handful of sample files to **500+ Scheme Information Document (SID) PDFs** located under `mf_copilot/data/docs/` (currently **544 SID PDFs, ~370 MB**) transitions the system from a prototype RAG to a production-grade regulatory document retrieval pipeline.

Unlike monthly factsheets (which bundle 40+ funds into one file and focus on monthly AUM/holdings), **each SID PDF is a dedicated, comprehensive statutory offer document (50–100+ pages) for a single scheme** covering investment objectives, asset allocation, risk factors, load structures, expense limits, benchmarks, and unitholder rights.

### Estimated Scale Metrics
| Metric | Initial Prototype (2–3 Docs) | Target Corpus (`data/docs/` — 544 SID PDFs) |
| :--- | :--- | :--- |
| **Document Count** | 2 PDFs + 3 TXT | **544 SID PDFs** (~370 MB on disk) |
| **Total Pages** | ~150 pages | **~30,000 – 45,000 pages** |
| **Estimated Chunks (`CHUNK_SIZE=150`, `OVERLAP=20`)** | 880 chunks | **~120,000 – 180,000 chunks** |
| **Embedding Size (`bge-small-en-v1.5`, 384-dim float32)** | ~1.3 MB | **~185 MB – 275 MB** (raw vectors) |
| **In-Memory BM25 Token Heap** | ~10 MB | **~1.5 GB – 2.5 GB RAM** (if unoptimized) |
| **Embedding API Calls (Batch size 16)** | ~55 requests | **~7,500 – 11,250 requests** |

---

## 2. Ingestion Pipeline Optimization (`app/rag/ingest.py` & `app/llm.py`)

### The Problem
Currently, `build_index()` in `app/rag/ingest.py` is a single-threaded, all-or-nothing batch job with no error handling or checkpointing:
* If **one single SID PDF** is malformed/encrypted, `pdfplumber.open()` aborts the entire run.
* If **one single embedding batch** out of ~10,000 hits a network hiccup or an HTTP 422 token-limit error (`> 512 tokens`), the entire process crashes and **100% of unsaved progress is lost**.
* `pdfplumber` parses pages sequentially on 1 CPU core (taking 1–3 hours for ~40,000 pages), and `httpx.post()` opens a fresh TLS connection for every batch.

### Recommended Architectural Changes

#### A. Incremental Indexing & File Hash Tracking (`manifest.json`)
Maintain a persistent manifest (`data/index/manifest.json`) storing the MD5 hash and chunk count of every processed SID PDF. When `ingest.py` runs, it automatically skips already-indexed files so ingestion can be safely paused, resumed, or extended when new SIDs are added to `data/docs/`:
```python
def get_file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()
```

#### B. Per-Document Checkpointing & Fault Isolation
Instead of aggregating all ~150,000 chunks in memory and writing `chunks.json` and `embeddings.npy` only at the end:
* Wrap each PDF in a `try / except` block so corrupt or image-only SIDs are logged and skipped instead of halting the pipeline.
* Persist chunks and vectors incrementally per document (e.g., to SQLite or append-only shard files) and print clear progress (`[120/544] Axis Credit Risk Fund_8518.pdf -> 240 chunks`).

#### C. Faster Multi-Core PDF Parsing (`pypdf` + `ProcessPoolExecutor`)
`pdfplumber` calculates layout bounding boxes on every character and is slow for bulk text extraction. Switching `_sections_from_pdf()` to **`pypdf.PdfReader`** (already installed) combined with `ProcessPoolExecutor(max_workers=4)` cuts parsing time for 544 SIDs from hours down to **2–5 minutes**:
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_sid_pdf, pdf_paths))
```

#### D. Strict 512-Token Guard & Persistent Connection Pooling (`app/llm.py`)
The remote `bge-small-en-v1.5` endpoint enforces a hard ceiling of **512 tokens per text** and **max batch size 16**. In SIDs, dense statutory tables, ISIN codes, and punctuation can cause 200 words to expand beyond 512 tokens (triggering HTTP `422 Unprocessable Entity`).
* Set `CHUNK_SIZE=150` and `CHUNK_OVERLAP=20` in `mf_copilot/.env`.
* Tighten `_clean_text_for_embed()` in `app/llm.py` to cap each text at **135 words and 650 characters max** (collapsing repeated leader dots/dashes first).
* Add a fallback handler: if a batch still returns HTTP 422, automatically bisect/truncate and retry rather than crashing.
* Reuse a single persistent `httpx.Client` (keep-alive connection pooling) across all ~10,000 embedding requests with exponential backoff on transient 429/503/timeout errors.

---

## 3. SID Metadata, Scheme & Section Extraction

### The Advantage of One-Scheme-Per-SID
Because every SID PDF in `data/docs/` represents a single scheme and follows the cleaned filename convention `{Scheme Name}_{doc_id}.pdf` (e.g. `Axis Banking and PSU Debt Fund_6404.pdf`, `Kotak Contra Fund_1427.pdf`), we can populate `chunk["scheme"]` accurately at the document level without complex per-page fund detection.

### Recommended Extraction Strategy in `_sections_from_pdf()`

#### A. Document-Level Scheme Tagging
Extract the scheme name directly from the filename stem (stripping the trailing `_{doc_id}` suffix) and apply it to every chunk generated from that PDF:
```python
def scheme_from_filename(stem: str) -> str:
    # "Axis Credit Risk Fund_8518" -> "Axis Credit Risk Fund"
    return re.sub(r"_\d+$", "", stem).replace("_", " ").strip()
```

#### B. SEBI SID Section Header Tagging
SEBI mandates a standardized structure across Indian mutual fund SIDs. Detecting major section headings as pages are parsed allows tagging `chunk["section"]` with both page number and topic (e.g. `"page 4 | Load Structure"`):
* `Highlights / Summary of the Scheme`
* `I. Introduction / Risk Factors`
* `II. Information about the Scheme` (Investment Objective, Asset Allocation, Benchmark, Fund Manager)
* `III. Units and Offer` (Plans & Options, Minimum Application / SIP Amount, Ongoing Offer)
* `IV. Fees and Expenses` (New Fund Offer Expenses, Annual Scheme Recurring Expenses / TER, Load Structure)

**Impact:** Filtering by `chunk["scheme"]` narrows the candidate search space from ~150,000 chunks down to **~200–300 chunks** belonging strictly to the target fund, eliminating cross-fund confusion and keeping retrieval latency under **2 ms**.

---

## 4. Memory & Storage Architecture

### Storage & Retrieval Evolution Path

| Stage | Scale | Chunks Storage | BM25 Engine | Dense Vector Engine |
| :--- | :--- | :--- | :--- | :--- |
| **Current** | < 1,000 chunks | `chunks.json` | `rank-bm25` (In-Memory) | `embeddings.npy` (In-Memory) |
| **Medium (Phase 1–2)** | 10k – 60k chunks | SQLite table (`index.db`) | SQLite FTS5 (Disk-backed) | `embeddings.npy` (`mmap_mode='r'`) |
| **Target (Phase 3)** | 100k – 200k chunks | SQLite / LanceDB | SQLite FTS5 / LanceDB Hybrid | `mmap_mode='r'` / LanceDB IVF-PQ |

### Recommended Replacement: SQLite FTS5 for BM25
Holding 150,000+ tokenized word lists in Python memory for `BM25Okapi` consumes **~1.5 GB – 2.5 GB of RAM** and adds 15–30 seconds of cold-start latency on every FastAPI reload.

Using SQLite's built-in **FTS5 (Full-Text Search)** extension:
* **Zero external servers:** Built into Python's standard `sqlite3`.
* **Disk-backed:** Consumes negligible RAM (<50 MB) and starts instantly.
* **Native BM25 + Scheme Filtering:** Filters by `scheme` and evaluates BM25 in compiled C in 1–3 milliseconds:
```sql
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    id, doc_id, section, scheme, text, tokenize='porter'
);
```

### Memory-Mapped Embeddings (`np.load(..., mmap_mode='r')`)
Instead of loading the entire ~250 MB vector matrix into RAM at server boot, memory-mapping lets the OS page in only the vector rows needed for the matched scheme:
```python
self.emb = np.load(index_path / "embeddings.npy", mmap_mode="r")
```

---

## 5. Role of `fund_metrics.csv` at 500+ SID Scale

### Why `fund_metrics.csv` Exists
In `mf_copilot`, `get_fund_metric` reads `data/fund_metrics.csv` to compensate for small ($\le 8\text{B}$) LLMs:
* **Structured `$N.numeric` Chaining with `calculator`:** `get_fund_metric` returns a typed `numeric` float (e.g. `0.85`) that `executor.py` plugs directly into `calculator` (`$1.numeric`), whereas RAG returns unstructured text snippets.
* **Strict Plan/Option Isolation:** Prevents small models from confusing Direct vs. Regular plan metrics.
* **Capstone Track B Compliance:** Satisfies the 3-tool agentic architecture (`search_scheme_docs`, `get_fund_metric`, `calculator`) and unit tests.

### Is It Recommended to Maintain for 500+ SIDs?
**No — manually maintaining (or regex-scraping) all 544 SIDs into `fund_metrics.csv` is NOT recommended:**
* Unlike monthly factsheets, **SIDs do not contain live monthly Expense Ratios (TER) or AUM**; they list multi-row SEBI statutory TER slabs and multi-condition exit load text.
* Scraping 544 heterogeneous AMFI SID layouts (from 30+ AMCs) with brittle regex risks polluting the "verified" CSV with wrong numbers, which the agent trusts blindly.

### Recommended Hybrid Approach
1. **Keep `get_fund_metric` intact** (preserving 3-tool architecture, `calculator` chaining, and all unit tests).
2. **Make `search_scheme_docs` (Hybrid RAG) the primary source of truth** for the 544 SIDs in `planner.py` (or auto-fallback transparently from `get_fund_metric` to `search_scheme_docs` when a scheme is absent from `fund_metrics.csv`).
3. **Maintain only a small, verified `fund_metrics.csv`** (e.g., 10–20 flagship schemes used in your gold eval set / demo for calculation and comparison queries), rather than attempting to keep 544 schemes in a flat CSV.

---

## 6. Application-Level Code Changes Across `mf_copilot`

Beyond `ingest.py` and `llm.py`, the following application modules require targeted updates for 544+ SID PDFs:

### A. Must-Fix Changes
1. **`app/rag/retriever.py` — Fix Cross-Fund Scheme Filter & Memory Footprint:**
   * **Bug:** `_match()` currently uses single-word intersection (`if doc_words & query_words: return True`). At 544 SIDs, querying `"Axis Credit Risk Fund"` matches **all 40+ Axis funds** and every other AMC's Credit Risk fund.
   * **Fix:** Match exact/substring against `chunk["scheme"]` first; if falling back to token overlap, require high overlap ($\ge 75\%$ of non-stopword scheme tokens or highest-scoring scheme match only).
   * **Memory:** Load `embeddings.npy` with `mmap_mode="r"` and migrate `BM25Okapi` to SQLite FTS5.
2. **`app/main.py` — Disable Blocking Auto-Ingest on Startup:**
   * **Bug:** `lifespan()` calls `build_index()` synchronously if `chunks.json` is missing, which will hang `uvicorn` indefinitely on 544 PDFs.
   * **Fix:** Remove synchronous `build_index()` from `lifespan()` (log a message instructing offline `python -m app.rag.ingest` instead), and preload `get_retriever().load()` on startup when the index exists so the first `/chat` request has no cold-start delay.

### B. High-Impact Performance & Quality Changes
3. **`app/agent/planner.py` — Route SID Queries Directly to `search_scheme_docs`:**
   * Avoid wasting an extra LLM replan round-trip (~3–5s + ~800 tokens) calling `get_fund_metric` first for schemes not present in `fund_metrics.csv`.
4. **`app/tools/fund_metrics.py` — Fuzzy Match & Bounded Error Messages:**
   * Replace exact string matching with punctuation/whitespace-normalized matching (handling `-` vs `–` en-dashes).
   * Replace `Known schemes: {schemes()}` (which would dump 544 names on every miss) with `difflib.get_close_matches(..., n=5)`.
5. **`app/tools/search_docs.py` — Prevent Snippet Chopping:**
   * Increase snippet truncation from `text[:600]` to `text[:1200]` so complete 150-word SID chunks (often 900–1,000 chars) reach `composer.py` intact.
6. **`mf_copilot/.env` & Cache Hygiene:**
   * Set `CHUNK_SIZE=150` and `CHUNK_OVERLAP=20`.
   * Clear `mf_copilot/data/cache/` after re-indexing so stale responses from the old 2-doc index are purged.

### C. Modules Requiring No Changes (Already $O(1)$ w.r.t. Corpus Size)
* `app/guardrails.py`, `app/agent/critic.py`, `app/agent/executor.py`, `app/agent/graph.py`, `app/budget.py`, `app/tools/calculator.py`, `app/agent/state.py`, `app/observability.py`, `app/schemas.py`, `ui/streamlit_app.py`.

---

## 7. Observability & Tracing (OpenTelemetry $\rightarrow$ Langfuse)

`app/observability.py` already instruments every request with **OpenTelemetry (OTel)** nested spans (`agent.request` $\rightarrow$ `planner` $\rightarrow$ `executor` $\rightarrow$ `tool.*` $\rightarrow$ `composer` $\rightarrow$ `critic`), writing to `logs/spans.jsonl` and supporting built-in OTLP HTTP export when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

### Recommended Backend: Langfuse
Because `mf_copilot` uses pure Python + `httpx` + remote `Qwen3-8B` (not LangChain), **Langfuse** is the easiest LLM-native observability dashboard to learn and connect—requiring **zero agent code rewrite** via OTLP export.

### Deployment Options (Local vs. Docker Desktop)
| Option | How `mf_copilot` Runs | How the Tracing UI Runs | Docker Needed? |
| :--- | :--- | :--- | :--- |
| **Option A: Langfuse Cloud (Free Tier) *(Recommended)*** | Locally in `.venv` | Hosted at `cloud.langfuse.com` (50k observations/mo free) | **No** |
| **Option B: Self-Hosted Langfuse** | Locally in `.venv` (or Docker) | Local Docker Compose (`http://localhost:3000`) | **Yes** (for Langfuse stack: Next.js, Postgres, ClickHouse, Redis, MinIO) |
| **Option C: Local Python Only** | Locally in `.venv` | Local `logs/spans.jsonl` (or `pip install arize-phoenix` at `localhost:6006`) | **No** |

### Configuration (Plug-and-Play in `mf_copilot/.env`)
Install `opentelemetry-exporter-otlp-proto-http` and set the standard OTLP env vars:
```env
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64(PUBLIC_KEY:SECRET_KEY)>
```
*(For self-hosted Langfuse in Docker Desktop, set `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000/api/public/otel`.)*

---

## 8. Unified Implementation Roadmap (One-Go Execution Plan)

### Phase 1: Core Scaling, Packaging, Observability & Gold Eval Set (Execute in One Go Before Bulk Indexing)
1. **`mf_copilot/.env`:** Set `CHUNK_SIZE=150` and `CHUNK_OVERLAP=20` (and document optional `OTEL_EXPORTER_OTLP_*` placeholders for Langfuse).
2. **`app/llm.py`:** Cap `_clean_text_for_embed()` at 135 words / 650 chars, switch to a persistent `httpx.Client` with keep-alive pooling & retries, and add HTTP 422 fallback truncation.
3. **`app/rag/ingest.py`:**
   * Switch PDF text extraction from `pdfplumber` to `pypdf` with per-file `try/except`, `ProcessPoolExecutor`, and progress logging.
   * Populate `chunk["scheme"]` from `{Scheme Name}_{doc_id}.pdf` filenames + detect major SEBI SID section headings (`Highlights`, `Risk Factors`, `Investment Objective`, `Asset Allocation`, `Load Structure`).
   * Implement incremental indexing & per-document checkpointing (`data/index/manifest.json`) writing chunks to SQLite (`data/index/chunks.db` + FTS5) and vectors to `embeddings.npy`.
4. **`app/rag/retriever.py`:**
   * Replace in-memory `chunks.json` + `BM25Okapi` with SQLite FTS5 disk-backed BM25 while preserving `mode=("bm25", "dense", "hybrid")` for `eval/retrieval_eval.py`.
   * Load `embeddings.npy` with `mmap_mode="r"`.
   * Fix `_match()` to use strict scheme matching (exact/substring + high token overlap) to eliminate cross-fund leakage.
5. **`app/main.py`:** Remove blocking auto-ingest in `lifespan()`, add eager `get_retriever().load()` when index exists, and purge stale `data/cache/`.
6. **`app/agent/planner.py`, `app/tools/fund_metrics.py`, `app/tools/search_docs.py`:**
   * Update planner guidance to prefer `search_scheme_docs` for SID documents while keeping `get_fund_metric` for verified CSV schemes.
   * Add normalized/fuzzy matching and top-5 error suggestions in `fund_metrics.py`.
   * Increase `search_docs.py` snippet cap from `600` to `1200`.
7. **Packaging, Observability & Repo Hygiene (`requirements.txt`, `.dockerignore`, `docker-compose.yml`, `.gitignore`, `.env.example`):**
   * Add `python-dotenv`, `pypdf`, and `opentelemetry-exporter-otlp-proto-http` to `requirements.txt` and remove the duplicate `rank-bm25` entry.
   * Add `.dockerignore` excluding `data/` and `logs/` so Docker builds stay lightweight, and configure `docker-compose.yml` to mount `../api-key:/app/api-key:ro` with container-relative CA cert paths.
   * Add `data/docs/*.pdf` to `.gitignore` to protect the repo from 370 MB+ binary PDFs.
8. **Build Verified Gold SID Eval Set (`eval/eval_set.jsonl`):**
   * Generate a verified 60–80 question gold evaluation set grounded directly in real SID PDFs under `mf_copilot/data/docs/` (verified `gold_doc` stems and `expected_contains` substrings extracted from PDF text across diverse AMCs, covering factual, qualitative, calculation, comparison, advice-refusal, and prompt-injection queries).

### Phase 2: Bulk Ingestion & Verification
1. Run `python -m app.rag.ingest` offline across all **544 SID PDFs** in `mf_copilot/data/docs/`.
2. Run unit tests (`pytest`) and live end-to-end SID queries across multiple AMCs (Axis, Kotak, HDFC, SBI, ICICI, Nippon) to confirm zero cross-fund contamination, low RAM usage, and accurate citations.

### Phase 3: Evaluation, Tracing & Precision Tuning (Post-Ingestion)
1. Connect OpenTelemetry to Langfuse (Cloud free tier or local Docker) to inspect latency waterfalls, replans, and critic flags.
2. Run `python -m eval.retrieval_eval` (recall@k, MRR), `python -m eval.run_eval` (accuracy, latency, failure taxonomy), and `python -m eval.compare_prompts` against the frozen gold `eval/eval_set.jsonl`.
3. Optionally enable cross-encoder reranking (`RERANKER=BAAI/bge-reranker-base`) on top-20 hybrid candidates if recall/MRR analysis warrants it.
