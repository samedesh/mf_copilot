# Mutual Fund Scheme Information Document (SID) Q&A Copilot (Capstone Track B)

An agentic Retrieval-Augmented Generation (RAG) system built with FastAPI, Streamlit, SQLite FTS5, and memory-mapped dense vectors over **544 real Indian Mutual Fund Scheme Information Document (SID) PDFs** (**132,905 indexed chunks** across 547 total documents, zero ingestion errors), powered by a hosted $\le 8\text{B}$ OpenAI-compatible LLM (`Qwen/Qwen3-8B`) and `bge-small-en-v1.5` embeddings.

Ships with offline `LLM_MODE=mock` and 3 test fixture files so the full `pytest` suite (`23` unit & integration tests) runs locally without network access.

---

## System Architecture

```text
User Query (Streamlit :8501 / FastAPI :8000)
   │
   ▼
Guardrails (PII redaction, investment-advice/prediction refusal, prompt-injection check)
   │
   ▼
Planner (Qwen3-8B JSON mode: structured | cot | decomposition, Pydantic schema validation + retry)
   │
   ▼
Executor (loop detector, 10-step cap, token budget, $N.numeric reference chaining)
   ├── 1. search_scheme_docs  → Hybrid RAG: SQLite FTS5 BM25 + mmap embeddings.npy (RRF, strict scheme filter)
   ├── 2. get_fund_metric     → Structured lookup over verified data/fund_metrics.csv (fuzzy/normalized matching)
   └── 3. calculator          → Deterministic financial math (pct_change, cagr, sip_fv, expense_cost, diff)
   │
   ▼
Composer (evidence-only synthesis with [doc_id] citations)
   │
   ▼
Deterministic Critic (plan/option consistency, numeric grounding, citation check, up to 2 replans)
   │
   ▼
Response (answered | refused | needs_verification) + OpenTelemetry Spans (logs/spans.jsonl / Langfuse OTLP)
```

### Key Technical Components
| Layer | File(s) | Technical Implementation |
| :--- | :--- | :--- |
| **Offline Ingestion** | `app/rag/ingest.py` | Multi-core `pypdf` parsing (`ProcessPoolExecutor`), scheme tagging from `{Scheme Name}_{doc_id}.pdf`, SEBI SID section heading tagging (`Load Structure`, `Investment Objective`, `Asset Allocation`, `Risk Factors`, `Benchmark`, `Highlights`, `Fees and Expenses`, `Plans and Options`, `Taxation`), MD5 manifest (`data/index/manifest.json`) + per-document shard checkpointing (`data/index/shards/`), consolidating into **SQLite FTS5 (`chunks.db`)** + **`embeddings.npy`** (384-dim `float32`, `PRAGMA journal_mode=DELETE` for cross-OS mount safety). |
| **Hybrid Retriever** | `app/rag/retriever.py` | Thread-safe per-thread read-only SQLite FTS5 (`mode=ro&immutable=1`) + memory-mapped `np.load(..., mmap_mode="r")` dense vectors (<50 MB RAM vs ~2 GB for in-memory BM25), BM25-weighted RRF (`0.70`/`0.30`, `RRF_K=60`), strict two-tier scheme matching ($\ge 75\%$ token overlap), SID domain reranker (TOC demotion + SEBI section/Highlights boost), small-to-big neighbor chunk stitching, preserving `mode=("bm25", "dense", "hybrid")` and optional CrossEncoder reranking. |
| **LLM & Embed Client** | `app/llm.py` | Persistent keep-alive `httpx.Client` pooling per CA certificate, `Retry-After`-aware exponential backoff retries on `429/502/503/504` & transport errors, 512-token embedding guard (`135` words / `650` chars + progressive HTTP `422` single-chunk fallback truncation), Pydantic JSON validation + retry. |
| **Agent & Tools** | `app/agent/`, `app/tools/` | Planner (`planner.py`, SID-first routing + 2-fund comparison & calculator few-shot examples), Executor (`executor.py`), Composer (`composer.py`, 8,500-char evidence budget + verbatim quoting), Critic (`critic.py`), SQLite session memory (`state.py`), `search_docs.py` (1,200-char snippets), `fund_metrics.py` (normalized/fuzzy match + top-5 error suggestions), `calculator.py`. |
| **API & UI** | `app/main.py`, `ui/streamlit_app.py` | FastAPI with eager `get_retriever().load()` in `lifespan()` (no blocking auto-ingest), `slowapi` rate limiting, first-turn `diskcache`; wide-layout Streamlit Q&A Copilot UI with sticky header, critic trust badges, expandable retrieved SID evidence snippets, telemetry strip, and one-click sample queries. |
| **Guardrails** | `app/guardrails.py` | Regex PII redaction (PAN, Aadhaar, folio, phone, email), scope/advice refusal, indirect prompt-injection sanitization of retrieved chunks. |
| **Observability** | `app/observability.py` | OpenTelemetry nested spans (`agent.request` $\rightarrow$ `planner` $\rightarrow$ `executor` $\rightarrow$ `tool.*` $\rightarrow$ `composer` $\rightarrow$ `critic`) written to `logs/spans.jsonl` with built-in OTLP HTTP export (Langfuse / Arize Phoenix compatible). |
| **SID Corpus Utilities** | `download_sid/` | `download_sid.py` (AMFI SID downloader) and `download-listname.py` (idempotent offline PDF scheme-name extractor with `--dry-run` and `--revert`). |

---

## Quick Start (Local `.venv`)

Run all commands from the `mf_copilot/` directory:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (copy template and fill API endpoints / CA cert paths)
copy .env.example .env

# 3. Build or incrementally update the SQLite FTS5 + dense vector index
python -m app.rag.ingest                  # Resumes via data/index/manifest.json
# Optional flags: --chunk-size 150 --overlap 20 --workers 4 --rebuild

# 4. Run unit & integration tests (offline mock mode, 23 tests)
pytest -q

# 5. Start API and Streamlit UI (two terminals)
uvicorn app.main:app --reload             # API + Swagger: http://localhost:8000/docs
streamlit run ui/streamlit_app.py         # Chat UI:       http://localhost:8501
```

Sample API call:
```bash
curl -s http://localhost:8000/chat -H "content-type: application/json" \
  -d '{"session_id":"s1","question":"What is the exit load of Nippon India ELSS Tax Saver Fund?"}'
```

---

## Running in Docker Desktop

The Docker setup packages both the FastAPI backend (`:8000`) and Streamlit UI (`:8501`) from a single shared image (`mf-copilot:latest`):
* **[`Dockerfile`](../mf_copilot/Dockerfile):** `python:3.11-slim` with `PYTHONUNBUFFERED=1`, installing `requirements.txt`.
* **[`.dockerignore`](../mf_copilot/.dockerignore):** Excludes `data/`, `logs/`, `.env`, and caches so the ~800 MB of PDFs and index files are never copied into the build context.
* **[`docker-compose.yml`](../mf_copilot/docker-compose.yml):** Bind-mounts `./data:/app/data`, `./logs:/app/logs`, and `../api-key:/app/api-key:ro`, and automatically overrides `LLM_CA_CERT` and `EMBED_CA_CERT` from Windows host paths in `.env` to Linux container paths (`/app/api-key/...`).
* **Windows Bind-Mount Safety:** `retriever.py` opens `chunks.db` in read-only immutable URI mode (`mode=ro&immutable=1`) to prevent SQLite `-shm`/`-wal` file-locking errors across Windows-to-Linux container mounts.

### Commands
Ensure Docker Desktop is running and local ports `8000` / `8501` are free, then from `mf_copilot/`:

```powershell
# Build image and start both API (:8000) and UI (:8501)
docker compose up --build

# Or run detached in the background
docker compose up --build -d
docker compose logs -f

# Stop containers (preserves data/ and logs/ on host disk)
docker compose down

# Run eval or tests inside the running API container
docker compose exec api python -m eval.retrieval_eval
docker compose exec api pytest -q
```

Access after startup:
* **Streamlit UI:** `http://localhost:8501`
* **API Docs (Swagger):** `http://localhost:8000/docs`
* **Health Check:** `http://localhost:8000/health`

---

## Observability & Tracing (OpenTelemetry $\rightarrow$ Langfuse)

[`app/observability.py`](../mf_copilot/app/observability.py) instruments every request with nested OpenTelemetry spans (`agent.request` $\rightarrow$ `planner` $\rightarrow$ `executor` $\rightarrow$ `tool.*` $\rightarrow$ `composer` $\rightarrow$ `critic`), writing locally to `mf_copilot/logs/spans.jsonl` and exporting via OTLP HTTP whenever `OTEL_EXPORTER_OTLP_ENDPOINT` is configured in `mf_copilot/.env` (requires `opentelemetry-exporter-otlp-proto-http`, already in `requirements.txt`).

### Self-Hosted Langfuse Setup (Docker Desktop — `http://localhost:3000`)

1. **Start Langfuse in Docker Desktop** (in a separate folder outside `mf_copilot/`):
   ```powershell
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```
   Wait ~30–60 seconds until containers are healthy (`docker compose ps`) and `http://localhost:3000/api/public/health` returns `{"status":"OK"}`.

2. **Create a Local Project & API Keys:**
   * Open `http://localhost:3000` in your browser and sign up for a local account.
   * Create an Organization $\rightarrow$ Create a Project (e.g., `mf-copilot`).
   * Go to **Settings $\rightarrow$ API Keys $\rightarrow$ Create new API keys** and copy both `pk-lf-...` (Public Key) and `sk-lf-...` (Secret Key).

3. **Base64-Encode Your Keys (PowerShell):**
   ```powershell
   $pair = "pk-lf-YOUR_PUBLIC_KEY:sk-lf-YOUR_SECRET_KEY"
   [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pair))
   ```

4. **Configure `mf_copilot/.env`** *(must be inside `mf_copilot/.env`, not `project-docs/.env`)*:
   ```env
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000/api/public/otel
   OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<PASTE_BASE64_STRING_HERE>
   ```
   > **Important notes:**
   > * Keep **`%20`** after `Basic` (`Basic%20...`) — Python's OpenTelemetry OTLP exporter requires URL-encoded spaces in `OTEL_EXPORTER_OTLP_HEADERS`.
   > * If **`mf_copilot` runs locally in `.venv`**, use `http://localhost:3000/api/public/otel`.
   > * If **`mf_copilot` ALSO runs inside Docker**, use `http://host.docker.internal:3000/api/public/otel` so the `api` container can reach Langfuse on the host's port `3000`.

5. **Restart & Inspect Traces:**
   * Restart `mf_copilot` (`uvicorn app.main:app --reload` or `docker compose up -d`) so it reloads `mf_copilot/.env`.
   * Send a query from Streamlit (`http://localhost:8501`) or Swagger (`http://localhost:8000/docs`), wait ~5 seconds (`BatchSpanProcessor` flush interval), and refresh **`http://localhost:3000` $\rightarrow$ Tracing $\rightarrow$ Traces**.

6. **Stop Langfuse Later** (from the `langfuse/` folder — preserves recorded traces in Docker volumes):
   ```powershell
   docker compose down
   ```

---

## Evaluation Suite & Latest Benchmarks

The frozen gold evaluation set ([`eval/eval_set.jsonl`](../mf_copilot/eval/eval_set.jsonl)) contains **68 verified queries** grounded directly in the real SID PDFs under `mf_copilot/data/docs/` (`40` factual, `12` qualitative, `6` calculation, `4` comparison, `5` advice refusals, `1` direct prompt injection).

Both `run_eval.py` and `compare_prompts.py` print live per-question progress with per-strategy checkpointing (default `--workers 1` to avoid remote API `HTTP 429` concurrency throttling; pass `--workers N` when running against an unthrottled endpoint):

```powershell
# Retrieval-only benchmark: recall@1/3/5 and MRR across bm25 / dense / hybrid
python -m eval.retrieval_eval

# End-to-end agent evaluation (writes eval/results.json)
python -m eval.run_eval

# Prompt strategy comparison: structured vs cot vs decomposition (writes eval/results_prompts.json)
python -m eval.compare_prompts
# Quick smoke subset: --limit 15
```

### Prompt Strategy & Post-Optimization Benchmark (`Qwen/Qwen3-8B`, Frozen `n = 68` Real SID Queries)
| Strategy / Version | Accuracy | Completion Rate | Avg Tool Calls | Avg Latency | Avg Tokens | Failure Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`structured` (Optimized)** | **88.2%** (`60/68`) | **97.1%** (`66/68`) | **1.12** | **3.13 s** | 2,423 | `reasoning_error: 5`, `retrieval_miss: 2`, `hallucination: 1` *(0 tool errors)* |
| **`structured` (Baseline)** | 67.6% (`46/68`) | 89.7% (`61/68`) | 1.50 | 5.97 s | 1,908 | `reasoning_error: 11`, `hallucination: 6`, `retrieval_miss: 5` *(0 tool errors)* |
| **`decomposition` (Baseline)** | 60.3% (`41/68`) | 76.5% (`52/68`) | 2.18 | 32.78 s | 2,191 | `tool_call_error: 16`, `reasoning_error: 7`, `retrieval_miss: 4` |
| **`cot` (Baseline)** | 8.8% (`6/68`) | 10.3% (`7/68`) | 2.84 | 41.99 s | 1,562 | `tool_call_error: 60`, `retrieval_miss: 1`, `reasoning_error: 1` |

*Key Capstone Findings:*
1. Structured few-shot prompting eliminated tool-call schema errors (`60` $\rightarrow$ `0`), lifting baseline accuracy from `8.8%` to `67.6%`.
2. Adding 2-fund comparison & calculator few-shot examples, explicit SID routing, BM25-weighted RRF (`0.70/0.30`), TOC demotion + SEBI section/Highlights domain reranking, neighbor chunk stitching, and verbatim composer grounding boosted accuracy on the **exact same frozen gold set** from **`67.6%` $\rightarrow$ `88.2%` (+20.6 pp)** while cutting average latency nearly in half (**`5.97 s` $\rightarrow$ `3.13 s`**) and cutting hallucinations by **83%** (`6` $\rightarrow$ `1`).

---

## Track B Requirement Map

| Requirement | Implementation |
| :--- | :--- |
| **I. Spec + 60–100 gold eval set** | `eval/eval_set.jsonl` (**68 verified SID queries** grounded in real PDFs in `data/docs/`, kept frozen during optimization). |
| **II. $\ge 3$ prompting strategies** | `app/agent/planner.py` (`structured`, `cot`, `decomposition`); evaluated via `python -m eval.compare_prompts` $\rightarrow$ `eval/results_prompts.json`. |
| **III. Hybrid RAG** | `app/rag/ingest.py` (`pypdf`, SID section tagging, `CHUNK_SIZE=150`, SQLite FTS5 + `embeddings.npy`), `app/rag/retriever.py` (BM25-weighted FTS5 + `mmap` dense RRF, strict scheme filter, SID domain reranker + neighbor stitching, optional CrossEncoder), `eval/retrieval_eval.py` (**96.6% Recall@5**), mandatory `[doc_id]` citations. |
| **IV. Agentic layer** | `app/agent/` (planner / executor / composer / critic); 3 tools in `app/tools/`; SQLite memory (`state.py`); Pydantic JSON validation + retry (`llm.py`); loop detector, 10-step cap, token budget (`budget.py`, `executor.py`). |
| **V. Eval + observability** | `eval/run_eval.py` (**88.2% accuracy**, **97.1% completion**, **3.13 s latency**, failure taxonomy); OpenTelemetry spans (`app/observability.py`) $\rightarrow$ `logs/spans.jsonl` + OTLP export (Self-Hosted Langfuse v4). |
| **VI. Production shipping** | FastAPI + Analytical Streamlit Q&A Copilot UI, Docker / Compose, `slowapi` rate limiting, `diskcache`, guardrails (`app/guardrails.py`: advice refusal, PII redaction, prompt-injection filtering), human-in-the-loop `needs_verification` status. |
| **"Compensating for a small model"** | Deterministic `calculator` tool, verified `fund_metrics.csv` lookup, strict scheme/plan filtering, deterministic numeric-grounding critic, constrained JSON + few-shot prompting + bounded retries. |

---

## Related Documentation
* [`project-docs/technical-report.pdf`](technical-report.pdf) & [`project-docs/technical-report.md`](technical-report.md) — 12-page Capstone Track B technical report.
* [`project-docs/target-500.md`](target-500.md) — Detailed scaling architecture for 500+ SIDs (storage evolution, token safety, memory footprint, and `fund_metrics.csv` trade-offs).
* [`project-docs/steps-Gemini.md`](steps-Gemini.md) — Post-implementation execution, evaluation, and submission checklist.
* [`project-docs/explain.md`](explain.md) — Codebase walkthrough.

---

## Limitations (For Transparent Reporting)
* **Scanned / Image-Only Pages:** `pypdf` extracts digital text only (no OCR), so scanned PDF pages or chart-only images are skipped.
* **Deterministic Critic Scope:** `app/agent/critic.py` deterministically verifies plan/option consistency, numeric grounding, and presence of citations, but does not perform semantic NLI verification of free-text prose.
* **Selective `fund_metrics.csv`:** SIDs contain statutory TER slabs and multi-clause exit load text rather than live monthly AUM/TER figures; therefore `fund_metrics.csv` is intentionally kept small and verified while hybrid RAG (`search_scheme_docs`) serves the 544 real SIDs.
* **Small-Model Sensitivity to Prompt Format:** Without few-shot JSON examples, `Qwen/Qwen3-8B` exhibits high tool-call schema error rates under pure `cot` (`60/68` failures), demonstrating why structured few-shot prompting and Pydantic retry loops are critical at $\le 8\text{B}$ scale.
* **Mock Mode vs. Real Eval:** `LLM_MODE=mock` uses a rule-based planner and hashed vectors for fast CI/unit tests (`pytest`); all reported accuracy numbers must come from `LLM_MODE=api`.
