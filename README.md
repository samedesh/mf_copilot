# Mutual Fund SID Q&A Copilot (`mf_copilot`)

A SEBI-compliant, multi-agent Q&A copilot for Indian Mutual Fund **Scheme Information Documents (SIDs)**, designed to deliver grounded fact retrieval, structured factsheet metrics, and deterministic financial calculations using hosted $\le$8B parameter language models.

> **Regulatory Notice:** This assistant strictly answers factual questions from verified SIDs and factsheet metrics. Per SEBI regulations, it does not provide investment advice, buy/sell recommendations, or return forecasts.

---

## Key Capabilities & Architecture

### 1. Bounded Multi-Agent Orchestrator (`app/agent/`)
Instead of an unconstrained ReAct loop, `mf_copilot` uses an explicit, bounded state machine (`app/agent/graph.py`):

```text
User Query
   │
   ▼
[ Guardrails ] ──(PII Redacted / Out-of-Scope or Advice)──► Refusal Response
   │ (Clean)
   ▼
[ Planner ] ◄───────────────────────────────────┐
   │ (Structured PlanStep DAG)                  │
   ▼                                            │ (Bounded Replan: max 2)
[ Executor ] (search_scheme_docs / get_fund_metric / calculator)
   │
   ▼
[ Composer ] ──► [ Critic ] ──(retry + feedback)┘
                    │
                    ▼ (approve)
            Verified Answer + Citations + OTel Trace
```

- **Guardrails (`app/guardrails.py`):** Redacts sensitive PII (PAN, Aadhaar, phone numbers, emails) before any LLM call and blocks investment-advice requests or prompt injections.
- **Planner (`app/agent/planner.py`):** Supports `structured`, `cot`, and `decomposition` strategies; enforces scheme filtering when a mutual fund scheme is detected in the query.
- **Executor (`app/agent/executor.py`):** Executes tool calls deterministically within step and token budgets (`app/budget.py`).
- **Composer (`app/agent/composer.py`):** Synthesizes answers strictly from tool observations and attaches document/section citations.
- **Critic (`app/agent/critic.py`):** Verifies grounding and numerical consistency, triggering bounded replans (`MAX_REPLANS=2`) when evidence is incomplete.
- **Session Memory (`app/agent/state.py`):** SQLite-backed conversation history (`data/sessions.db`) supporting multi-turn follow-ups (e.g., *"What about the regular plan?"*).

### 2. Hybrid RAG Pipeline (`app/rag/`)
- **Ingestion (`app/rag/ingest.py`):** Parses SID PDFs into sentence-aware overlapping chunks (`CHUNK_SIZE=150`, `CHUNK_OVERLAP=20`) stored in SQLite (`data/index/chunks.db`) with FTS5 full-text search and dense embeddings (`bge-small-en-v1.5`).
- **Hybrid Retrieval (`app/rag/retriever.py`):** Combines SQLite FTS5 **BM25** (`W_BM25=0.70`) and memory-mapped **dense vector cosine similarity** (`W_DENSE=0.30`) using **Reciprocal Rank Fusion** (`RRF_K=60`), enhanced with section-intent matching, scheme stopword normalization, and optional cross-encoder reranking.

### 3. Deterministic Tool Suite (`app/tools/`)
- **`search_scheme_docs` (`search_docs.py`):** Queries the hybrid RAG index with optional scheme-level filtering.
- **`get_fund_metric` (`fund_metrics.py`):** Looks up verified quantitative metrics (AUM, NAV, Direct/Regular expense ratios, etc.) from `data/fund_metrics.csv`.
- **`calculator` (`calculator.py`):** Performs exact Python arithmetic (`pct_change`, `cagr`, `sip_fv`, `expense_cost`, `diff`) so the LLM never performs mental math.

### 4. Interactive UI & Observability (`ui/`, `app/observability.py`)
- **Streamlit UI (`ui/streamlit_app.py`):**
  - Chat interface with AMC & Scheme filter dropdowns (`ui/amc_schemes.json`).
  - Sidebar example questions across SID Facts, Financial Calculations, Guardrails, and Multi-turn Memory (`ui/components/sidebar.py`).
  - **Failure Taxonomy** modal (`ui/components/taxonomy_modal.py`) for inspecting benchmark failures and root causes.
  - **Observability & Tracing** popover with Langfuse deep-links and JSON chat history export.
- **OpenTelemetry + Langfuse (`app/observability.py`):** Writes local spans to `logs/spans.jsonl` and exports OTLP HTTP traces (`/api/public/otel/v1/traces`) to Langfuse.

---

## Project Structure

```text
mf_copilot/
├── app/
│   ├── agent/                 # planner, executor, composer, critic, graph, state, mock
│   ├── rag/                   # ingest.py & hybrid BM25 + dense RRF retriever.py
│   ├── tools/                 # calculator.py, fund_metrics.py, search_docs.py
│   ├── budget.py              # Step and token budget enforcement
│   ├── config.py              # Environment-driven Settings dataclass
│   ├── errors.py              # Custom BudgetExceeded, LLMError, ToolError exceptions
│   ├── guardrails.py          # PII redaction & SEBI scope/injection checks
│   ├── llm.py                 # OpenAI-compatible LLM & embedding HTTP clients
│   ├── main.py                # FastAPI server (/health, /chat, /sessions/{id}/history)
│   ├── observability.py       # OpenTelemetry span initialization & OTLP export
│   └── schemas.py             # Pydantic request/response & citation models
├── data/                      # docs/, index/, cache/, fund_metrics.csv, sessions.db
├── eval/                      # 68-query eval_set.jsonl, run_eval.py, compare_prompts.py
├── logs/                      # spans.jsonl (local OpenTelemetry span log)
├── tests/                     # pytest unit tests (agent, guardrails, tools)
├── ui/                        # Streamlit frontend & components
├── Dockerfile                 # python:3.11-slim container image
├── docker-compose.yml         # API + UI stack
├── docker-compose.langfuse.yml# API + UI + self-hosted Langfuse stack
├── requirements.txt           # Python dependencies
└── .env.example               # Template environment configuration
```

---

## Prerequisites

Before running any commands or starting the stack, ensure the following are in place:

1. **Docker Desktop (for Docker Compose / Langfuse stacks):**
   - **Docker Desktop** installed and **running** (status: *Engine running*).
   - Switched to **Linux containers** mode on Windows (verify with `docker info`).
   - **Docker Compose v2+** (`docker compose version`).
2. **Python 3.11+ (for local runs, ingestion, evals, and tests):**
   - Required if running `uvicorn`, `streamlit`, `eval/`, or `pytest` directly on your host.
3. **Model & Embedding Access (for `LLM_MODE=api`):**
   - Valid OpenAI-compatible endpoints and API keys for your hosted $\le$8B LLM (`LLM_BASE_URL`, `LLM_API_KEY`) and embedding model (`EMBED_BASE_URL`, `EMBED_API_KEY`).
   - CA certificate files (`LLM_CA_CERT`, `EMBED_CA_CERT`) if your endpoints use private/self-signed TLS certificates.
   - *(Optional)* Set `LLM_MODE=mock` in `.env` to run offline without external LLM endpoints.
4. **RAG Index & Data Files:**
   - Ensure `data/fund_metrics.csv` and `data/index/chunks.db` exist. If the index is missing, build it from `data/docs/` using `python -m app.rag.ingest`.
5. **Available Ports:**
   - `8000` (FastAPI), `8501` (Streamlit UI), and `3000` / `9090` (plus localhost-bound Langfuse backing services `3030`, `5432`, `6379`, `8123`, `9000`, `9091` when using `docker-compose.langfuse.yml`).

---

## Setup & Running

### 1. Configure Environment
Copy `.env.example` to `.env` and set your model, embedding, and Langfuse values:
```powershell
Copy-Item .env.example .env
```
Key variables in `.env`:
- `LLM_MODE`: `mock` (offline rule-based planner for dev/tests) or `api` (hosted $\le$8B OpenAI-compatible endpoint).
- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_CA_CERT`
- `EMBED_BASE_URL`, `EMBED_API_KEY`, `EMBED_MODEL`, `EMBED_CA_CERT`
- `PLANNER_STRATEGY`: `structured` (recommended), `cot`, or `decomposition`.
- `MAX_STEPS=10`, `MAX_REPLANS=2`, `TOKEN_BUDGET=8000`, `RATE_LIMIT=20/minute`.

### 2. Option A — Docker Compose with Langfuse (Recommended)
Runs FastAPI (`:8000`), Streamlit UI (`:8501`), and self-hosted Langfuse (`:3000`):
```powershell
docker compose -f docker-compose.langfuse.yml up -d --build
```
- **Streamlit UI:** `http://localhost:8501`
- **FastAPI Docs:** `http://localhost:8000/docs`
- **Langfuse UI:** `http://localhost:3000` (default provisioned login: `admin@example.com` / `password123`)

To stop the stack:
```powershell
docker compose -f docker-compose.langfuse.yml down
```

### 3. Option B — Docker Compose (App Only)
```powershell
docker compose up -d --build
```

### 4. Option C — Local Python Setup
```powershell
pip install -r requirements.txt
```
*(Optional) Rebuild the hybrid RAG index from PDFs in `data/docs/`:*
```powershell
python -m app.rag.ingest
```
Start the backend and UI in separate terminals:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
streamlit run ui/streamlit_app.py --server.port 8501
```

---

## API Endpoints (`app/main.py`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Returns service status, `llm_mode`, and `max_steps`. |
| `POST` | `/chat` | Runs the agent on `{"session_id": "...", "question": "..."}` with `slowapi` rate limiting (`20/minute`) and `diskcache` first-turn caching. |
| `GET` | `/sessions/{session_id}/history` | Returns up to 50 stored conversation turns from `data/sessions.db`. |

---

## Evaluation & Benchmark Results (`eval/`)

Evaluated against the frozen **68-query** benchmark (`eval/eval_set.jsonl`):

| Strategy | Accuracy | Completion Rate | Avg Tool Calls | Avg Latency | Failure Taxonomy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`structured` (Multi-Agent)** | **88.2%** | **95.6%** | **1.15** | **3.52 s** | 4 reasoning, 2 retrieval, 2 hallucination, **0 tool errors** |
| `cot` | 69.1% | 94.1% | 1.49 | 6.60 s | 15 reasoning, 3 retrieval, 2 hallucination, 1 tool error |
| `decomposition` | 63.2% | 95.6% | 1.62 | 6.10 s | 21 reasoning, 2 retrieval, 2 hallucination, 0 tool errors |

Run evaluations and tests from inside `mf_copilot/`:
```powershell
# Run full 68-query evaluation
python -m eval.run_eval

# Compare structured vs cot vs decomposition prompts
python -m eval.compare_prompts

# Evaluate retrieval recall
python -m eval.retrieval_eval

# Run unit tests
pytest
```
