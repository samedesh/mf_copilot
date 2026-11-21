# Post-Implementation Execution & Capstone Checklist (`mf_copilot` Target-500 SID)

Follow this chronological checklist after Phase 1 upgrades from `project-docs/target-500.md` have been applied.

---

## 1. Install Updated Dependencies (~1 min)
From the `mf_copilot/` directory (with your `.venv` active):
```powershell
pip install -r requirements.txt
```
*(Ensures `pypdf`, `python-dotenv`, and `opentelemetry-exporter-otlp-proto-http` are installed.)*

---

## 2. Run Bulk Ingestion Across All 544 SID PDFs
From `mf_copilot/`:
```powershell
python -m app.rag.ingest
```
* Monitor progress output (`[100/544] ...`).
* Because of incremental checkpointing (`data/index/manifest.json`), if your network drops or you pause the script, simply re-run the same command to resume cleanly without losing completed work.

---

## 3. Verify & Smoke-Test Locally
1. **Run unit tests:**
   ```powershell
   pytest -q
   ```
2. **Launch FastAPI & Streamlit UI (in two terminals):**
   ```powershell
   uvicorn app.main:app --reload
   ```
   ```powershell
   streamlit run ui/streamlit_app.py
   ```
3. **Spot-check 5–10 live queries in Streamlit** across multiple AMCs (Axis, Kotak, HDFC, SBI, ICICI, Nippon):
   * **Factual SID queries:** exit load, benchmark, minimum SIP/application amount, lock-in period.
   * **Qualitative SID queries:** investment objective, asset allocation, specific risk factors, taxation.
   * **Multi-turn follow-ups:** confirm SQLite session memory carries context across turns.
   * **Out-of-scope / advice queries:** e.g. *"Which fund should I buy?"* $\rightarrow$ must return `refused`.

---

## 4. Connect Observability & Tracing Dashboard (~10 mins)
* **Option A (Recommended — Zero Docker):** Sign up for the free tier at `https://cloud.langfuse.com` (50,000 observations/month free).
* **Option B (Local Docker):** Run Langfuse locally via Docker Compose (`http://localhost:3000`).
* **Option C (Local Python Only):** Run Arize Phoenix (`pip install arize-phoenix` $\rightarrow$ `phoenix serve` at `http://localhost:6006`) or inspect `logs/spans.jsonl`.
* Add the two OTLP variables to `mf_copilot/.env`:
  ```env
  OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
  OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64(PUBLIC_KEY:SECRET_KEY)>
  ```
* Send a few test queries and capture **trace waterfall screenshots** (`agent.request` -> `planner` -> `executor` -> `tool.*` -> `composer` -> `critic`) showing latencies, token usage, and replans for your Capstone report.

---

## 5. Run Capstone Track B Evaluation & Experiments
Run all three evaluation scripts from `mf_copilot/` against the frozen gold `eval/eval_set.jsonl`:

1. **Retrieval Quality (BM25 vs. Dense vs. Hybrid — `recall@1/3/5` & `MRR`):**
   ```powershell
   python -m eval.retrieval_eval
   ```
2. **Prompting Strategy Comparison (`structured` vs. `cot` vs. `decomposition`):**
   ```powershell
   python -m eval.compare_prompts
   ```
3. **End-to-End Agent Evaluation (Accuracy, Completion Rate, Tool Calls, Latency, Tokens, Failure Taxonomy):**
   ```powershell
   python -m eval.run_eval
   ```
4. **Optional Report Ablation Experiments:**
   * Compare a second chunk size (e.g., `--chunk-size 150` vs `250`) on retrieval metrics.
   * Test enabling `RERANKER=BAAI/bge-reranker-base` (requires `pip install sentence-transformers`) to measure before/after retrieval gains.
   * Verify prompt-injection resistance using `sample_sid_large_cap.txt` and injection rows in `eval_set.jsonl`.

---

## 6. Final Submission & Demo Preparation
* *(Optional)* Verify containerized deployment if required for your demo:
  ```powershell
  docker compose up --build
  ```
  *(API on `http://localhost:8000/docs`, Streamlit UI on `http://localhost:8501`.)*
* Assemble your Capstone report & presentation using:
  * System & scaling architecture (`project-docs/target-500.md`)
  * Retrieval benchmarks (`recall@k`, `MRR`), prompt-strategy comparison, and failure taxonomy (`eval/results.json`)
  * Guardrail & prompt-injection verification results
  * Langfuse / OpenTelemetry trace screenshots
  * Honest limitations (`project-docs/README.md`)
