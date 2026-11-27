# Mutual Fund SID Copilot — Streamlit UI

Interactive web application for the **Mutual Fund Scheme Information Document (SID) Q&A Copilot**.

## Features & UI Layout

- **Fixed Top Header & Action Toolbar**:
  - **Fixed Header**: Pinned "Mutual Fund SID Q&A Copilot" section at the top of the page with active session tracking.
  - **`📋 Failure Taxonomy` Button**: Dedicated benchmark & diagnostic popover modal on the top-right toolbar:
    - **KPI Metric Tiles**: Overall accuracy (88.2%), completion rate (95.6%), latency, and benchmark query count.
    - **4-Way Failure Taxonomy Breakdown**: Reasoning errors (4), retrieval misses (2), hallucinations (2), and tool-call errors (0).
    - **Strategy Comparison Table**: Prompt engineering baselines (CoT: 8.8%, Decomposition: 60.3%, Structured: 67.6%) vs. Multi-Agent (88.2%).
    - **Case-Study Drill-Down**: Interactive breakdown of all 8 failed queries with gold answers, diagnoses, remedies, and Langfuse trace links.
  - **`📊 Observability & Tracing` Button**: One-click popover modal on the top-right of the main page:
    - Direct link to open the **Langfuse Dashboard** (`http://localhost:3000`).
    - Inspect the latest query's **OpenTelemetry Trace ID** with deep-link navigation.
    - Configure Langfuse connection settings (`LANGFUSE_URL`, `LANGFUSE_PROJECT_ID`).
    - Chat history export button (JSON).

- **Unified Pinned Bottom Search & Scoping Box**:
  - Pinned directly to the bottom of the page across all viewports.
  - **Dynamic Cascading Dropdowns**: First dropdown selects the Mutual Fund Company (AMC), which dynamically filters and populates the second dropdown with schemes specifically belonging to that AMC (e.g. Axis Mutual Fund -> all 22 Axis schemes).
  - **Full-Width Question Input**: Sits directly underneath the dropdowns in the same unified floating card box.

- **Clean Sidebar**:
  - **`➕ New Chat` Button**: Pinned at the top of the sidebar for instant 1-click session resets and fresh conversation starts.
  - **💡 Example Questions Library**: Grouped by SID Facts, Calculations, Guardrails & Refusal Tests, and Multi-turn Memory. Clicking any question auto-populates and runs it immediately.
  - **🔒 SEBI Regulatory Compliance**: Notice explaining non-advice factual scope.

- **💬 Grounded SID Q&A Copilot**:
  - Factual SID Q&A across 500+ Indian mutual fund Scheme Information Documents.
  - Verifiable responses with structured **Citations** (document ID, section tag, and quoted snippets).
  - **SEBI-Compliant Guardrails**: Automatic PII redaction (PAN, Aadhaar, Folio, phone, email) and strict refusal of investment advice or return forecasts.
  - **Status Badges**: Visual indicators for `🟢 Verified Answer`, `🟡 Guardrail Refusal`, `🟠 Needs Verification`, `🔴 Error`, and `⚡ Cache Hit`.
  - **Execution Trace**: Step-by-step tool observation inspector.

## How to Run

### 1. Local Development
```powershell
streamlit run ui/streamlit_app.py
```
*(or via alias: `streamlit run ui/app.py`)*

### 2. With FastAPI Server
Terminal 1:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Terminal 2:
```powershell
streamlit run ui/streamlit_app.py --server.port 8501
```

### 3. Docker Compose
```bash
docker compose up -d --build
```
- Streamlit UI: `http://localhost:8501`
- FastAPI Docs: `http://localhost:8000/docs`
- Langfuse (if using `docker-compose.langfuse.yml`): `http://localhost:3000`
