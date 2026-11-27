"""Client layer for the Streamlit UI: supports both FastAPI HTTP backend and Embedded Direct Agent mode."""
from __future__ import annotations
import os
import re
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import httpx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.schemas import ChatResponse, Citation

# Sanitize host.docker.internal when running natively on host OS outside containers
otel_ep = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
if "host.docker.internal" in otel_ep:
    import socket
    try:
        socket.gethostbyname("host.docker.internal")
    except socket.gaierror:
        try:
            with socket.create_connection(("127.0.0.1", 3000), timeout=0.2):
                os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_ep.replace("host.docker.internal", "localhost")
        except Exception:
            os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_LANGFUSE_URL = os.getenv("LANGFUSE_URL", "http://localhost:3000").rstrip("/")
DEFAULT_LANGFUSE_PROJECT_ID = os.getenv("LANGFUSE_PROJECT_ID", "cmwqq4lkt0009mt3uub7ab39y")


def check_api_health(api_url: str = DEFAULT_API_URL, timeout: float = 2.5) -> dict[str, Any]:
    """Check if the FastAPI backend is running and healthy."""
    try:
        t0 = time.time()
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{api_url}/health")
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                data = r.json()
                data["online"] = True
                data["latency_ms"] = latency_ms
                return data
            return {"online": False, "status": f"HTTP {r.status_code}", "latency_ms": latency_ms}
    except Exception as e:
        return {"online": False, "status": "offline", "error": str(e), "latency_ms": None}


def send_chat_api(question: str, session_id: str, api_url: str = DEFAULT_API_URL, timeout: float = 60.0) -> dict[str, Any]:
    """Send question to FastAPI /chat endpoint."""
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{api_url}/chat", json={"session_id": session_id, "question": question})
        r.raise_for_status()
        return r.json()


def get_api_history(session_id: str, api_url: str = DEFAULT_API_URL, timeout: float = 5.0) -> list[dict[str, str]]:
    """Retrieve chat history from FastAPI /sessions/{session_id}/history."""
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{api_url}/sessions/{session_id}/history")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return []


def send_chat_direct(question: str, session_id: str, strategy: str | None = None) -> dict[str, Any]:
    """Execute agent directly in-process using app.agent.graph.run_agent."""
    from app.agent import state
    from app.agent.graph import run_agent

    res = run_agent(question=question, session_id=session_id, strategy=strategy)
    state.add_turn(session_id, "user", question)
    state.add_turn(session_id, "assistant", res.answer)

    # Format into response dict compatible with ChatResponse
    citations = [
        c.model_dump() if hasattr(c, "model_dump")
        else (asdict(c) if is_dataclass(c) else dict(c))
        for c in res.citations
    ]
    observations = [o.to_dict() for o in res.observations]

    return {
        "answer": res.answer,
        "status": res.status,
        "citations": citations,
        "observations": observations,
        "steps": res.steps,
        "tool_calls": res.tool_calls,
        "tokens": res.tokens,
        "latency_ms": res.latency_ms,
        "cached": False,
        "notes": res.notes,
        "unsupported": res.unsupported,
        "trace_id": res.trace_id,
    }


def list_saved_sessions(limit: int = 50) -> list[str]:
    """List session IDs from local sessions.db."""
    db_file = settings.session_db
    if not db_file.exists():
        return []
    try:
        with closing(sqlite3.connect(db_file)) as conn:
            rows = conn.execute(
                "SELECT session_id, MAX(ts) as last_ts FROM turns GROUP BY session_id ORDER BY last_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def list_session_summaries(limit: int = 25) -> list[dict[str, Any]]:
    """List sessions with their first question title, last active timestamp, and message count."""
    db_file = settings.session_db
    if not db_file.exists():
        return []
    try:
        with closing(sqlite3.connect(db_file)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT session_id, MAX(ts) as last_ts, COUNT(*) as turn_count
                FROM turns
                GROUP BY session_id
                ORDER BY last_ts DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            summaries = []
            for r in rows:
                sid = r["session_id"]
                first_turn = conn.execute(
                    "SELECT content FROM turns WHERE session_id = ? AND role = 'user' ORDER BY id ASC LIMIT 1",
                    (sid,),
                ).fetchone()

                title = first_turn["content"] if first_turn else sid
                if len(title) > 36:
                    title = title[:34] + "..."

                summaries.append({
                    "session_id": sid,
                    "title": title,
                    "last_ts": r["last_ts"],
                    "turn_count": r["turn_count"],
                })
            return summaries
    except Exception:
        return []


def delete_session(session_id: str) -> bool:
    """Delete all turns associated with a session ID."""
    db_file = settings.session_db
    if not db_file.exists():
        return False
    try:
        with closing(sqlite3.connect(db_file)) as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.commit()
            return True
    except Exception:
        return False


def get_local_history(session_id: str, n: int = 50) -> list[dict[str, str]]:
    """Get history directly from local SQLite state database."""
    from app.agent import state
    try:
        return state.history(session_id, n=n)
    except Exception:
        return []


def build_langfuse_trace_url(trace_id: str | None, base_url: str = DEFAULT_LANGFUSE_URL,
                            project_id: str = DEFAULT_LANGFUSE_PROJECT_ID) -> str | None:
    """Construct deep-link to Langfuse trace if trace_id and configuration are present."""
    if not trace_id:
        return None
    return f"{base_url}/project/{project_id}/traces/{trace_id}"


KNOWN_FUND_COMPANIES: list[str] = [
    "Axis Mutual Fund",
    "Bandhan Mutual Fund",
    "Bank of India Mutual Fund",
    "Baroda BNP Paribas Mutual Fund",
    "Baroda Pioneer Mutual Fund",
    "Birla Sun Life Mutual Fund",
    "BNP Paribas Mutual Fund",
    "Canara Robeco Mutual Fund",
    "Daiwa Mutual Fund",
    "DHFL Pramerica Mutual Fund",
    "DSP Mutual Fund",
    "DWS Mutual Fund",
    "ICICI Prudential Mutual Fund",
    "JPMorgan Mutual Fund",
    "Nippon India Mutual Fund",
    "Tata Mutual Fund",
]


def get_available_fund_companies(data_dir: Path | None = None) -> list[str]:
    """Return sorted list of mutual fund companies for which documents exist in data folder."""
    target_dir = Path(data_dir) if data_dir else (settings.data_dir / "docs")
    if not target_dir.exists():
        alt_dir = Path(__file__).resolve().parent.parent / "data" / "docs"
        if alt_dir.exists():
            target_dir = alt_dir
        else:
            return KNOWN_FUND_COMPANIES

    found: set[str] = set()
    try:
        for f in target_dir.glob("*.pdf"):
            fl = f.name.lower()
            for amc in KNOWN_FUND_COMPANIES:
                root_words = amc.replace(" Mutual Fund", "").lower().split()
                if all(w in fl for w in root_words) or (amc == "Bank of India Mutual Fund" and "boi" in fl):
                    found.add(amc)
                    break
    except Exception:
        pass

    return sorted(list(found if found else KNOWN_FUND_COMPANIES))


_AMC_SCHEMES_CACHE: dict[str, list[str]] | None = None


def _load_embedded_amc_schemes() -> dict[str, list[str]]:
    """Load pre-indexed AMC-to-schemes dictionary from amc_schemes.json."""
    global _AMC_SCHEMES_CACHE
    if _AMC_SCHEMES_CACHE is not None:
        return _AMC_SCHEMES_CACHE

    json_path = Path(__file__).resolve().parent / "amc_schemes.json"
    if json_path.exists():
        try:
            import json
            _AMC_SCHEMES_CACHE = json.loads(json_path.read_text(encoding="utf-8"))
            return _AMC_SCHEMES_CACHE
        except Exception:
            pass
    _AMC_SCHEMES_CACHE = {}
    return _AMC_SCHEMES_CACHE


def get_schemes_for_fund_company(company: str | None = None) -> list[str]:
    """Return sorted list of mutual fund schemes for a given fund company / AMC."""
    if not company or company == "Choose Mutual Fund":
        return []

    # 1. Start with pre-indexed schemes from amc_schemes.json if present
    embedded_map = _load_embedded_amc_schemes()
    schemes_set: set[str] = set(embedded_map.get(company, []))

    root = company.replace(" Mutual Fund", "").replace(" MF", "").strip().lower()
    root_words = root.split()

    # 2. Query chunks.db if available to include any dynamically ingested schemes
    dbp = settings.index_dir / "chunks.db"
    if not dbp.exists():
        dbp = Path(__file__).resolve().parent.parent / "data" / "index" / "chunks.db"
    if dbp.exists():
        try:
            with closing(sqlite3.connect(dbp)) as conn:
                for r in conn.execute("SELECT DISTINCT scheme FROM chunks WHERE scheme IS NOT NULL"):
                    if r[0] and r[0].strip():
                        s = r[0].strip()
                        sl = s.lower()
                        if "bank of india" in root or "boi" in root:
                            if "bank of india" in sl or "boi" in sl:
                                schemes_set.add(s)
                        elif all(w in sl for w in root_words):
                            schemes_set.add(s)
        except Exception:
            pass

    # 3. Inspect document filenames in data/docs to catch any newly dropped PDFs
    docs_dir = settings.docs_dir
    if not docs_dir.exists():
        docs_dir = Path(__file__).resolve().parent.parent / "data" / "docs"
    if docs_dir.exists():
        try:
            for p in docs_dir.glob("*.pdf"):
                bn = p.name
                clean = re.sub(r"_\d+\.pdf$", "", bn, flags=re.I).strip()
                if clean:
                    sl = clean.lower()
                    if "bank of india" in root or "boi" in root:
                        if "bank of india" in sl or "boi" in sl:
                            schemes_set.add(clean)
                    elif all(w in sl for w in root_words):
                        schemes_set.add(clean)
        except Exception:
            pass

    return sorted(list(schemes_set))
