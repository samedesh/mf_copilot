"""Mutual Fund SID Q&A Copilot — Analytical Research UI (Streamlit)."""
from __future__ import annotations
import html
import json
import os
import uuid
from pathlib import Path

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
META_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "meta.json"

st.set_page_config(
    page_title="MF SID Q&A Copilot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle styling for copilot trust badges, telemetry strip, and evidence cards
st.markdown(
    """
    <style>
      .badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; align-items: center; }
      .badge { font-size: 0.78rem; font-weight: 600; padding: 3px 9px; border-radius: 999px; display: inline-block; }
      .b-ok { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
      .b-warn { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
      .b-refuse { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
      .b-err { background: #fce7f3; color: #9d174d; border: 1px solid #f9a8d4; }
      .b-cache { background: #e0f2fe; color: #075985; border: 1px solid #7dd3fc; }
      .telemetry { font-size: 0.8rem; color: #475569; margin-top: 6px; padding-top: 6px; border-top: 1px solid #e2e8f0; }
      .cite-card { background: #f8fafc; border-left: 3px solid #2563eb; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px; }
      .cite-hdr { font-size: 0.82rem; font-weight: 700; color: #1e3a8a; margin-bottom: 4px; }
      .cite-snip { font-size: 0.83rem; color: #334155; white-space: pre-wrap; }
      .q-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #2563eb; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fetch_health() -> dict | None:
    try:
        r = httpx.get(f"{API}/health", timeout=2.5)
        if r.status_code == 200:
            return r.json()
    except httpx.HTTPError:
        pass
    return None


def _load_index_meta() -> dict | None:
    try:
        if META_PATH.exists():
            return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _status_badges(turn: dict) -> str:
    status = turn.get("status", "answered")
    pills = []
    if status == "answered":
        pills.append('<span class="badge b-ok">✓ Verified by Critic</span>')
    elif status == "needs_verification":
        pills.append('<span class="badge b-warn">⚠ Needs Analyst Verification</span>')
    elif status == "refused":
        pills.append('<span class="badge b-refuse">🛡 Blocked by Guardrail</span>')
    else:
        pills.append('<span class="badge b-err">✕ Endpoint Error</span>')

    if turn.get("cached"):
        pills.append('<span class="badge b-cache">⚡ Cached (0 ms)</span>')
    return f'<div class="badge-row">{"".join(pills)}</div>'


def _ask_api(question: str) -> None:
    try:
        with st.spinner("Retrieving SID clauses, executing tools & verifying with critic..."):
            r = httpx.post(
                f"{API}/chat",
                json={"session_id": st.session_state.sid, "question": question},
                timeout=120,
            )
            r.raise_for_status()
            d = r.json()
            turn = {
                "question": question,
                "answer": d["answer"],
                "status": d.get("status", "answered"),
                "citations": d.get("citations", []),
                "steps": d.get("steps", 0),
                "tool_calls": d.get("tool_calls", 0),
                "tokens": d.get("tokens", 0),
                "latency_ms": d.get("latency_ms", 0),
                "cached": d.get("cached", False),
                "notes": d.get("notes", []),
            }
    except httpx.HTTPError as e:
        turn = {
            "question": question,
            "answer": f"API request failed: `{e}`",
            "status": "error",
            "citations": [],
            "steps": 0,
            "tool_calls": 0,
            "tokens": 0,
            "latency_ms": 0,
            "cached": False,
            "notes": [str(e)],
        }
    st.session_state.turns.append(turn)


st.session_state.setdefault("sid", str(uuid.uuid4())[:8])
st.session_state.setdefault("turns", [])
st.session_state.setdefault("pending_q", None)

# ── Sidebar: System Status, Corpus Info, One-Click Demo Queries & Session Controls ──
with st.sidebar:
    st.header("📄 SID Research Copilot")
    st.caption("Evidence-grounded Q&A over SEBI Scheme Information Documents (SIDs).")

    health = _fetch_health()
    if health:
        st.success(f"API Online · mode=`{health.get('llm_mode')}` · cap=`{health.get('max_steps')}` steps")
    else:
        st.error(f"API Unreachable at `{API}`")

    meta = _load_index_meta()
    if meta:
        c1, c2 = st.columns(2)
        c1.metric("Indexed Docs", f"{meta.get('docs', 0):,}")
        c2.metric("SID Chunks", f"{meta.get('n', 0):,}")

    st.divider()
    st.subheader("Quick Sample Queries")
    SAMPLES = [
        ("Factual · Exit Load", "What is the exit load of Axis Credit Risk Fund?"),
        ("Qualitative · Objective", "What is the investment objective of Nippon India ELSS Tax Saver Fund?"),
        ("Calculator · Expense Drag", "How much does a 1.5% expense ratio cost on Rs 5 lakh over 10 years at 10% gross return?"),
        ("Guardrail · Advice Refusal", "Which mutual fund should I buy to double my money?"),
    ]
    for label, sample_q in SAMPLES:
        if st.button(label, use_container_width=True, help=sample_q):
            st.session_state.pending_q = sample_q

    st.divider()
    st.caption(f"Active Session ID: `{st.session_state.sid}`")
    if st.button("🔄 New Research Session", use_container_width=True):
        st.session_state.sid = str(uuid.uuid4())[:8]
        st.session_state.turns = []
        st.session_state.pending_q = None
        st.rerun()

    st.divider()
    st.caption(
        "**Compliance Notice:** Answers are strictly grounded in indexed SIDs. "
        "This tool does not provide SEBI investment advice or return forecasts."
    )

# ── Main Analytical Q&A Workspace ──
st.title("📄 Mutual Fund Scheme Information Document (SID) Q&A Copilot")
st.caption(
    "Ask factual, qualitative, comparative, or calculation questions about indexed SIDs. "
    "Follow-up turns retain session context automatically."
)

if not st.session_state.turns and not st.session_state.pending_q:
    st.info(
        "Enter a question below or pick a **Quick Sample Query** from the sidebar to inspect "
        "grounded SID citations, deterministic tool calculations, critic verification, and guardrails."
    )

for idx, turn in enumerate(st.session_state.turns, 1):
    with st.container(border=True):
        st.markdown(f'<div class="q-label">Query #{idx}</div>', unsafe_allow_html=True)
        st.markdown(f"**{turn['question']}**")
        st.markdown(_status_badges(turn), unsafe_allow_html=True)
        st.markdown(turn["answer"])

        # Compact Agent Telemetry Strip
        lat_s = turn["latency_ms"] / 1000.0
        st.markdown(
            f'<div class="telemetry">'
            f'⏱ <strong>Latency:</strong> {lat_s:.2f}s &nbsp;|&nbsp; '
            f'🔧 <strong>Tool Calls:</strong> {turn["tool_calls"]} &nbsp;|&nbsp; '
            f'🪜 <strong>Steps:</strong> {turn["steps"]} &nbsp;|&nbsp; '
            f'🪙 <strong>Tokens:</strong> {turn["tokens"]:,}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Expandable Retrieved Evidence & Citations Drawer
        cites = turn.get("citations") or []
        if cites:
            with st.expander(f"📚 Sources & Retrieved SID Evidence ({len(cites)})", expanded=False):
                for c in cites:
                    doc_id = html.escape(str(c.get("doc_id", "")))
                    sec = html.escape(str(c.get("section") or "-"))
                    snip = html.escape(str(c.get("snippet") or ""))
                    st.markdown(
                        f'<div class="cite-card">'
                        f'<div class="cite-hdr">📎 {doc_id} &nbsp;·&nbsp; {sec}</div>'
                        f'<div class="cite-snip">{snip}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Expandable Critic & Guardrail Audit Notes
        notes = turn.get("notes") or []
        if notes:
            auto_open = turn.get("status") == "needs_verification"
            with st.expander(f"🔍 Guardrail & Critic Audit Trail ({len(notes)})", expanded=auto_open):
                for n in notes:
                    st.markdown(f"- `{n}`")

# Process either a clicked sidebar sample query or typed query
typed_q = st.chat_input(
    "Ask about investment objective, asset allocation, exit load, benchmark, risk factors, or expense cost..."
)
next_q = st.session_state.pending_q or typed_q
if next_q:
    st.session_state.pending_q = None
    _ask_api(next_q)
    st.rerun()
