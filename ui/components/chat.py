"""Chat message renderer, citations viewer, and execution trace inspection."""
from __future__ import annotations
import json
from typing import Any

import streamlit as st

from ui.api_client import build_langfuse_trace_url


def render_status_badge(status: str, cached: bool = False) -> str:
    """Generate HTML pill badge for query response status."""
    labels = {
        "answered": ("🟢 Verified SID Answer", "badge-answered"),
        "refused": ("🟡 Guardrail Refusal (Advice/Safety)", "badge-refused"),
        "needs_verification": ("🟠 Needs Verification", "badge-needs_verification"),
        "error": ("🔴 Execution Error", "badge-error"),
    }
    label, css_cls = labels.get(status, (f"⚪ {status}", "badge-answered"))
    badge_html = f'<span class="badge-status {css_cls}">{label}</span>'
    if cached:
        badge_html += ' <span class="badge-status badge-cached">⚡ Cache Hit</span>'
    return badge_html


def render_citations(citations: list[dict[str, Any]]):
    """Render citations in an organized visual layout."""
    if not citations:
        return

    with st.expander(f"📚 Verified Sources & Citations ({len(citations)})", expanded=True):
        for i, c in enumerate(citations, 1):
            doc_id = c.get("doc_id", "Unknown Document")
            section = c.get("section") or "General"
            snippet = c.get("snippet", "").strip()

            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-header">
                        <span>[{i}] <strong>{doc_id}</strong></span>
                        <span style="opacity: 0.75; font-size: 0.8rem;">§ {section}</span>
                    </div>
                    <div class="citation-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_agent_metadata(meta: dict[str, Any], langfuse_base_url: str = "http://localhost:3000",
                          langfuse_project_id: str = "cmwqq4lkt0009mt3uub7ab39y"):
    """Render latency, tokens, steps, trace ID, and notes."""
    latency_ms = meta.get("latency_ms", 0)
    steps = meta.get("steps", 0)
    tool_calls = meta.get("tool_calls", 0)
    tokens = meta.get("tokens", 0)
    cached = meta.get("cached", False)
    trace_id = meta.get("trace_id")
    notes = meta.get("notes", [])
    observations = meta.get("observations", [])
    unsupported = meta.get("unsupported", [])

    with st.expander("🔍 Observability & Execution Trace", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Latency", f"{latency_ms} ms" if not cached else "0 ms (Cache)")
        with c2:
            st.metric("Total Steps", steps)
        with c3:
            st.metric("Tool Calls", tool_calls)
        with c4:
            st.metric("Tokens", tokens)

        if notes:
            st.markdown("##### 🛡️ Guardrails & Runtime Notes")
            for note in notes:
                if "redacted" in note.lower():
                    st.info(f"🔒 {note}")
                elif "replan" in note.lower() or "flagged" in note.lower():
                    st.warning(f"⚠️ {note}")
                elif "refused" in note.lower():
                    st.error(f"🛑 {note}")
                else:
                    st.write(f"• {note}")

        if unsupported:
            st.warning(f"🚨 Numbers unverified against source documents: {', '.join(unsupported)}")

        if trace_id:
            st.markdown("##### 🧭 OpenTelemetry Trace")
            st.code(trace_id, language="text")
            trace_url = build_langfuse_trace_url(trace_id, base_url=langfuse_base_url, project_id=langfuse_project_id)
            if trace_url:
                st.markdown(f"[🔗 View Full Trace in Langfuse]({trace_url})")

        if observations:
            st.markdown("##### ⚙️ Tool Execution Details")
            for obs in observations:
                step_num = obs.get("step", "?")
                tool_name = obs.get("tool", "unknown")
                ok = obs.get("ok", False)
                status_icon = "✅" if ok else "❌"

                with st.expander(f"Step {step_num}: {status_icon} `{tool_name}`", expanded=False):
                    st.write("**Arguments:**")
                    st.json(obs.get("args", {}))
                    if ok:
                        st.write("**Result:**")
                        st.json(obs.get("result", {}))
                    else:
                        st.error(f"Error: {obs.get('error', 'unknown error')}")


def display_chat_message(role: str, content: str, meta: dict[str, Any] | None = None,
                         langfuse_url: str = "http://localhost:3000",
                         langfuse_project_id: str = "cmwqq4lkt0009mt3uub7ab39y"):
    """Render a single message with role avatar and optional metadata."""
    avatar = "👤" if role == "user" else "📈"
    with st.chat_message(role, avatar=avatar):
        if role == "assistant" and meta:
            status = meta.get("status", "answered")
            cached = meta.get("cached", False)
            st.markdown(render_status_badge(status, cached), unsafe_allow_html=True)

        st.markdown(content)

        if role == "assistant" and meta:
            citations = meta.get("citations", [])
            if citations:
                render_citations(citations)
            render_agent_metadata(meta, langfuse_base_url=langfuse_url, langfuse_project_id=langfuse_project_id)
