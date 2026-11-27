"""Main entrypoint for the Mutual Fund Scheme Information Document (SID) Q&A Copilot Streamlit Application."""
from __future__ import annotations
import json
import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from ui.api_client import (
    DEFAULT_API_URL,
    DEFAULT_LANGFUSE_PROJECT_ID,
    DEFAULT_LANGFUSE_URL,
    build_langfuse_trace_url,
    check_api_health,
    get_api_history,
    get_available_fund_companies,
    get_schemes_for_fund_company,
    get_local_history,
    send_chat_api,
    send_chat_direct,
)
from ui.components.chat import display_chat_message
from ui.components.sidebar import render_sidebar
from ui.components.taxonomy_modal import render_taxonomy_button

# Page Configuration
st.set_page_config(
    page_title="Mutual Fund SID Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_session_state():
    """Initialize necessary session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sid-{uuid.uuid4().hex[:8]}"
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "exec_mode" not in st.session_state:
        st.session_state.exec_mode = "FastAPI Backend"
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "strategy" not in st.session_state:
        st.session_state.strategy = "structured"
    if "langfuse_url" not in st.session_state:
        st.session_state.langfuse_url = DEFAULT_LANGFUSE_URL
    if "langfuse_project_id" not in st.session_state:
        st.session_state.langfuse_project_id = DEFAULT_LANGFUSE_PROJECT_ID
    if "latest_trace_id" not in st.session_state:
        st.session_state.latest_trace_id = None


def handle_prompt_select(prompt_text: str):
    """Callback when a sample prompt is clicked in the sidebar."""
    st.session_state.pending_prompt = prompt_text


def load_history_if_needed(session_id: str, exec_mode: str, api_url: str):
    """Load previous conversation history from database or API if local messages list is empty."""
    if not st.session_state.messages:
        turns = []
        if exec_mode == "FastAPI Backend":
            turns = get_api_history(session_id, api_url=api_url)
        if not turns:
            turns = get_local_history(session_id)
        if turns:
            st.session_state.messages = [
                {"role": t["role"], "content": t["content"], "meta": None}
                for t in turns
            ]


def main():
    init_session_state()

    def handle_new_chat():
        st.session_state.session_id = f"sid-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.latest_trace_id = None
        st.rerun()

    # Sidebar: New Chat button above Example Questions & Regulatory Notice
    render_sidebar(on_select_prompt=handle_prompt_select, on_new_chat=handle_new_chat)

    # Top Fixed Header: Title on Left, Failure Taxonomy & Observability on Right
    with st.container(key="top_fixed_header"):
        header_col1, header_col2 = st.columns([6.8, 5.2])
        with header_col1:
            st.markdown(
                """
                <div style="margin-bottom: 0.2rem;">
                    <h1 style="margin-bottom: 0.2rem; font-size: 1.85rem; line-height: 1.2;">📈 Mutual Fund SID Q&A Copilot</h1>
                    <p style="font-size: 0.92rem; opacity: 0.8; margin-top: 0; margin-bottom: 0.2rem;">
                        SEBI-Compliant Fact Retrieval, Factsheet Metrics & Grounded Reasoning across 500+ Scheme Information Documents.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Session: `{st.session_state.session_id}`")

        with header_col2:
            tax_col, obs_col = st.columns([1, 1], gap="small")
            with tax_col:
                render_taxonomy_button(
                    langfuse_url=st.session_state.langfuse_url,
                    langfuse_project_id=st.session_state.langfuse_project_id,
                )

            with obs_col:
                # Observability and Tracing Button (popover modal on top right)
                with st.popover("📊 Observability & Tracing", use_container_width=True):
                    st.markdown("### 📊 Observability & Tracing")
                    st.caption("Inspect live OpenTelemetry trace spans, Langfuse telemetry, and runtime configuration.")

                    # Langfuse Deep-Link
                    st.markdown(f"**Langfuse Web UI:** [🔗 Open Langfuse Dashboard]({st.session_state.langfuse_url})")

                    if st.session_state.latest_trace_id:
                        st.markdown(f"**Latest Trace ID:** `{st.session_state.latest_trace_id}`")
                        trace_url = build_langfuse_trace_url(
                            st.session_state.latest_trace_id,
                            base_url=st.session_state.langfuse_url,
                            project_id=st.session_state.langfuse_project_id,
                        )
                        if trace_url:
                            st.markdown(f"[🔎 View Latest Trace in Langfuse]({trace_url})")
                    else:
                        st.info("ℹ️ Ask a question in chat to generate an execution trace.")

                    st.divider()

                    # Telemetry Settings
                    st.markdown("#### 🧭 Langfuse Settings")
                    st.session_state.langfuse_url = st.text_input("Langfuse Base URL", value=st.session_state.langfuse_url)
                    st.session_state.langfuse_project_id = st.text_input("Langfuse Project ID", value=st.session_state.langfuse_project_id)

                    # Export Chat
                    if st.session_state.messages:
                        st.divider()
                        chat_json = json.dumps(st.session_state.messages, indent=2, default=str)
                        st.download_button(
                            "📥 Export Chat History (JSON)",
                            data=chat_json,
                            file_name=f"chat_{st.session_state.session_id}.json",
                            mime="application/json",
                            use_container_width=True,
                        )

    # Chat Interface
    load_history_if_needed(st.session_state.session_id, st.session_state.exec_mode, st.session_state.api_url)

    # Show initial guidance if no chat history
    if not st.session_state.messages:
        st.info(
            """
            👋 **Welcome to the Mutual Fund SID Copilot!**
            
            You can ask factual questions regarding mutual fund schemes in India:
            - **Scheme Information Document (SID) Facts:** Investment objectives, exit loads, benchmark indices, asset allocation, recurring expense limits, riskometer, and minimum investment amounts.
            - **Factsheet Metrics:** Verified AUM, NAV, and Direct vs. Regular expense ratios.
            - **Deterministic Financial Calculations:** SIP future values, expense ratio drag, CAGR, and metric differences.
            
            *⚠️ Per SEBI regulations, this copilot strictly refuses investment advice, buy/sell recommendations, and return predictions.*
            """
        )

    # Display Message History
    for msg in st.session_state.messages:
        display_chat_message(
            role=msg["role"],
            content=msg["content"],
            meta=msg.get("meta"),
            langfuse_url=st.session_state.langfuse_url,
            langfuse_project_id=st.session_state.langfuse_project_id,
        )

    # Callback to synchronize scheme dropdown when AMC selection changes
    def on_company_dropdown_change():
        st.session_state.selected_scheme_dropdown = "Choose Scheme"

    # Fixed Bottom Area: Unified box holding Dropdowns Row + Question Input Box
    with st.bottom:
        with st.container(key="bottom_pinned_box"):
            col_fund, col_scheme = st.columns([1, 1], gap="small")
            with col_fund:
                available_funds = get_available_fund_companies()
                company_options = ["Choose Mutual Fund"] + available_funds
                curr_company = st.session_state.get("selected_mf_dropdown", "Choose Mutual Fund")
                company_idx = company_options.index(curr_company) if curr_company in company_options else 0

                selected_company = st.selectbox(
                    "Choose Mutual Fund",
                    options=company_options,
                    index=company_idx,
                    label_visibility="collapsed",
                    key="selected_mf_dropdown",
                    on_change=on_company_dropdown_change,
                    help="Filter queries exclusively to this mutual fund company",
                )

            with col_scheme:
                # Dynamically retrieve schemes for selected AMC
                if selected_company and selected_company != "Choose Mutual Fund":
                    available_schemes = get_schemes_for_fund_company(selected_company)
                else:
                    available_schemes = []

                scheme_placeholder = "Choose Scheme" if available_schemes else "Choose Scheme (Select AMC first)"
                scheme_options = [scheme_placeholder] + available_schemes

                curr_scheme = st.session_state.get("selected_scheme_dropdown")
                scheme_idx = scheme_options.index(curr_scheme) if curr_scheme in scheme_options else 0

                is_scheme_disabled = (selected_company == "Choose Mutual Fund" or not bool(available_schemes))

                selected_scheme = st.selectbox(
                    "Choose Scheme",
                    options=scheme_options,
                    index=scheme_idx,
                    label_visibility="collapsed",
                    key="selected_scheme_dropdown",
                    disabled=is_scheme_disabled,
                    help="Filter queries specifically to this mutual fund scheme",
                )

            chat_query = st.chat_input("Ask a question about a mutual fund Scheme Information Document...")

    # Determine if we have an input from chat box or sidebar prompt click
    query_to_run = None
    if st.session_state.pending_prompt:
        query_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    elif chat_query:
        query_to_run = chat_query

    if query_to_run:
        # Determine scoping: specific scheme takes priority over company
        if selected_scheme and selected_scheme not in ("Choose Scheme", "Choose Scheme (Select AMC first)"):
            if selected_scheme.lower() not in query_to_run.lower():
                prompt_for_agent = f"For {selected_scheme}: {query_to_run}"
            else:
                prompt_for_agent = query_to_run
            display_question = f"**[{selected_scheme}]** {query_to_run}"
            meta_tag = selected_scheme
        elif selected_company and selected_company != "Choose Mutual Fund":
            clean_company = (selected_company.replace(" Mutual Fund", "").replace(" MF", "")).strip()
            if clean_company.lower() not in query_to_run.lower():
                prompt_for_agent = f"For {selected_company}: {query_to_run}"
            else:
                prompt_for_agent = query_to_run
            display_question = f"**[{selected_company}]** {query_to_run}"
            meta_tag = selected_company
        else:
            prompt_for_agent = query_to_run
            display_question = query_to_run
            meta_tag = None

        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": display_question,
            "meta": {"company": meta_tag},
        })
        display_chat_message("user", display_question)

        # Execute Copilot
        with st.spinner("Analyzing Scheme Information Documents & calculating..."):
            try:
                if st.session_state.exec_mode == "FastAPI Backend":
                    try:
                        resp = send_chat_api(question=prompt_for_agent, session_id=st.session_state.session_id, api_url=st.session_state.api_url)
                    except Exception as e:
                        if "connect" in str(e).lower() or "refused" in str(e).lower():
                            resp = send_chat_direct(question=prompt_for_agent, session_id=st.session_state.session_id, strategy=st.session_state.strategy)
                        else:
                            raise
                else:
                    resp = send_chat_direct(question=prompt_for_agent, session_id=st.session_state.session_id, strategy=st.session_state.strategy)

                assistant_text = resp.get("answer", "No answer received.")
                trace_id = resp.get("trace_id")
                if trace_id:
                    st.session_state.latest_trace_id = trace_id

                meta = {
                    "status": resp.get("status", "answered"),
                    "citations": resp.get("citations", []),
                    "steps": resp.get("steps", 0),
                    "tool_calls": resp.get("tool_calls", 0),
                    "tokens": resp.get("tokens", 0),
                    "latency_ms": resp.get("latency_ms", 0),
                    "cached": resp.get("cached", False),
                    "notes": resp.get("notes", []),
                    "unsupported": resp.get("unsupported", []),
                    "trace_id": trace_id,
                    "observations": resp.get("observations", []),
                }

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_text,
                    "meta": meta,
                })
            except Exception as e:
                err_msg = f"Failed to execute query: {e}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {err_msg}",
                    "meta": {"status": "error", "latency_ms": 0},
                })

        # Rerun so message history renders above and the chat input is fresh and ready for the next question
        st.rerun()


if __name__ == "__main__":
    main()
