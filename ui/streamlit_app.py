import os
import uuid

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="MF Factsheet Copilot", page_icon="📊")
st.title("📊 Mutual Fund Factsheet & SID Copilot")
st.caption("Facts from scheme documents only. No investment advice or predictions.")

st.session_state.setdefault("sid", str(uuid.uuid4())[:8])
st.session_state.setdefault("msgs", [])

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for c in m.get("citations", []):
            st.caption(f"📎 {c['doc_id']} / {c.get('section') or '-'}")
        if m.get("flag"):
            st.warning("Needs analyst verification")

if q := st.chat_input("Ask about exit load, expense ratio, benchmark, risk factors..."):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    try:
        r = httpx.post(f"{API}/chat", json={"session_id": st.session_state.sid, "question": q}, timeout=90)
        r.raise_for_status()
        d = r.json()
        msg = {"role": "assistant", "content": d["answer"], "citations": d["citations"],
               "flag": d["status"] == "needs_verification"}
    except httpx.HTTPError as e:
        msg = {"role": "assistant", "content": f"Request failed: {e}"}
    st.session_state.msgs.append(msg)
    st.rerun()
