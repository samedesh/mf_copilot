"""Document catalog, fund metrics table explorer, and RAG retrieval inspector."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config import settings
from app.rag.retriever import get_retriever
from app.tools.search_docs import run as run_search


def render_explorer():
    """Render documents catalog, verified metrics table, and retrieval tester."""
    st.markdown("## 🔍 SID Document Catalog & Fund Metrics")
    st.caption("Inspect the grounded data corpus: verified factsheet metrics and indexed SEBI Scheme Information Documents.")

    exp_tab1, exp_tab2, exp_tab3 = st.tabs([
        "📊 Verified Fund Metrics Table",
        "📑 SID Corpus & Index Catalog",
        "🎯 Raw Hybrid Search Tester",
    ])

    with exp_tab1:
        st.markdown("### Verified Scheme Metrics (`fund_metrics.csv`)")
        st.caption("Direct, deterministic metric store used by the `get_fund_metric` tool.")

        if not settings.metrics_csv.exists():
            st.warning(f"Metrics CSV not found at `{settings.metrics_csv}`.")
        else:
            try:
                df_metrics = pd.read_csv(settings.metrics_csv)
                c_filter1, c_filter2, c_filter3 = st.columns(3)
                with c_filter1:
                    schemes = ["All"] + sorted(df_metrics["scheme"].dropna().unique().tolist())
                    sel_scheme = st.selectbox("Filter Scheme", schemes)
                with c_filter2:
                    plans = ["All"] + sorted(df_metrics["plan"].dropna().unique().tolist())
                    sel_plan = st.selectbox("Filter Plan", plans)
                with c_filter3:
                    metrics = ["All"] + sorted(df_metrics["metric"].dropna().unique().tolist())
                    sel_metric = st.selectbox("Filter Metric", metrics)

                filtered_df = df_metrics.copy()
                if sel_scheme != "All":
                    filtered_df = filtered_df[filtered_df["scheme"] == sel_scheme]
                if sel_plan != "All":
                    filtered_df = filtered_df[filtered_df["plan"] == sel_plan]
                if sel_metric != "All":
                    filtered_df = filtered_df[filtered_df["metric"] == sel_metric]

                st.dataframe(filtered_df, use_container_width=True)
                st.caption(f"Showing {len(filtered_df)} of {len(df_metrics)} records")

                csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Filtered CSV", data=csv_bytes, file_name="fund_metrics_filtered.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error loading metrics: {e}")

    with exp_tab2:
        st.markdown("### Indexed SID Corpus Statistics")

        meta_file = settings.index_dir / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Indexed Chunks", f"{meta.get('n', 0):,}")
                with col2:
                    st.metric("Total SID Documents", meta.get("docs", 0))
                with col3:
                    st.metric("Chunk Size / Overlap", f"{meta.get('chunk_size', 0)} / {meta.get('overlap', 0)}")
                with col4:
                    st.metric("Backend Engine", meta.get("backend", "sqlite_fts5"))
            except Exception:
                pass

        st.markdown("#### Document Catalog")
        try:
            retriever = get_retriever()
            retriever.load()
            catalog = retriever._doc_catalog
            if catalog:
                cat_data = [
                    {"Document ID": c[0], "Scheme Name": c[1].title() if c[1] else "General SID"}
                    for c in catalog
                ]
                df_cat = pd.DataFrame(cat_data)
                search_term = st.text_input("Search catalog by AMC or scheme name", "")
                if search_term:
                    df_cat = df_cat[df_cat["Document ID"].str.contains(search_term, case=False) |
                                    df_cat["Scheme Name"].str.contains(search_term, case=False)]
                st.dataframe(df_cat, use_container_width=True)
                st.caption(f"Showing {len(df_cat)} of {len(catalog)} documents in SQLite index.")
            else:
                st.info("No documents catalog loaded.")
        except Exception as e:
            st.warning(f"Could not load document catalog: {e}")

    with exp_tab3:
        st.markdown("### Raw Hybrid Search Tester (`search_scheme_docs`)")
        st.caption("Test BM25 + Dense vector Reciprocal Rank Fusion (RRF) directly on the indexed SIDs.")

        col_q, col_s, col_k = st.columns([3, 2, 1])
        with col_q:
            test_q = st.text_input("Search Query", value="exit load lock in period")
        with col_s:
            test_scheme = st.text_input("Scheme Filter (Optional)", value="Axis Banking and PSU Debt Fund")
        with col_k:
            test_k = st.number_input("Top K", min_value=1, max_value=10, value=4)

        if st.button("Run Hybrid Search", type="primary"):
            with st.spinner("Searching SQLite FTS5 + Dense embeddings..."):
                try:
                    res = run_search(query=test_q, scheme=test_scheme or None, top_k=int(test_k))
                    hits = res.get("hits", [])
                    st.success(f"Retrieved {len(hits)} matching passages. Injection lines removed: {res.get('injection_lines_removed', 0)}")

                    for i, h in enumerate(hits, 1):
                        with st.expander(f"Hit [{i}] — {h['doc_id']} | {h['section']} (Score: {h['score']})", expanded=(i == 1)):
                            st.write(f"**Scheme:** {h['scheme'] or 'N/A'} | **Month:** {h['month'] or 'N/A'}")
                            st.markdown(f"```text\n{h['snippet']}\n```")
                except Exception as e:
                    st.error(f"Search failed: {e}")
