"""
Run with: streamlit run dashboard/app.py

Reads directly from anchor_traces.db — no separate backend needed.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anchor.tracer import Tracer  # noqa: E402

st.set_page_config(page_title="Anchor — RAG Eval Dashboard", layout="wide")
st.title("⚓ Anchor")
st.caption("Deterministic retrieval metrics + grounded faithfulness scoring for RAG pipelines.")

db_path = st.sidebar.text_input("Trace database", value="anchor_traces.db")

try:
    tracer = Tracer(db_path)
    runs = tracer.fetch_runs()
except Exception as e:
    st.error(f"Couldn't open {db_path}: {e}")
    st.stop()

if not runs:
    st.info("No runs yet. Run `python -m anchor.cli demo` first, then reload this page.")
    st.stop()

df = pd.DataFrame(runs)
df["faithfulness_score"] = df["judge_json"].apply(lambda j: json.loads(j)["faithfulness_score"])
df["judge_backend"] = df["judge_json"].apply(lambda j: json.loads(j)["judge_backend"])
df["recall_at_k"] = df["retrieval_json"].apply(
    lambda r: json.loads(r)["recall_at_k"] if r and r != "null" else None
)
df["mrr"] = df["retrieval_json"].apply(lambda r: json.loads(r)["mrr"] if r and r != "null" else None)

versions = sorted(df["pipeline_version"].unique())
selected_versions = st.sidebar.multiselect("Pipeline versions", versions, default=versions)
df = df[df["pipeline_version"].isin(selected_versions)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Runs", len(df))
col2.metric("Mean faithfulness", f"{df['faithfulness_score'].mean():.2f}")
col3.metric("Mean recall@k", f"{df['recall_at_k'].mean():.2f}" if df["recall_at_k"].notna().any() else "—")
col4.metric("Mean latency (ms)", f"{df['latency_ms'].mean():.0f}")

st.subheader("Faithfulness by pipeline version")
st.bar_chart(df.groupby("pipeline_version")["faithfulness_score"].mean())

if df["recall_at_k"].notna().any():
    st.subheader("Recall@k by pipeline version")
    st.bar_chart(df.groupby("pipeline_version")["recall_at_k"].mean())

st.subheader("Runs")
show_cols = ["run_id", "pipeline_version", "query", "faithfulness_score", "judge_backend", "recall_at_k", "mrr", "latency_ms"]
st.dataframe(df[show_cols], use_container_width=True)

st.subheader("Inspect a run")
run_id = st.selectbox("Run ID", df["run_id"].tolist())
if run_id:
    row = df[df["run_id"] == run_id].iloc[0]
    st.write("**Query:**", row["query"])
    judge_data = json.loads(row["judge_json"])
    st.write(f"**Faithfulness score:** {judge_data['faithfulness_score']:.2f}  (backend: {judge_data['judge_backend']})")
    st.write("**Claim-level audit:**")
    st.table(pd.DataFrame(judge_data["claims"]))

    spans = tracer.fetch_spans(run_id)
    st.write("**Trace (spans):**")
    st.table(pd.DataFrame(spans)[["name", "duration_ms", "created_at"]])
