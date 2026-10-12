"""
Run with: streamlit run dashboard/app.py

Reads directly from the trace database — no separate backend needed. If the
database is empty (e.g. a fresh Streamlit Community Cloud deploy), it seeds
itself by running the bundled demo with the offline heuristic judge.

API key policy (see DEPLOYMENT.md):
  1. a key the visitor pastes in the sidebar — held in their session only
  2. otherwise the server's ANTHROPIC_API_KEY (st.secrets / env), if the
     owner set one, capped per session
  3. otherwise the offline heuristic judge
Keys are never written to the trace DB, logs, or the page.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from anchor.cli import run_demo  # noqa: E402
from anchor.compare import per_query_diff, summarize_runs  # noqa: E402
from anchor.judge import FaithfulnessJudge  # noqa: E402
from anchor.schema import Chunk  # noqa: E402
from anchor.tracer import Tracer  # noqa: E402

# Fixed server-side path; deliberately not a text box, so visitors to a
# public deploy can't point the app at arbitrary files on the host.
DB_PATH = os.environ.get("ANCHOR_DB", str(ROOT / "anchor_traces.db"))
SERVER_KEY_SESSION_LIMIT = int(os.environ.get("ANCHOR_SERVER_KEY_SESSION_LIMIT", "5"))

st.set_page_config(page_title="Anchor — RAG Eval Dashboard", page_icon="⚓", layout="wide")


# ------------------------------------------------------------------ keys


def _server_key():
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets.toml locally is fine
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY") or None


with st.sidebar:
    st.header("Judge")
    user_key = st.text_input(
        "Anthropic API key (optional)",
        type="password",
        help="Used only for this browser session to run the LLM judge in the playground. "
        "Never stored or logged. Leave empty to use the offline heuristic judge.",
    ).strip()
    server_key = _server_key()
    st.session_state.setdefault("server_key_calls", 0)

    if user_key:
        api_key, key_source = user_key, "your key"
    elif server_key and st.session_state.server_key_calls < SERVER_KEY_SESSION_LIMIT:
        api_key, key_source = server_key, "the demo's key"
    else:
        api_key, key_source = None, None

    if api_key:
        st.caption(f"Playground uses the **LLM judge** with {key_source}.")
    else:
        st.caption("Playground uses the **offline heuristic judge**. Add your own API key above for claim-level LLM grounding.")
    if server_key and not user_key:
        left = max(SERVER_KEY_SESSION_LIMIT - st.session_state.server_key_calls, 0)
        st.caption(f"{left} LLM judge calls left on the demo's key this session.")


# ------------------------------------------------------------------ data


@st.cache_resource
def _seed_if_empty(db_path: str) -> bool:
    tracer = Tracer(db_path)
    empty = not tracer.fetch_runs()
    tracer.close()
    if empty:
        run_demo(db_path, judge=FaithfulnessJudge(backend="heuristic"), quiet=True)
    return empty


def load_runs(db_path: str) -> pd.DataFrame:
    tracer = Tracer(db_path)
    runs = tracer.fetch_runs()
    tracer.close()
    df = pd.DataFrame(runs)
    if df.empty:
        return df
    judge = df["judge_json"].apply(json.loads)
    retrieval = df["retrieval_json"].apply(lambda r: json.loads(r) if r else None)
    df["faithfulness"] = judge.apply(lambda j: j["faithfulness_score"])
    df["judge_backend"] = judge.apply(lambda j: j["judge_backend"])
    for key in ("recall_at_k", "mrr", "ndcg_at_k", "map_score"):
        df[key] = retrieval.apply(lambda r, k=key: r[k] if r else None)
    return df


try:
    seeded = _seed_if_empty(DB_PATH)
    df = load_runs(DB_PATH)
except Exception as e:
    st.error(f"Couldn't open the trace database ({type(e).__name__}).")
    st.stop()

st.title("⚓ Anchor")
st.caption(
    "Deterministic retrieval metrics + claim-level, citation-grounded faithfulness scoring for RAG pipelines. "
    "[Source on GitHub](https://github.com/arnabmanandhar/anchor)"
)
if seeded:
    st.info("This database was empty, so it was seeded with the bundled demo (two retriever variants, heuristic judge).")
if df.empty:
    st.info("No runs yet. Run `anchor demo`, then reload this page.")
    st.stop()

versions = sorted(df["pipeline_version"].unique())
overview, compare_tab, inspect_tab, playground = st.tabs(["Overview", "Compare versions", "Inspect a run", "Judge playground"])

# -------------------------------------------------------------- overview

with overview:
    selected = st.multiselect("Pipeline versions", versions, default=versions)
    view = df[df["pipeline_version"].isin(selected)]
    if view.empty:
        st.warning("Pick at least one version.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Runs", len(view))
        c2.metric("Mean faithfulness", f"{view['faithfulness'].mean():.2f}")
        c3.metric("Mean recall@k", f"{view['recall_at_k'].mean():.2f}" if view["recall_at_k"].notna().any() else "—")
        c4.metric("Mean nDCG@k", f"{view['ndcg_at_k'].mean():.2f}" if view["ndcg_at_k"].notna().any() else "—")
        c5.metric("Mean latency (ms)", f"{view['latency_ms'].mean():.0f}")

        means = view.groupby("pipeline_version")[["faithfulness", "recall_at_k", "mrr", "ndcg_at_k", "map_score"]].mean()
        st.subheader("Mean metrics by version")
        st.bar_chart(means.T, stack=False)

        st.subheader("Runs")
        cols = ["run_id", "pipeline_version", "query", "faithfulness", "judge_backend", "recall_at_k", "mrr", "ndcg_at_k", "latency_ms"]
        st.dataframe(view[cols], hide_index=True, width="stretch")

# --------------------------------------------------------------- compare

with compare_tab:
    if len(versions) < 2:
        st.info("Compare needs runs from at least two pipeline versions.")
    else:
        ca, cb = st.columns(2)
        va = ca.selectbox("Baseline (A)", versions, index=0)
        vb = cb.selectbox("Candidate (B)", versions, index=1)
        runs_a = df[df["pipeline_version"] == va].to_dict("records")
        runs_b = df[df["pipeline_version"] == vb].to_dict("records")
        sa, sb = summarize_runs(runs_a), summarize_runs(runs_b)

        rows = []
        for key, label in [
            ("mean_faithfulness", "Faithfulness"),
            ("mean_recall_at_k", "Recall@k"),
            ("mean_mrr", "MRR"),
            ("mean_ndcg_at_k", "nDCG@k"),
            ("mean_map", "MAP"),
            ("mean_latency_ms", "Latency (ms)"),
        ]:
            a, b = sa.get(key), sb.get(key)
            rows.append({"metric": label, va: a, vb: b, "B − A": (b - a) if a is not None and b is not None else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.subheader("Per-query changes")
        st.caption("Runs matched by query text. A flat mean can hide a regression on one question that another question's gain cancels out.")
        diffs = per_query_diff(runs_a, runs_b)
        if not diffs:
            st.info("No queries in common between these versions.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "query": d["query"],
                            "status": d["status"],
                            "Δ faithfulness": d["delta"]["faithfulness"],
                            "Δ recall@k": d["delta"]["recall_at_k"],
                            "Δ nDCG@k": d["delta"]["ndcg_at_k"],
                            "Δ MAP": d["delta"]["map_score"],
                        }
                        for d in diffs
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

# --------------------------------------------------------------- inspect

with inspect_tab:
    labels = {f"{r.pipeline_version} · {r.query} · {r.run_id}": r.run_id for r in df.itertuples()}
    choice = st.selectbox("Run", list(labels))
    row = df[df["run_id"] == labels[choice]].iloc[0]
    judge_data = json.loads(row["judge_json"])

    st.markdown(f"**Query:** {row['query']}")
    if row.get("generated_answer"):
        st.markdown(f"**Generated answer:** {row['generated_answer']}")
    st.markdown(
        f"**Faithfulness:** {judge_data['faithfulness_score']:.2f} "
        f"({sum(c['supported'] for c in judge_data['claims'])}/{len(judge_data['claims'])} claims supported, "
        f"backend: {judge_data['judge_backend']})"
    )
    if judge_data.get("reasoning"):
        st.caption(judge_data["reasoning"])

    st.write("**Claim-level audit**")
    if judge_data["claims"]:
        st.dataframe(pd.DataFrame(judge_data["claims"]), hide_index=True, width="stretch")

    retrieved = json.loads(row["retrieved_json"]) if row.get("retrieved_json") else []
    reference = set(json.loads(row["reference_json"]) or []) if row.get("reference_json") else set()
    if retrieved:
        st.write("**Retrieved chunks (ranked)**")
        cited = {c["chunk_id"] for c in judge_data["claims"] if c["supported"]}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rank": i,
                        "id": c["id"],
                        "relevant (ground truth)": c["id"] in reference if reference else None,
                        "cited by judge": c["id"] in cited,
                        "score": c.get("score"),
                        "text": c["text"],
                    }
                    for i, c in enumerate(retrieved, start=1)
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    tracer = Tracer(DB_PATH)
    spans = tracer.fetch_spans(row["run_id"])
    tracer.close()
    if spans:
        st.write("**Trace (spans)**")
        st.dataframe(pd.DataFrame(spans)[["name", "duration_ms", "metadata_json"]], hide_index=True, width="stretch")

# ------------------------------------------------------------ playground

with playground:
    st.write(
        "Paste retrieved chunks and an answer to see how the judge audits it. "
        "Nothing here is saved to the trace database."
    )
    question = st.text_input("Question", "How many provinces does Nepal have?")
    chunks_text = st.text_area(
        "Retrieved chunks, one per line as `id: text`",
        "c2: Nepal's federal structure divides the country into seven provinces.\n"
        "c1: The Constitution of Nepal establishes Nepal as a federal democratic republic.",
        height=120,
    )
    answer = st.text_area(
        "Generated answer",
        "Nepal is divided into seven provinces. Each province has its own elected governor.",
        height=80,
    )

    if st.button("Run judge", type="primary"):
        chunks = []
        for i, line in enumerate(l for l in chunks_text.splitlines() if l.strip()):
            cid, sep, text = line.partition(":")
            chunks.append(Chunk(id=cid.strip(), text=text.strip()) if sep else Chunk(id=f"c{i + 1}", text=line.strip()))
        if not chunks or not answer.strip():
            st.warning("Add at least one chunk and an answer.")
        else:
            judge = FaithfulnessJudge(api_key=api_key, backend="auto" if api_key else "heuristic")
            if api_key and key_source == "the demo's key":
                st.session_state.server_key_calls += 1
            with st.spinner("Judging…"):
                verdict = judge.score(question, chunks, answer)
            st.metric("Faithfulness", f"{verdict.faithfulness_score:.2f}")
            st.caption(f"Backend: {verdict.judge_backend}. {verdict.reasoning or ''}")
            st.dataframe(pd.DataFrame([c.model_dump() for c in verdict.claims]), hide_index=True, width="stretch")
