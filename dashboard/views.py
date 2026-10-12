"""Page implementations for the Anchor dashboard. Entry point: dashboard/app.py."""


import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui  # noqa: E402
from anchor.cli import run_demo  # noqa: E402
from anchor.compare import per_query_diff, significance, summarize_runs  # noqa: E402
from anchor.gate import GATE_METRICS, GATE_RUN_KEYS  # noqa: E402
from anchor.judge import DEFAULT_MODEL, FaithfulnessJudge  # noqa: E402
from anchor.records import load_records  # noqa: E402
from anchor.retrieval_metrics import all_retrieval_metrics  # noqa: E402
from anchor.runner import evaluate_pipeline, evaluate_records, summarize  # noqa: E402
from anchor.sandbox import RETRIEVERS, make_pipeline, parse_corpus, parse_dataset, parse_reference_ids  # noqa: E402
from anchor.schema import Chunk  # noqa: E402
from anchor.tracer import Tracer  # noqa: E402

EXAMPLES = ROOT / "examples"
def _setting(name: str, default: str = "") -> str:
    """Environment variable, else a top-level Streamlit secret, else default."""
    value = os.environ.get(name)
    if value is None:
        try:
            value = st.secrets.get(name)
        except Exception:  # no secrets.toml locally is fine
            value = None
    return str(value) if value is not None else default


def public_mode() -> bool:
    return _setting("ANCHOR_PUBLIC").lower() in ("1", "true", "yes")
SERVER_KEY_SESSION_LIMIT = int(_setting("ANCHOR_SERVER_KEY_SESSION_LIMIT", "5"))
MAX_LLM_CASES = int(_setting("ANCHOR_MAX_LLM_CASES", "25"))



# =================================================================== state


def _db_path() -> str:
    """Fixed server-side location (never a text box, so visitors can't point
    the app at other files). Public mode: one private DB per session."""
    if public_mode():
        if "db_path" not in st.session_state:
            st.session_state.db_path = str(Path(tempfile.mkdtemp(prefix="anchor-")) / "traces.db")
        return st.session_state.db_path
    return os.environ.get("ANCHOR_DB", str(ROOT / "anchor_traces.db"))




def db() -> str:
    return _db_path()


def setup() -> None:
    """Per-run setup. Called from app.py on every rerun (module-level code
    in an imported module would only run once per server process)."""
    st.set_page_config(page_title="Anchor · RAG evaluation", page_icon=":material/anchor:", layout="wide")
    ui.inject_css()
    _seed_once()


def _seed_once() -> None:
    key = f"seeded::{db()}"
    if st.session_state.get(key):
        return
    tracer = Tracer(db())
    empty = not tracer.fetch_runs()
    tracer.close()
    if empty:
        run_demo(db(), judge=FaithfulnessJudge(backend="heuristic"), quiet=True)
        st.session_state.just_seeded = True
    st.session_state[key] = True


def load_df() -> pd.DataFrame:
    tracer = Tracer(db())
    runs = tracer.fetch_runs()
    tracer.close()
    return ui.runs_frame(runs)


def all_versions(df: pd.DataFrame):
    return sorted(df["pipeline_version"].unique()) if not df.empty else []



# =================================================================== keys


def _server_key():
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets.toml locally is fine
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY") or None


def resolve_key():
    """(api_key, source) for the next LLM call, or (None, None)."""
    user_key = st.session_state.get("user_api_key", "").strip()
    if user_key:
        return user_key, "your key"
    server_key = _server_key()
    if server_key and st.session_state.get("server_key_calls", 0) < SERVER_KEY_SESSION_LIMIT:
        return server_key, "the server's key"
    return None, None


def charge_server_key(n: int = 1) -> None:
    _, source = resolve_key()
    if source == "the server's key":
        st.session_state.server_key_calls = st.session_state.get("server_key_calls", 0) + n


def make_judge(choice: str) -> FaithfulnessJudge:
    key, _ = resolve_key()
    if choice == "LLM (Claude)" and key:
        return FaithfulnessJudge(api_key=key, backend="llm")
    return FaithfulnessJudge(backend="heuristic")


# ================================================================= sidebar


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### :material/anchor: Anchor")
        st.caption("Deterministic-first evaluation for RAG pipelines.")
        st.divider()
        st.markdown("**LLM judge (optional)**")
        st.text_input(
            "Anthropic API key",
            type="password",
            key="user_api_key",
            placeholder="sk-ant-…",
            help="Used only in this browser session, for the LLM judge and the Claude generator. "
            "Never stored or logged. Leave empty to run everything offline.",
        )
        key, source = resolve_key()
        if key:
            st.success(f"LLM judge available, using {source}.", icon=":material/check_circle:")
        else:
            st.info("Running offline: the heuristic judge is used. Add a key to audit claims with Claude.", icon=":material/info:")
        if _server_key() and not st.session_state.get("user_api_key"):
            left = max(SERVER_KEY_SESSION_LIMIT - st.session_state.get("server_key_calls", 0), 0)
            st.caption(f"{left} LLM calls left on the server's key this session.")
        st.divider()
        if public_mode():
            st.caption(":material/lock: Private workspace: your runs are visible only in this browser session.")
        else:
            st.caption(f"Workspace: `{Path(db()).name}` (shared with the `anchor` CLI)")
        st.caption("[GitHub](https://github.com/arnabmanandhar/anchor) · [User guide](https://github.com/arnabmanandhar/anchor/blob/main/docs/USER_GUIDE.md)")


# ==================================================================== home


def page_home():
    st.markdown(
        '<div class="anchor-hero"><h1>Anchor</h1>'
        "<p>Did this change to my RAG pipeline make it better or worse? Anchor answers that with "
        "<b>retrieval metrics that are pure math</b> and a <b>faithfulness judge that has to show its work</b>.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown(ui.card("Deterministic retrieval metrics",
                        "Recall@k, MRR, nDCG@k and MAP from ranked chunk ids against ground truth. "
                        "No model call, so the same input always gives the same number and CI never flakes."),
                unsafe_allow_html=True)
    c2.markdown(ui.card("Audited faithfulness",
                        "The judge splits an answer into claims and must cite the retrieved chunk that "
                        "supports each one. The score is supported ÷ total claims, so every number has a reason."),
                unsafe_allow_html=True)
    c3.markdown(ui.card("Traces and regression gates",
                        "Every run is stored with timed spans. Compare two versions query by query, and "
                        "fail a build when recall or faithfulness drops."),
                unsafe_allow_html=True)

    st.markdown("#### Start here")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Try it in 30 seconds**")
        st.caption("Pick a sample corpus and retriever and run an evaluation right here in the browser.")
        nav_link("evaluate", label="Evaluate a pipeline", icon=":material/play_circle:")
    with b:
        st.markdown("**Bring your own pipeline**")
        st.caption("Upload a JSONL of the queries, ranked chunks and answers your system produced.")
        nav_link("evaluate", label="Upload outputs", icon=":material/upload_file:")
    with c:
        st.markdown("**Understand the numbers**")
        st.caption("What each metric measures, with the formulas and worked examples.")
        nav_link("learn", label="Learn the metrics", icon=":material/menu_book:")

    st.markdown("#### How it works")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fillcolor="#eef4fc", color="#2a78d6", fontname="Helvetica", fontsize=11];
          edge [color="#888888", fontname="Helvetica", fontsize=10];
          q [label="Question"]; p [label="Your RAG pipeline\\n(any retriever + LLM)"];
          r [label="Retrieval metrics\\nRecall · MRR · nDCG · MAP", fillcolor="#e8f6f0", color="#1baf7a"];
          j [label="Faithfulness judge\\nclaim → cited chunk", fillcolor="#e8f6f0", color="#1baf7a"];
          t [label="Trace store\\n(SQLite)"]; o [label="Compare · Inspect · Gate"];
          q -> p; p -> r [label="ranked chunk ids"]; p -> j [label="chunks + answer"];
          r -> t; j -> t; t -> o;
        }
        """
    )

    df = load_df()
    if not df.empty:
        st.markdown("#### This workspace")
        m1, m2, m3 = st.columns(3)
        m1.metric("Pipeline versions", df["pipeline_version"].nunique())
        m2.metric("Evaluated runs", len(df))
        m3.markdown(f"Judge backends used<br><span class='ws-value'>{', '.join(sorted(df['judge_backend'].unique()))}</span>",
                    unsafe_allow_html=True)
        if st.session_state.pop("just_seeded", False):
            st.caption("Seeded with the bundled demo: a TF-IDF retriever and a weaker keyword retriever on 8 questions.")


# ================================================================ evaluate


def _sample_corpus():
    return parse_corpus((EXAMPLES / "sample_corpus.json").read_text(), "sample_corpus.json")


def _questions_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [{"query": r["query"], "reference_chunk_ids": ", ".join(r.get("reference_chunk_ids") or [])} for r in rows]
    )


def _show_run_summary(results, versions):
    s = summarize(results)
    st.success(f"Evaluated {s['n_runs']} cases across {len(versions)} version(s): {', '.join(versions)}.", icon=":material/check_circle:")
    ui.score_cards({k: s.get(sk) for k, sk, _, _ in ui.METRICS})
    c1, c2 = st.columns(2)
    nav_link("results", label="See full results", icon=":material/bar_chart:", container=c1)
    if len(versions) > 1 or len(all_versions(load_df())) > 1:
        nav_link("compare", label="Compare versions", icon=":material/compare_arrows:", container=c2)


def _tab_build():
    st.markdown(
        "Build a small RAG pipeline from parts and evaluate it. Useful for learning what each metric "
        "rewards, and for comparing retrievers on your own documents."
    )

    st.markdown("##### 1 · Corpus")
    source = st.radio("Corpus source", ["Sample: Constitution of Nepal (20 chunks)", "Upload a file", "Paste text"],
                      horizontal=True, label_visibility="collapsed")
    chunks, corpus_error = None, None
    try:
        if source.startswith("Sample"):
            chunks = _sample_corpus()
        elif source == "Upload a file":
            f = st.file_uploader("Corpus file", type=["csv", "json", "jsonl", "txt", "md"],
                                 help="CSV with id,text columns · JSON list of {id,text} · JSONL · or plain text "
                                 "with chunks separated by blank lines.")
            if f:
                chunks = parse_corpus(f.getvalue(), f.name)
        else:
            text = st.text_area("Paste chunks, separated by blank lines", height=160,
                                placeholder="First chunk of text…\n\nSecond chunk…")
            if text.strip():
                chunks = parse_corpus(text, "pasted.txt")
    except Exception as e:
        corpus_error = str(e)
    if corpus_error:
        st.error(f"Couldn't read the corpus: {corpus_error}")
    if chunks:
        with st.expander(f"Preview corpus · {len(chunks)} chunks"):
            st.dataframe(pd.DataFrame([c.model_dump(exclude={"score"}) for c in chunks]), hide_index=True,
                         width="stretch", height=240)

    st.markdown("##### 2 · Questions")
    st.caption("Reference chunk ids are optional ground truth (comma separated). Without them only faithfulness is scored.")
    qsrc_key = f"questions::{source}"
    if qsrc_key not in st.session_state:
        if source.startswith("Sample"):
            st.session_state[qsrc_key] = _questions_frame(json.loads((EXAMPLES / "sample_questions.json").read_text()))
        else:
            st.session_state[qsrc_key] = pd.DataFrame([{"query": "", "reference_chunk_ids": ""}])
    qfile = st.file_uploader("Or load questions from a file", type=["csv", "json", "jsonl"], key=f"qfile::{source}",
                             help="JSON / JSONL of {query, reference_chunk_ids} or CSV with those columns.")
    if qfile is not None and st.session_state.get(f"qfile_loaded::{source}") != qfile.file_id:
        try:
            st.session_state[qsrc_key] = _questions_frame(parse_dataset(qfile.getvalue(), qfile.name))
            st.session_state[f"qfile_loaded::{source}"] = qfile.file_id
        except Exception as e:
            st.error(f"Couldn't read the questions file: {e}")
    edited = st.data_editor(
        st.session_state[qsrc_key],
        num_rows="dynamic",
        width="stretch",
        key=f"qeditor::{source}",
        column_config={
            "query": st.column_config.TextColumn("Question", width="large", required=True),
            "reference_chunk_ids": st.column_config.TextColumn("Reference chunk ids", help="e.g. art-031, art-035"),
        },
    )
    dataset = [
        {"query": str(r["query"]).strip(), "reference_chunk_ids": parse_reference_ids(r.get("reference_chunk_ids"))}
        for _, r in edited.iterrows()
        if str(r.get("query") or "").strip()
    ]

    st.markdown("##### 3 · Pipeline")
    key, _ = resolve_key()
    c1, c2, c3 = st.columns(3)
    retriever = c1.selectbox(
        "Retriever", RETRIEVERS, format_func=lambda r: {"bm25": "BM25", "tfidf": "TF-IDF", "keyword": "Keyword overlap",
                                                        "hybrid": "Hybrid (BM25 + TF-IDF, RRF)"}[r],
        help="All built in and offline. Keyword overlap is deliberately weak, to show what a regression looks like.",
    )
    top_k = c2.slider("Chunks retrieved (k)", 1, 10, 3, help="Also the cutoff for Recall@k and nDCG@k.")
    generator = c3.selectbox(
        "Answer generator", ["extractive", "llm"],
        format_func=lambda g: "Extractive (offline)" if g == "extractive" else f"Claude ({DEFAULT_MODEL})",
        help="Extractive copies the best-matching sentences from the retrieved chunks. Claude writes an answer "
        "from the chunks and needs an API key.",
    )
    c4, c5 = st.columns([1, 2])
    judge_choice = c4.selectbox("Faithfulness judge", ["Heuristic (offline)", "LLM (Claude)"],
                                index=1 if key else 0, help="The LLM judge needs an API key.")
    default_name = f"{retriever}-k{top_k}-{'claude' if generator == 'llm' else 'extractive'}"
    version = c5.text_input("Version name", value=default_name,
                            help="Runs are stored under this label. Reuse a name to replace its runs.")
    all_four = st.checkbox("Also run every other retriever with the same settings (quick comparison)")

    needs_key = generator == "llm" or judge_choice == "LLM (Claude)"
    problems = []
    if not chunks:
        problems.append("add a corpus")
    if not dataset:
        problems.append("add at least one question")
    if needs_key and not key:
        problems.append("add an API key in the sidebar for Claude, or switch to the offline options")
    if needs_key and len(dataset) * (4 if all_four else 1) > MAX_LLM_CASES:
        problems.append(f"keep LLM runs to {MAX_LLM_CASES} cases or fewer per click")
    if problems:
        st.caption("To run: " + "; ".join(problems) + ".")

    if st.button("Run evaluation", type="primary", disabled=bool(problems)):
        corpus_ids = {c.id for c in chunks}
        unknown = sorted({i for row in dataset for i in (row["reference_chunk_ids"] or []) if i not in corpus_ids})
        if unknown:
            st.warning(f"These reference ids aren't in the corpus, so they can never be retrieved: {', '.join(unknown[:8])}")
        plan = [(version, retriever)]
        if all_four:
            plan += [(version.replace(retriever, r, 1) if retriever in version else f"{version}-{r}", r)
                     for r in RETRIEVERS if r != retriever]
        tracer = Tracer(db())
        bar = st.progress(0.0, text="Evaluating…")
        all_results, done, total = [], 0, len(plan) * len(dataset)
        try:
            for name, method in plan:
                tracer.delete_version(name)
                pipeline = make_pipeline(chunks, method, top_k, generator, api_key=key if generator == "llm" else None)

                def tick(i, n, name=name):
                    bar.progress((done + i) / total, text=f"{name}: {i}/{n}")

                all_results += evaluate_pipeline(pipeline, dataset, pipeline_version=name, k=top_k, tracer=tracer,
                                                 judge=make_judge(judge_choice), progress=tick)
                done += len(dataset)
            charge_server_key(total if needs_key else 0)
        except Exception as e:
            st.error(f"Evaluation stopped: {type(e).__name__}. Runs finished so far were saved.")
        finally:
            tracer.close()
            bar.empty()
        if all_results:
            _show_run_summary(all_results, [p[0] for p in plan])


SAMPLE_RECORDS = EXAMPLES / "sample_records.jsonl"


def _tab_upload():
    st.markdown(
        "Already have a pipeline? Log what it produced for each question and upload the file. Anchor scores it "
        "without calling your system, so it works for any stack."
    )
    with st.expander("File format", expanded=False):
        st.markdown(
            "One record per question, as **JSONL** (one JSON object per line), a **JSON** list, or **CSV** "
            "(with `retrieved_chunks` as a JSON array string):"
        )
        st.code(
            '{"query": "How many provinces does Nepal have?",\n'
            ' "retrieved_chunks": [{"id": "c2", "text": "...seven provinces."}, {"id": "c6", "text": "..."}],\n'
            ' "answer": "Nepal has seven provinces.",\n'
            ' "reference_chunk_ids": ["c2"]}',
            language="json",
        )
        st.markdown(
            "- `retrieved_chunks` are ranked, best first. A chunk can be a plain string (ids become r1, r2, …).\n"
            "- `reference_chunk_ids` is optional; without it only faithfulness is scored.\n"
            "- Also accepted: `contexts` / `chunks` for chunks, `response` / `generated_answer` for the answer."
        )
        st.download_button("Download a sample file", SAMPLE_RECORDS.read_bytes(), file_name="sample_records.jsonl",
                           mime="application/jsonl")
    f = st.file_uploader("Records file", type=["jsonl", "json", "csv"])
    use_sample = st.checkbox("Use the sample file instead", value=f is None)
    records, error = None, None
    try:
        if f is not None and not use_sample:
            records = load_records(f.getvalue(), f.name)
        elif use_sample:
            records = load_records(SAMPLE_RECORDS.read_bytes(), SAMPLE_RECORDS.name)
    except Exception as e:
        error = str(e)
    if error:
        st.error(f"Couldn't read the file: {error}")
    if records:
        with st.expander(f"Preview · {len(records)} records"):
            st.dataframe(pd.DataFrame([{"query": r["query"], "chunks": len(r["retrieved_chunks"]), "answer": r["answer"],
                                        "reference_chunk_ids": ", ".join(r["reference_chunk_ids"] or [])} for r in records]),
                         hide_index=True, width="stretch")
    key, _ = resolve_key()
    c1, c2, c3 = st.columns([2, 1, 1])
    version = c1.text_input("Version name", value="my-pipeline-v1" if not use_sample else "sample-logged-pipeline")
    k = c2.number_input("k (metric cutoff)", 1, 50, 5)
    judge_choice = c3.selectbox("Judge", ["Heuristic (offline)", "LLM (Claude)"], index=1 if key else 0, key="upload_judge")
    blocked = not records or (judge_choice == "LLM (Claude)" and (not key or len(records) > MAX_LLM_CASES))
    if judge_choice == "LLM (Claude)" and not key:
        st.caption("Add an API key in the sidebar to use the LLM judge.")
    if st.button("Score these outputs", type="primary", disabled=blocked):
        tracer = Tracer(db())
        tracer.delete_version(version)
        bar = st.progress(0.0, text="Scoring…")
        try:
            results = evaluate_records(records, pipeline_version=version, k=int(k), tracer=tracer,
                                       judge=make_judge(judge_choice),
                                       progress=lambda i, n: bar.progress(i / n, text=f"{i}/{n}"))
            if judge_choice == "LLM (Claude)":
                charge_server_key(len(records))
        finally:
            tracer.close()
            bar.empty()
        _show_run_summary(results, [version])


def _tab_single():
    st.markdown("Score one answer by hand: paste what was retrieved (in rank order) and what was generated.")
    question = st.text_input("Question", "How many provinces does Nepal have?", key="single_q")
    st.caption("Retrieved chunks, best first")
    chunk_df = st.data_editor(
        pd.DataFrame(
            [
                {"id": "prov-007", "text": "Nepal is divided into seven provinces, each with its own provincial assembly and government."},
                {"id": "art-056", "text": "Article 56 sets out three levels of the State: the federation, the provinces, and the local level."},
            ]
        ),
        num_rows="dynamic",
        width="stretch",
        key="single_chunks",
        column_config={"id": st.column_config.TextColumn("Chunk id", required=True),
                       "text": st.column_config.TextColumn("Text", width="large", required=True)},
    )
    answer = st.text_area("Generated answer",
                          "Nepal has seven provinces. Each province is governed by a chief minister appointed by the king.",
                          key="single_a")
    c1, c2, c3 = st.columns([2, 1, 1])
    refs = c1.text_input("Reference chunk ids (optional)", "prov-007, art-056", key="single_refs")
    k = c2.number_input("k", 1, 50, 3, key="single_k")
    key, _ = resolve_key()
    judge_choice = c3.selectbox("Judge", ["Heuristic (offline)", "LLM (Claude)"], index=1 if key else 0, key="single_judge")
    if st.button("Score", type="primary"):
        chunks = [Chunk(id=str(r["id"]).strip(), text=str(r["text"]).strip())
                  for _, r in chunk_df.iterrows() if str(r.get("id") or "").strip() and str(r.get("text") or "").strip()]
        if not chunks or not answer.strip():
            st.warning("Add at least one chunk and an answer.")
            return
        judge = make_judge(judge_choice)
        if judge.active_backend == "llm":
            charge_server_key()
        with st.spinner("Scoring…"):
            verdict = judge.score(question, chunks, answer)
        ref_ids = parse_reference_ids(refs)
        retrieval = all_retrieval_metrics([c.id for c in chunks], ref_ids, k=int(k)) if ref_ids else None
        values = {"faithfulness": verdict.faithfulness_score}
        if retrieval:
            values.update({m: getattr(retrieval, m) for m in ("recall_at_k", "mrr", "ndcg_at_k", "map_score")})
        ui.score_cards(values)
        st.caption(f"Judge: {verdict.judge_backend}. {verdict.reasoning or ''}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Claim-level audit**")
            ui.claim_cards([c.model_dump() for c in verdict.claims])
        with right:
            st.markdown("**Retrieved chunks**")
            cited = {c.chunk_id for c in verdict.claims if c.supported}
            ui.chunk_cards([c.model_dump() for c in chunks], set(ref_ids or []), cited)


def page_evaluate():
    st.title("Evaluate")
    t1, t2, t3 = st.tabs(["Build and run a pipeline", "Upload your pipeline's outputs", "Score a single answer"])
    with t1:
        _tab_build()
    with t2:
        _tab_upload()
    with t3:
        _tab_single()


# ================================================================= results


def page_results():
    st.title("Results")
    df = load_df()
    versions = all_versions(df)
    if df.empty:
        st.info("No runs yet.")
        nav_link("evaluate", label="Run an evaluation", icon=":material/play_circle:")
        return
    selected = st.multiselect("Versions", versions, default=versions)
    view = df[df["pipeline_version"].isin(selected)]
    if view.empty:
        st.warning("Pick at least one version.")
        return

    metric_keys = [m[0] for m in ui.METRICS]
    means = view.groupby("pipeline_version")[metric_keys].mean()
    meta = view.groupby("pipeline_version").agg(runs=("run_id", "count"), k=("k", "max"),
                                                judge=("judge_backend", lambda s: ", ".join(sorted(set(s)))),
                                                latency_ms=("latency_ms", "mean"), cost_usd=("cost_usd", "sum"))
    table = meta.join(means).reset_index()

    if len(selected) == 1:
        ui.score_cards(means.iloc[0].to_dict())
    st.markdown("##### Mean scores by version")
    st.caption("Every score is on a 0–1 scale; bars and axes always run to 1.")
    st.altair_chart(ui.metric_bars(means, versions), width="stretch")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "pipeline_version": "Version",
            "runs": "Runs",
            "k": st.column_config.NumberColumn("k", help="Cutoff used for Recall@k and nDCG@k"),
            "judge": "Judge",
            "latency_ms": st.column_config.NumberColumn("Mean latency (ms)", format="%.1f"),
            "cost_usd": st.column_config.NumberColumn("Judge cost ($)", format="%.4f"),
            **ui.SCORE_COLUMNS,
        },
    )

    st.markdown("##### Per question")
    metric = st.selectbox("Metric", metric_keys, format_func=lambda m: ui.METRIC_LABELS[m])
    if view[metric].notna().any():
        st.altair_chart(ui.per_query_bars(view, metric, versions), width="stretch")
    else:
        st.caption("No ground truth for these runs, so retrieval metrics weren't computed.")

    st.markdown("##### All runs")
    cols = ["run_id", "pipeline_version", "query", *metric_keys, "claims", "judge_backend", "latency_ms"]
    st.dataframe(view[cols], hide_index=True, width="stretch",
                 column_config={"pipeline_version": "Version", "query": st.column_config.TextColumn("Question", width="large"),
                                "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f"),
                                **ui.SCORE_COLUMNS})
    d1, d2, d3 = st.columns([1, 1, 2])
    d1.download_button("Download runs (CSV)", view[cols].to_csv(index=False), "anchor_runs.csv", "text/csv")
    d2.download_button("Download summary (JSON)", table.to_json(orient="records", indent=2), "anchor_summary.json",
                       "application/json")
    with d3.popover("Delete a version"):
        doomed = st.selectbox("Version to delete", versions, key="delete_choice")
        if st.button("Delete its runs", type="secondary"):
            tracer = Tracer(db())
            n = tracer.delete_version(doomed)
            tracer.close()
            st.toast(f"Deleted {n} runs from {doomed}.")
            st.rerun()


# ================================================================= compare


def page_compare():
    st.title("Compare versions")
    df = load_df()
    versions = all_versions(df)
    if len(versions) < 2:
        st.info("Compare needs runs from at least two versions.")
        nav_link("evaluate", label="Run another version", icon=":material/play_circle:")
        return
    # Default to the pair that shares the most questions, preferring the
    # pair whose scores differ most (an identical pair is a dull default).
    queries = {v: set(df.loc[df["pipeline_version"] == v, "query"]) for v in versions}
    metric_keys = [m[0] for m in ui.METRICS]
    means = df.groupby("pipeline_version")[metric_keys].mean().fillna(0)
    best = max(((a, b) for i, a in enumerate(versions) for b in versions[i + 1:]),
               key=lambda p: (len(queries[p[0]] & queries[p[1]]), float((means.loc[p[0]] - means.loc[p[1]]).abs().sum())))
    c1, c2, c3 = st.columns([2, 2, 1])
    va = c1.selectbox("Baseline (A)", versions, index=versions.index(best[0]))
    vb = c2.selectbox("Candidate (B)", versions, index=versions.index(best[1]))
    alpha = c3.selectbox("Significance level", [0.01, 0.05, 0.10], index=1, format_func=lambda a: f"{a:g}",
                         help="A difference counts as real when its p-value is below this. 0.05 is the usual choice.")
    if va == vb:
        st.warning("Pick two different versions.")
        return
    runs_a = df[df["pipeline_version"] == va].to_dict("records")
    runs_b = df[df["pipeline_version"] == vb].to_dict("records")
    sa, sb = summarize_runs(runs_a), summarize_runs(runs_b)
    sig = significance(runs_a, runs_b, alpha=alpha)

    cols = st.columns(len(ui.METRICS))
    for col, (key, skey, label, hint) in zip(cols, ui.METRICS):
        a, b = sa.get(skey), sb.get(skey)
        if a is None or b is None:
            col.metric(label, "—")
            continue
        delta = f"{b - a:+.3f}" if abs(b - a) > 1e-9 else None
        col.metric(label, f"{b:.3f}", delta=delta, help=f"{hint}. B = {vb}, A = {va}.")
        test = sig[key]
        if delta is None or test["n"] < 2:
            continue
        if test["significant"]:
            col.caption(f":{'green' if test['mean_delta'] > 0 else 'red'}[:material/verified: significant] · p = {test['p_value']:.3f}")
        else:
            col.caption(f":gray[within noise] · p = {test['p_value']:.3f}")
    st.caption(f"Values are {vb} (B); the delta is B − A. *Significant* means a paired test on the shared "
               f"questions says the change is unlikely to be chance (p < {alpha:g}).")

    pair = df[df["pipeline_version"].isin([va, vb])]
    st.altair_chart(ui.metric_bars(pair.groupby("pipeline_version")[metric_keys].mean(), versions), width="stretch")

    _significance_section(sig, va, vb, alpha)

    st.markdown("##### What changed, question by question")
    st.caption("Runs are matched by question text. A flat mean can hide one question getting worse while another gets better.")
    diffs = per_query_diff(runs_a, runs_b)
    if not diffs:
        st.info("These versions have no questions in common.")
    else:
        order = ["regressed", "mixed", "improved", "unchanged"]
        colour = {"regressed": "red", "mixed": "orange", "improved": "green", "unchanged": "gray"}
        counts = pd.Series([d["status"] for d in diffs]).value_counts()
        st.markdown(" · ".join(f"**{counts.get(s, 0)}** :{colour[s]}[{s}]" for s in order))
        st.dataframe(
            pd.DataFrame([{
                "Question": d["query"], "Status": d["status"],
                "Faithfulness A": d["a"]["faithfulness"], "Faithfulness B": d["b"]["faithfulness"],
                "Δ faithfulness": d["delta"]["faithfulness"], "Δ recall@k": d["delta"]["recall_at_k"],
                "Δ MRR": d["delta"]["mrr"], "Δ nDCG@k": d["delta"]["ndcg_at_k"],
            } for d in sorted(diffs, key=lambda d: order.index(d["status"]))]),
            hide_index=True, width="stretch",
            column_config={"Question": st.column_config.TextColumn(width="large"),
                           "Faithfulness A": ui.score_column("Faithfulness A"),
                           "Faithfulness B": ui.score_column("Faithfulness B"),
                           **{c: st.column_config.NumberColumn(format="%+.3f")
                              for c in ("Δ faithfulness", "Δ recall@k", "Δ MRR", "Δ nDCG@k")}},
        )

    st.markdown("##### Regression gate")
    st.caption("The same check `anchor gate` runs in CI. B fails if any mean falls below its floor or drops more than allowed versus A.")
    g1, g2 = st.columns(2)
    floor_recall = g1.slider("Minimum recall@k", 0.0, 1.0, 0.8, 0.05)
    floor_faith = g1.slider("Minimum faithfulness", 0.0, 1.0, 0.8, 0.05)
    max_drop = g1.slider("Largest allowed drop vs A (any metric)", 0.0, 0.5, 0.02, 0.01)
    only_sig = g1.toggle("Only fail on significant drops", value=False,
                         help="A drop larger than allowed fails only if the paired test says it is real. "
                              "Floors always apply. Useful on small question sets, where means are noisy.")
    checks = []
    for name, floor in (("recall", floor_recall), ("faithfulness", floor_faith)):
        v = sb.get(GATE_METRICS[name])
        checks.append((v is not None and v >= floor, f"{name} {('n/a' if v is None else f'{v:.3f}')} ≥ {floor:.2f}"))
    for name, skey in GATE_METRICS.items():
        a, b = sa.get(skey), sb.get(skey)
        if a is None or b is None:
            continue
        ok, text = a - b <= max_drop + 1e-12, f"{name} drop {a - b:+.3f} ≤ {max_drop:.2f}"
        test = sig[GATE_RUN_KEYS[name]]
        if only_sig and test["n"] >= 2:
            text += f" (p = {test['p_value']:.3f})"
            if not ok and not test["significant"]:
                ok, text = True, text + ", not significant"
        checks.append((ok, text))
    passed = all(ok for ok, _ in checks)
    g2.markdown(f'<span class="badge {"pass" if passed else "fail"}" style="font-size:1rem;padding:.2rem .7rem">'
                f'{"PASS" if passed else "FAIL"}</span>', unsafe_allow_html=True)
    for ok, text in checks:
        g2.markdown(f"{':green[:material/check_circle:]' if ok else ':red[:material/cancel:]'} {text}")
    floors_cli = f"--min-recall {floor_recall:g} --min-faithfulness {floor_faith:g}"
    drops_cli = " ".join(f"--max-{n}-drop {max_drop:g}" for n in GATE_METRICS)
    sig_cli = f" --require-significant --alpha {alpha:g}" if only_sig else ""
    st.code(f"anchor gate {vb} {floors_cli} --baseline {va} {drops_cli}{sig_cli}", language="bash")


def _verdict(test) -> str:
    if abs(test["mean_delta"]) < 1e-12:
        return "no change"
    if test["significant"]:
        return "real improvement" if test["mean_delta"] > 0 else "real regression"
    return "could be chance"


def _significance_section(sig, va, vb, alpha):
    st.markdown("##### Is the difference real?")
    tested = {k: t for k, t in sig.items() if t["n"] >= 2}
    if not tested:
        st.caption("Significance needs at least two questions answered by both versions.")
        return
    n = max(t["n"] for t in tested.values())
    st.caption(
        f"Each bar is the change B − A averaged over the {n} questions both versions answered, with its "
        f"{(1 - alpha) * 100:g}% bootstrap confidence interval. If the interval crosses 0, the change could be chance."
    )
    st.altair_chart(ui.delta_ci_chart(tested), width="stretch")
    st.dataframe(
        pd.DataFrame([{
            "Metric": ui.METRIC_LABELS[k], "Questions": t["n"], "A": t["mean_a"], "B": t["mean_b"],
            "Δ (B − A)": t["mean_delta"], "CI low": t["ci_low"], "CI high": t["ci_high"],
            "p-value": t["p_value"], "Verdict": _verdict(t),
        } for k, t in tested.items()]),
        hide_index=True, width="stretch",
        column_config={"A": ui.score_column("A"), "B": ui.score_column("B"),
                       **{c: st.column_config.NumberColumn(format="%+.3f") for c in ("Δ (B − A)", "CI low", "CI high")},
                       "p-value": st.column_config.NumberColumn(format="%.3f"),
                       "Verdict": st.column_config.TextColumn(width="medium")},
    )
    underpowered = [t["mde"] for t in tested.values()
                    if not t["significant"] and t["mde"] is not None and t["mde"] > abs(t["mean_delta"])]
    if underpowered:
        st.info(f"With {n} shared questions, changes smaller than about ±{max(underpowered):.2f} can't be told apart "
                "from noise. Add questions to detect smaller changes.", icon=":material/science:")
    with st.expander("How this is computed"):
        st.markdown(
            "Anchor pairs the two versions' scores on each shared question and works on the differences, "
            "which removes question-to-question variation.\n\n"
            "- **p-value:** a sign-flip permutation test. If A and B were interchangeable, each difference would be "
            "equally likely to be positive or negative. The p-value is the share of sign patterns with a mean at least "
            "as extreme as the observed one. Exact for up to 14 questions, sampled above that.\n"
            "- **Confidence interval:** percentile bootstrap of the mean difference, 5,000 resamples.\n"
            "- **Smallest detectable change:** (1.96 + 0.84) × sd(Δ) / √n, the change a test this size catches 80% of the time.\n\n"
            "Everything is seeded, so the same runs always give the same numbers. See *Learn the metrics* for details."
        )


# ================================================================= inspect


def page_inspect():
    st.title("Inspect a run")
    df = load_df()
    if df.empty:
        st.info("No runs yet.")
        return
    versions = all_versions(df)
    c1, c2 = st.columns([1, 2])
    version = c1.selectbox("Version", versions)
    sub = df[df["pipeline_version"] == version]
    labels = {f"{r.query}  ·  {r.run_id}": r.run_id for r in sub.itertuples()}
    choice = c2.selectbox("Question", list(labels))
    row = sub[sub["run_id"] == labels[choice]].iloc[0]
    judge = json.loads(row["judge_json"])

    ui.score_cards({k: row.get(k) for k, _, _, _ in ui.METRICS})
    st.markdown(f"**Question:** {ui.esc(row['query'])}")
    st.markdown(f"**Generated answer:** {ui.esc(row.get('generated_answer') or '—')}")
    supported = sum(c["supported"] for c in judge["claims"])
    st.caption(f"{supported} of {len(judge['claims'])} claims supported · judge: {judge['judge_backend']}"
               f"{' (' + judge['judge_model'] + ')' if judge.get('judge_model') else ''} · {judge.get('reasoning') or ''}")

    left, right = st.columns(2)
    with left:
        st.markdown("##### Claim-level audit")
        ui.claim_cards(judge["claims"])
    with right:
        st.markdown("##### Retrieved chunks (ranked)")
        retrieved = json.loads(row["retrieved_json"]) if row.get("retrieved_json") else []
        reference = set(json.loads(row["reference_json"]) or []) if row.get("reference_json") else set()
        cited = {c["chunk_id"] for c in judge["claims"] if c["supported"]}
        if retrieved:
            ui.chunk_cards(retrieved, reference, cited)
            missed = reference - {c["id"] for c in retrieved}
            if missed:
                st.caption(f"Relevant but not retrieved: {', '.join(sorted(missed))}")
        else:
            st.caption("This run was recorded before chunks were stored.")

    tracer = Tracer(db())
    spans = tracer.fetch_spans(row["run_id"])
    tracer.close()
    if spans:
        st.markdown("##### Trace")
        st.altair_chart(ui.span_chart(spans), width="stretch")


# =================================================================== learn


def page_learn():
    doc = ROOT / "docs" / "METRICS.md"
    st.markdown(doc.read_text() if doc.exists() else "docs/METRICS.md not found.")


# Filled in by app.py with the registered st.Page objects. Left empty when a
# page function is rendered on its own (tests), where links become text.
PAGES = {}


def nav_link(name: str, label: str, icon: str = None, container=None) -> None:
    target = container or st
    if name in PAGES:
        target.page_link(PAGES[name], label=label, icon=icon)
    else:
        target.caption(f"{icon or ''} {label}")


def render(page: str) -> None:
    """Render one page without navigation (used by tests)."""
    setup()
    sidebar()
    {"home": page_home, "evaluate": page_evaluate, "results": page_results, "compare": page_compare,
     "inspect": page_inspect, "learn": page_learn}[page]()
