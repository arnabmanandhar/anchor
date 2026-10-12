# Anchor

**A deterministic-first evaluation and tracing layer for RAG pipelines.**

Anchor answers one question you should be able to answer before every deploy:
*did this change to my RAG pipeline make it better or worse?*

Most "AI eval" tools ask a language model to rate an answer 1–10 and call it
science. Anchor doesn't trust that. It splits evaluation into two honest
halves:

1. **Retrieval quality is pure math.** Recall@k, MRR, nDCG, MAP — computed
   from ranked chunk IDs against a ground-truth relevance set. No model
   call, no randomness, same input always produces the same number. Safe
   to run as a regression gate in CI.
2. **Faithfulness is a judgment call, so it's bounded and audited.** The
   judge doesn't get to output a bare "8/10." It has to decompose the
   answer into atomic claims and, for each one, either point at the exact
   retrieved chunk that supports it or mark it unsupported. The
   faithfulness score is *derived* from that audit trail, not invented
   directly — so you can always see *why* a run scored what it scored.

Every run — retrieval, generation, judging — is captured as a **trace**
(named spans with latency) and persisted to SQLite, so you can pull up any
past run and see exactly what happened, and diff two pipeline versions
side by side before you ship either one.

No API key required to try it: if `ANTHROPIC_API_KEY` isn't set, the
faithfulness judge falls back to a lexical-overlap heuristic so the whole
pipeline still runs end-to-end, offline, for free.

---

## Why this exists

I evaluate retrieval pipelines for a living, informally — I built retrieval
evaluation (BM25 vs. SBERT vs. hybrid, scored on Recall/MRR/nDCG/MAP) for a
legal RAG system over the Constitution of Nepal, and I'm currently building
an automated grading pipeline at Fusemachines. Anchor is that muscle,
generalized: a tool that plugs into *any* RAG pipeline's I/O, not just the
ones I happened to build.

## Architecture

```
your pipeline (any RAG system)
        │
        │  (query) → (retrieved chunks, answer)
        ▼
┌───────────────────────────────────────────────┐
│                    Anchor                      │
│                                                 │
│  retrieval_metrics.py   →  Recall@k / MRR /    │
│  (pure functions,           nDCG / MAP         │
│   no model calls)                              │
│                                                 │
│  judge.py                →  claim-level        │
│  (LLM or heuristic)          faithfulness audit│
│                                                 │
│  tracer.py               →  spans + runs       │
│  (SQLite)                    persisted          │
│                                                 │
│  runner.py                →  orchestrates all  │
│                                three per case    │
└───────────────────────────────────────────────┘
        │
        ▼
  anchor_traces.db  ──►  dashboard/app.py (Streamlit)
                          compare.py (CLI diff between versions)
```

## Quickstart

```bash
git clone https://github.com/arnabmanandhar/anchor.git
cd anchor
pip install -r requirements.txt
pip install -e . --no-deps        # adds the `anchor` command

# Runs two toy pipelines (TF-IDF vs. a weaker keyword retriever) against an
# 8-question sample dataset. Works with zero API keys (heuristic judge).
anchor demo

# Which version is better, and on which questions?
anchor compare demo-tfidf demo-keyword

# View everything in the dashboard
streamlit run dashboard/app.py
```

`python -m anchor <command>` works too if you skip the editable install.

Optional: use the real LLM faithfulness judge instead of the heuristic
fallback:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
anchor demo
```

The judge defaults to `claude-sonnet-5-5`. Set `ANCHOR_JUDGE_MODEL` to use
a different model. It uses structured outputs, so the audit always parses,
and any claim citing a chunk that wasn't actually retrieved is downgraded
to unsupported.

Run the test suite (no key needed; CI runs it the same way):

```bash
pytest -q
```

## Using it on your own pipeline

Anchor only needs your pipeline to speak one contract: given a query,
return ranked chunks and a generated answer.

```python
from anchor import evaluate_pipeline, summarize, Chunk

def my_pipeline(query: str):
    chunks = my_retriever.search(query)          # -> list[Chunk]
    answer = my_llm.generate(query, chunks)       # -> str
    return chunks, answer

dataset = [
    {"query": "...", "reference_chunk_ids": ["doc_12", "doc_47"]},
    # reference_chunk_ids is optional — omit it to skip retrieval
    # metrics and only run faithfulness scoring
]

results = evaluate_pipeline(my_pipeline, dataset, pipeline_version="v3-hybrid-retrieval")
print(summarize(results))
```

Then compare two versions once both have runs in the same database:

```bash
anchor compare v2-bm25 v3-hybrid-retrieval
```

The comparison reports mean deltas and lists the individual queries that
regressed or improved. A flat mean can hide a regression on one question
that another question's gain cancels out.

Already logging your pipeline's outputs? Score the log directly, in the
same format the dashboard's upload tab accepts:

```bash
anchor score outputs.jsonl --version prod-2026-10-11
```

Or run it from the command line. `my_rag:pipeline` is any function that
takes a query and returns `(chunks, answer)`. Chunks can be `Chunk` objects
or plain `{"id", "text"}` dicts:

```bash
anchor run my_rag:pipeline eval_set.json --version v3-hybrid-retrieval -k 5
```

## Using it as a CI regression gate

`anchor gate` exits non-zero when a version misses an absolute floor or
drops too far below a baseline, so a pipeline change that makes retrieval
worse fails the build:

```bash
anchor gate v3-hybrid-retrieval \
  --min-recall 0.9 --min-faithfulness 0.8 \
  --baseline v2-bm25 --max-mrr-drop 0.02 --max-faithfulness-drop 0.05
```

Metrics: `faithfulness`, `recall`, `mrr`, `ndcg`, `map`. With the heuristic
judge (no key), every number is deterministic, so the gate never flakes.
See `.github/workflows/ci.yml` for a working example.

## The dashboard

```bash
streamlit run dashboard/app.py
```

You can run evaluations in the browser, not only view CLI runs:

- **Evaluate** offers three ways in:
  - **Build & run a pipeline.** Load a corpus (sample, upload, or paste),
    edit the questions in a table, and pick a retriever (BM25, TF-IDF,
    keyword or hybrid), k, a generator (extractive or Claude) and a judge.
    Then run. One checkbox runs every retriever for a quick comparison.
  - **Upload your pipeline's outputs.** Use a JSONL of the queries, ranked
    chunks and answers your own system produced. Anchor scores them
    without calling your pipeline.
  - **Score a single answer.** Paste chunks and an answer to see the
    metrics and the claim-by-claim audit.
- **Results** shows per-version means on a fixed 0–1 axis, per-question
  bars, every run, and CSV and JSON downloads.
- **Compare** shows headline deltas, which questions regressed, and an
  interactive regression gate that prints the matching `anchor gate`
  command.
- **Inspect** shows one run in full: the answer, each claim with the chunk
  it cites, the ranked chunks marked relevant or cited, and timed trace
  spans.
- **Learn the metrics** gives every formula with a worked example.

The dashboard can be hosted publicly with no key leakage. Visitors can
paste their own Anthropic key for the session, and each one gets a private
workspace. See [DEPLOYMENT.md](DEPLOYMENT.md).

## Documentation

- [User guide](docs/USER_GUIDE.md): dashboard walkthrough, CLI, Python
  API, file formats, configuration
- [Metrics](docs/METRICS.md): what each score measures, the formulas, and
  worked examples
- [Architecture](docs/ARCHITECTURE.md): modules, data model, design
  decisions
- [Deployment](DEPLOYMENT.md): secure public hosting on Streamlit
  Community Cloud

## What's deliberately out of scope

- No hosted service, no auth, no multi-tenant anything — this is a library
  + a local dashboard, on purpose, so it's auditable and cheap to run.
- No generation-side cost accounting. The judge's own cost comes from
  real token usage at list prices, but your pipeline's LLM calls aren't
  metered.
- The heuristic judge is intentionally crude (lexical overlap). It exists
  so the project runs with zero setup, not to replace the LLM judge.

## Project layout

```
anchor/
  schema.py             Pydantic models: Chunk, RunInput, JudgeVerdict, RunResult
  retrieval_metrics.py  Recall@k, MRR, nDCG, MAP — pure functions
  judge.py              LLM faithfulness judge + offline heuristic fallback
  tracer.py             SQLite span/run storage (auto-upgrades older DBs)
  runner.py             One evaluation path for pipelines and logged outputs
  records.py            Load/validate logged outputs (JSON / JSONL / CSV)
  sandbox.py            Built-in toy retrievers + generators for the dashboard
  compare.py            Diffs two pipeline versions, overall and per query
  gate.py               Pass/fail thresholds for CI
  cli.py                `anchor demo | run | score | compare | gate`
dashboard/
  app.py                Streamlit entry point + navigation
  views.py              Home, Evaluate, Results, Compare, Inspect, Learn
  ui.py                 Styles, score cards, fixed 0–1 charts
docs/                   User guide, metrics derivations, architecture
examples/               Sample corpus, questions, logged records, toy pipelines
tests/                  Metrics, judge, sandbox, records, gate, dashboard (60 tests, offline)
DEPLOYMENT.md           Secure public deploy on Streamlit Community Cloud
```
