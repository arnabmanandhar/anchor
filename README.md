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

## Live dashboard

The Streamlit dashboard has four views. **Overview** shows per-version
means. **Compare versions** shows deltas and per-query changes. **Inspect
a run** shows the answer, the claim-by-claim audit, ranked chunks marked
relevant or cited, and trace spans. **Judge playground** lets you paste
chunks and an answer and watch the audit.

It can be hosted publicly with no key leakage. Visitors can paste their
own Anthropic key for the session, and otherwise get the offline judge.
See [DEPLOYMENT.md](DEPLOYMENT.md).

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
  runner.py             Orchestrates metrics + judge + tracing per case
  compare.py            Diffs two pipeline versions, overall and per query
  gate.py               Pass/fail thresholds for CI
  cli.py                `anchor demo | run | compare | gate`
dashboard/app.py        Streamlit UI over the trace database
examples/               Toy TF-IDF / keyword pipelines + 8-question sample dataset
tests/                  Hand-computed metric values, judge, tracer, gate, dashboard
DEPLOYMENT.md           Secure public deploy on Streamlit Community Cloud
```
