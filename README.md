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
git clone <this-repo>
cd anchor
pip install -r requirements.txt

# Runs a toy TF-IDF pipeline against a 5-question sample dataset.
# Works with zero API keys (uses the heuristic judge fallback).
python -m anchor.cli demo

# View results in a dashboard
streamlit run dashboard/app.py
```

Optional — use the real LLM-based faithfulness judge instead of the
heuristic fallback:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m anchor.cli demo
```

Run the deterministic metrics test suite:

```bash
pytest tests/ -v
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
python -m anchor.cli compare v2-bm25 v3-hybrid-retrieval
```

## What's deliberately out of scope (v0.1)

- No hosted service, no auth, no multi-tenant anything — this is a library
  + a local dashboard, on purpose, so it's auditable and cheap to run.
- No generation-side cost/token accounting beyond a rough per-call
  estimate — real token counts from your LLM provider would replace this
  in a production version.
- The heuristic judge is intentionally crude (lexical overlap). It exists
  so the project runs with zero setup, not to replace the LLM judge.

## Project layout

```
anchor/
  schema.py             Pydantic models: Chunk, RunInput, JudgeVerdict, RunResult
  retrieval_metrics.py  Recall@k, MRR, nDCG, MAP — pure functions
  judge.py              LLM faithfulness judge + offline heuristic fallback
  tracer.py             SQLite span/run storage
  runner.py             Orchestrates metrics + judge + tracing per case
  compare.py            Diffs two pipeline_version labels
  cli.py                `python -m anchor.cli demo|compare`
dashboard/app.py        Streamlit UI over the trace database
examples/                Toy TF-IDF pipeline + 5-question sample dataset
tests/test_metrics.py    Hand-computed expected values for every metric
```
