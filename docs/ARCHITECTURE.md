# Architecture

```
            your pipeline  ──or──  logged outputs (JSONL)  ──or──  dashboard sandbox
                 │                         │                          │
      evaluate_pipeline()          evaluate_records()          sandbox.make_pipeline()
                 └──────────────┬──────────┘                          │
                                ▼                                     │
                     runner._evaluate_case()  ◄────────────────────────┘
                     ├─ retrieval_metrics   (pure functions)
                     ├─ judge               (LLM or heuristic)
                     └─ tracer              (SQLite: runs + spans)
                                │
                 ┌──────────────┼───────────────┐
                 ▼              ▼               ▼
           compare.py        gate.py       dashboard/app.py
```

## Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `schema.py` | Pydantic models: `Chunk`, `ClaimVerdict`, `JudgeVerdict`, `RetrievalMetrics`, `RunResult` | pydantic |
| `retrieval_metrics.py` | Recall@k, MRR, nDCG@k, MAP as pure functions of `(ranked_ids, relevant_ids, k)` | — |
| `judge.py` | `FaithfulnessJudge`: LLM backend (structured outputs, citation enforcement, token cost) and heuristic backend (overlap coefficient). Never stores or prints the key. | anthropic (optional) |
| `tracer.py` | SQLite `runs` and `spans` tables. Upgrades older databases in place. | sqlite3 |
| `runner.py` | `_evaluate_case` is the single code path. `evaluate_pipeline` calls your function; `evaluate_records` replays logged outputs. | the four above |
| `records.py` | Parses and validates logged outputs (JSON/JSONL/CSV, common field aliases) | — |
| `sandbox.py` | Toy RAG parts for the dashboard: corpus parsing, BM25 / TF-IDF / keyword / hybrid retrievers, extractive or Claude generators. Sample input, not part of evaluation. | — |
| `compare.py` | Aggregate means, plus a per-question diff matched by query text | tracer |
| `gate.py` | Floors and allowed drops, returning a pass/fail result | compare |
| `cli.py` | `anchor demo · run · score · compare · gate` | all |
| `dashboard/app.py`, `dashboard/ui.py` | Streamlit pages; styles and charts | all, streamlit, altair |

## Design decisions

- **Retrieval-agnostic.** Anchor never ranks anything for you. The contract
  is just `(ranked chunks, answer)` per question, so any stack fits. The
  sandbox retrievers exist only so the dashboard is useful without code.
- **Deterministic first.** Retrieval scores are pure functions. The
  heuristic judge is exact too. CI runs with no key, so the gate never
  flakes.
- **Derived, auditable faithfulness.** The judge never outputs a score.
  It outputs claims and citations, and the score is computed from them.
  Citations of chunks that weren't retrieved are rejected.
- **One evaluation path.** Pipelines, logs and the sandbox all go through
  `_evaluate_case`, so the numbers are comparable whatever the source.
- **SQLite, no server.** Traces are one file, easy to inspect, copy or
  delete. The dashboard reads the same file the CLI writes.
- **Safe to host publicly.** Keys stay in the session. Errors are reduced
  to their type. The database path is fixed on the server. Public mode
  gives each visitor a private database. See [DEPLOYMENT.md](../DEPLOYMENT.md).

## Data model (SQLite)

`runs`: `run_id`, `pipeline_version`, `query`, `generated_answer`,
`retrieval_json` (metrics + k), `judge_json` (score, claims, backend, model,
tokens), `retrieved_json` (ranked chunks), `reference_json` (ground-truth
ids), `latency_ms`, `cost_usd`, `created_at`.

`spans`: `run_id`, `name`, `duration_ms`, `metadata_json`, `created_at`.

## Testing

`pytest -q` runs everything offline:

- `test_metrics.py`: hand-computed values for every metric
- `test_judge.py`: no network without a key, citation enforcement, no key
  leakage on failure
- `test_tracer_compare_gate.py`: storage, upgrades of old databases,
  per-question regressions, gate exit codes
- `test_sandbox_records.py`: retrievers, corpus, dataset and record parsing
- `test_dashboard.py`: every page renders headlessly, the evaluate flows
  run, and keys never appear on the page
