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
| `llm.py` | Provider-neutral client over the standard library: a registry of providers (OpenAI-compatible chat completions and the Anthropic messages protocol), `Secret` key wrapper, error redaction, retries with backoff, structured-output negotiation (JSON schema → JSON mode → prompt), reply validation, endpoint guard | — |
| `judge.py` | `FaithfulnessJudge`: LLM backend (three-way verdicts, citation enforcement, repair retry, verdict cache) and heuristic backend (stemmed overlap, number check, abstention). Never stores or prints the key. | llm |
| `tracer.py` | SQLite `runs`, `spans` and `judge_cache` tables, WAL mode, thread-safe writes. Upgrades older databases in place. | sqlite3 |
| `runner.py` | `_evaluate_case` is the single code path. `evaluate_pipeline` calls your function; `evaluate_records` replays logged outputs. Optional thread pool (`max_workers`) keeps result order. | the four above |
| `records.py` | Parses and validates logged outputs (JSON/JSONL/CSV, common field aliases) | — |
| `sandbox.py` | Toy RAG parts for the dashboard: corpus parsing, BM25 / TF-IDF / keyword / hybrid retrievers, extractive or LLM generators. Sample input, not part of evaluation. | llm |
| `compare.py` | Aggregate means, a per-question diff matched by query text, and per-metric significance | tracer, stats |
| `synth.py` | Question generation (templates or any LLM) with reference ids known by construction; de-duplication and difficulty labels | sandbox, llm |
| `stats.py` | Paired bootstrap CIs, sign-flip permutation test, minimum detectable effect. Pure Python, seeded. | — |
| `gate.py` | Floors and allowed drops (optionally only significant drops), returning a pass/fail result | compare |
| `cli.py` | `anchor demo · run · score · compare · gate · synth · providers` | all |
| `dashboard/connections.py` | Per-session model connections, the capped built-in model, key expiry. All dashboard key handling is here. | llm, streamlit |
| `dashboard/app.py`, `views.py`, `ui.py` | Streamlit navigation, pages, styles and charts | all, streamlit, altair |

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
- **Statistics, not vibes.** A mean delta comes with a confidence interval
  and a p-value from a paired test, so "v3 is better" is a claim Anchor can
  back up. Tests are seeded and the permutation test is exact on small
  sets, so these numbers are as reproducible as the metrics.
- **Ground truth by construction.** Generated questions carry the ids of
  the chunks they were written from, and every generated set passes the
  same deterministic checks (duplicates, leaked answers, difficulty),
  whichever generator wrote it.
- **SQLite, no server.** Traces are one file, easy to inspect, copy or
  delete. The dashboard reads the same file the CLI writes.
- **Any model, no SDKs.** Every provider is reached over plain HTTPS with
  the standard library, so there are no vendor SDKs to pin, and adding a
  provider is one registry entry. Most providers speak the OpenAI chat
  completions protocol; Anthropic's messages protocol is the one other
  shape. Free tiers come first in the registry, so the default costs
  nothing.
- **Safe to host publicly.** Keys are wrapped so they can't be printed or
  pickled, stay in the browser session's memory, and expire when idle.
  Provider errors are scrubbed and reduced to a kind. The database path is
  fixed on the server. Public mode gives each visitor a private database
  and refuses private endpoints. See [DEPLOYMENT.md](../DEPLOYMENT.md).

## Data model (SQLite)

`runs`: `run_id`, `pipeline_version`, `query`, `generated_answer`,
`retrieval_json` (metrics + k), `judge_json` (score, claims, backend, model,
tokens), `retrieved_json` (ranked chunks), `reference_json` (ground-truth
ids), `latency_ms`, `cost_usd`, `created_at`.

`spans`: `run_id`, `name`, `duration_ms`, `metadata_json`, `created_at`.

## Testing

`pytest -q` runs everything offline:

- `test_metrics.py`: hand-computed values for every metric
- `test_llm.py`: key masking, redaction, error classification, retries,
  both protocols, the endpoint guard, provider selection
- `test_judge.py`: no network without a key, every provider backing the
  judge, schema fallback, repair retry, citation enforcement, caching, no
  key leakage on failure
- `test_repo_hygiene.py`: no key-shaped strings in tracked files
- `test_tracer_compare_gate.py`: storage, upgrades of old databases,
  per-question regressions, gate exit codes
- `test_synth.py`: each template rule, ground truth, unanswerable topics
  absent from the corpus, determinism, de-duplication, difficulty, the
  LLM path with the HTTP layer replaced by a fake, and the CLI round trip
- `test_stats.py`: hand-computed p-values, CIs and detectable effect;
  determinism; the gate ignoring noise but not real drops
- `test_sandbox_records.py`: retrievers, corpus, dataset and record parsing
- `test_dashboard.py`: every page renders headlessly, the evaluate flows
  run, keys never appear on the page or in session state, connections are
  per session, and the built-in model is capped
