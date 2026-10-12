# Anchor user guide

Anchor can be used three ways, all writing to the same trace database:

| Interface | Good for |
|---|---|
| **Dashboard** (`streamlit run dashboard/app.py`) | Trying it out, evaluating without code, inspecting and comparing runs |
| **CLI** (`anchor …`) | Scripted runs, CI regression gates |
| **Python** (`from anchor import …`) | Plugging Anchor into an existing pipeline or notebook |

For what each score means and how it's computed, see [METRICS.md](METRICS.md).

---

## 1. Install

```bash
git clone https://github.com/arnabmanandhar/anchor.git
cd anchor
pip install -r requirements.txt
pip install -e . --no-deps      # adds the `anchor` command
```

Python 3.10 or newer is required. No API key is needed: everything runs
offline with the heuristic judge.

On Windows PowerShell the commands are the same. To make sure no key is
set during a test run, use `Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue`.

---

## 2. The dashboard

```bash
streamlit run dashboard/app.py
```

The browser opens on **Home**. If the trace database is empty, it is seeded
with the bundled demo so every page has something to show. The sidebar is
present on every page and holds:

- **Anthropic API key (optional).** Paste a key to use Claude as the
  faithfulness judge and as an answer generator. The key lives only in
  your browser session and is never written to disk, logs or the page.
  The status box under it says which judge is active.
- **Workspace.** Where runs are stored. Locally this is `anchor_traces.db`,
  shared with the CLI. On a public deploy each visitor gets a private
  workspace (see [DEPLOYMENT.md](../DEPLOYMENT.md)).

### Evaluate: run evaluations from the browser

There are three tabs, one for each situation.

#### 🧪 Build & run a pipeline

Assemble a small RAG pipeline from built-in parts and evaluate it.

1. **Corpus.** Use the sample (20 chunks about the Constitution of Nepal),
   upload a file, or paste text.
   - CSV with `id,text` columns
   - JSON list of `{"id": ..., "text": ...}` (or plain strings)
   - JSONL, one `{"id", "text"}` per line
   - Plain text or Markdown, with chunks separated by blank lines (ids
     become `c1`, `c2`, …)

   Open **Preview corpus** to check how it was split.
2. **Questions.** Edit the table directly: add rows with the ➕ at the
   bottom, delete with the row checkbox. You can also load a CSV, JSON or
   JSONL file of `{query, reference_chunk_ids}`. *Reference chunk ids* are
   the ground truth: the ids of chunks that should be retrieved, comma
   separated. They are optional. Without them only faithfulness is scored.
   Anchor warns you if an id isn't in the corpus.
3. **Pipeline.**
   - **Retriever:** BM25, TF-IDF, keyword overlap (deliberately weak), or
     hybrid (BM25 and TF-IDF fused with reciprocal-rank fusion). All
     built-in retrievers are offline and deterministic.
   - **Chunks retrieved (k):** how many chunks go to the generator. It is
     also the cutoff for Recall@k and nDCG@k.
   - **Answer generator:** *Extractive* copies the best-matching sentences
     from the retrieved chunks. It is offline, and faithful by
     construction. *Claude* writes an answer from the chunks and needs a key.
   - **Faithfulness judge:** *Heuristic* (offline) or *LLM* (Claude).
   - **Version name:** the label runs are stored under. Reusing a name
     replaces that version's runs.
   - **Also run every other retriever** evaluates all four retrievers with
     the same settings in one click, ready for Compare.
4. Click **Run evaluation.** A progress bar runs, then the mean scores
   appear with links to Results and Compare.

LLM runs are capped at 25 cases per click (`ANCHOR_MAX_LLM_CASES`), so a
mistaken click can't run up a bill.

#### 📤 Upload your pipeline's outputs

Use this to evaluate a pipeline you already have, built with LangChain,
LlamaIndex, raw API calls or anything else, without connecting it to
Anchor. Log one record per question and upload the file:

```json
{"query": "How many provinces does Nepal have?",
 "retrieved_chunks": [{"id": "c2", "text": "...seven provinces."}, {"id": "c6", "text": "..."}],
 "answer": "Nepal has seven provinces.",
 "reference_chunk_ids": ["c2"]}
```

- Accepted formats are **JSONL** (one object per line), a **JSON** list, or
  **CSV** with `retrieved_chunks` as a JSON array string.
- `retrieved_chunks` are ranked, best first. A chunk may be a bare string,
  in which case ids become `r1`, `r2`, … by rank. That's fine for
  faithfulness, but retrieval metrics need real ids.
- Aliases from other tools are accepted: `contexts`, `chunks` or
  `retrieved_contexts` for chunks; `response`, `generated_answer` or
  `output` for the answer; `relevant_ids` or `ground_truth_ids` for the
  references.
- **Download a sample file** shows a working example. It contains two
  deliberately hallucinated answers, a retrieval miss, and a relevant
  chunk ranked second.

Latency in these runs measures Anchor's scoring, not your pipeline.

#### ✍️ Score a single answer

Paste a question, the retrieved chunks in rank order, and an answer. You
get every metric plus the claim-by-claim audit next to the chunks. Use
this to understand the judge, or to debug one bad answer. Nothing is saved.

### Results

- **Mean scores by version:** grouped bars on a fixed 0–1 axis, so a 0.6 is
  60% of the way up and never a full bar. The table below the chart shows
  every mean as a 0–1 progress bar, plus runs, k, judge, mean latency and
  judge cost.
- **Per question:** pick a metric to see every question's score for each
  selected version.
- **All runs:** every run as a row. Download it as CSV, or the summary as
  JSON.
- **Delete a version** removes its runs from the workspace.

### Compare

Pick a baseline (A) and a candidate (B). The page defaults to the two
versions that share the most questions and differ the most.

- **Headline deltas** show B's value and B − A for each metric.
- **What changed, question by question** matches runs by question text and
  labels each one *regressed*, *mixed*, *improved* or *unchanged*. Use this
  to catch a regression that the mean hides.
- **Regression gate:** set floors and an allowed drop with sliders, see
  PASS or FAIL, and copy the matching `anchor gate` command for CI.

### Inspect

Choose a version and a question to see one run in full:

- the score cards
- the generated answer
- the **claim-level audit**: each claim marked ✓ with the chunk it cites,
  or ✗ with the reason
- the **retrieved chunks** in rank order, tagged *relevant* (ground truth)
  and *cited by judge*, plus any relevant chunks that weren't retrieved
- the **trace**: how long each step took

### Learn the metrics

This page renders [METRICS.md](METRICS.md): every formula with a worked
example.

---

## 3. The CLI

```bash
anchor demo                                   # two toy pipelines on 8 questions
anchor run my_rag:pipeline questions.json -v v3 -k 5
anchor score outputs.jsonl -v prod-2026-10-11  # score logged outputs (same format as the upload tab)
anchor compare v2 v3                           # means, deltas, regressed questions
anchor gate v3 --min-recall 0.9 --baseline v2 --max-faithfulness-drop 0.05
```

Every command takes `--db PATH` (default `anchor_traces.db`). `run` and
`score` take `--judge auto|llm|heuristic`. `auto` uses the LLM if
`ANTHROPIC_API_KEY` is set.

- **`anchor run`** imports `module:function`, a function taking a question
  and returning `(chunks, answer)`. Add `--factory` if the function builds
  and returns the pipeline instead.
- **`anchor gate`** exits with code 1 on failure, so it can fail a CI job.
  Available metrics are `faithfulness`, `recall`, `mrr`, `ndcg` and `map`,
  each with `--min-X` and `--max-X-drop`.

## 4. Python

```python
from anchor import evaluate_pipeline, evaluate_records, summarize, Chunk

def my_pipeline(query: str):
    chunks = my_retriever.search(query)           # list[Chunk] or [{"id", "text"}]
    answer = my_llm.generate(query, chunks)
    return chunks, answer

results = evaluate_pipeline(my_pipeline, dataset, pipeline_version="v3", k=5)
print(summarize(results))

# or, from logs:
from anchor import load_records
records = load_records(open("outputs.jsonl", "rb").read(), "outputs.jsonl")
evaluate_records(records, pipeline_version="prod-logs")
```

Then open the dashboard to inspect the runs.

## 5. Configuration

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables the LLM judge (CLI and Python). In the dashboard it is the server's fallback key. |
| `ANCHOR_JUDGE_MODEL` | `claude-sonnet-5-5` | Model for the LLM judge and the Claude generator |
| `ANCHOR_DB` | `anchor_traces.db` in the repo | Dashboard workspace path (local mode) |
| `ANCHOR_PUBLIC` | unset | `1` gives every dashboard session a private, demo-seeded workspace |
| `ANCHOR_SERVER_KEY_SESSION_LIMIT` | `5` | LLM calls per session allowed on the server's key |
| `ANCHOR_MAX_LLM_CASES` | `25` | Most LLM cases one dashboard click may run |

## 6. Troubleshooting

- **"LLM judge failed (AuthenticationError); used heuristic fallback."**
  The key is wrong or revoked. The run still completed, using the heuristic.
- **Retrieval columns are blank.** Those questions have no reference chunk
  ids, so only faithfulness was scored.
- **Compare says no questions in common.** The two versions were run on
  different question sets. Questions are matched by exact text.
- **Scores look identical across retrievers.** On a small corpus where the
  answer words appear in only one chunk, every lexical retriever finds it.
  Try the keyword retriever, a smaller k, or your own corpus.
