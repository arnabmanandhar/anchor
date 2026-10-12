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

On Windows PowerShell the commands are the same. The test suite clears
provider keys itself, so a key in your environment never reaches a test.

---

## 2. The dashboard

```bash
streamlit run dashboard/app.py
```

The browser opens on **Home**. If the trace database is empty, it is seeded
with the bundled demo so every page has something to show. The sidebar is
present on every page and holds:

- **Models (optional).** Click **Add a model**, pick a provider (Google
  Gemini, Groq, OpenRouter, Hugging Face, Mistral, OpenAI, xAI Grok,
  Anthropic, or a custom OpenAI-compatible endpoint such as a local
  Ollama), pick or type a model id, and paste your key. Providers marked
  *free tier* cost nothing to try, and a link under the form goes to the
  provider's key page. You can add several models, for example a free
  Gemini key for judging and an OpenAI key for generating, and choose one
  per task wherever a page offers a model. **Test** sends one tiny request
  to check the key and model; **Remove** forgets that key at once.

  Keys live only in this browser session's server memory. They are never
  written to disk, logs, the trace database or the page; the card shows
  only the last four characters. They are forgotten after 60 idle minutes
  (`ANCHOR_KEY_TTL_MINUTES`), when you close the tab, or when you click
  **Forget all keys**. If the owner configured one, a *built-in* model
  appears too, with a small call allowance per session.
- **Workspace.** Where runs are stored. Locally this is `anchor_traces.db`,
  shared with the CLI. On a public deploy each visitor gets a private
  workspace (see [DEPLOYMENT.md](../DEPLOYMENT.md)).

### Evaluate: run evaluations from the browser

There are three tabs, one for each situation.

#### Build and run a pipeline

Assemble a small RAG pipeline from built-in parts and evaluate it.

1. **Corpus.** Use the sample (20 chunks about the Constitution of Nepal),
   upload a file, or paste text.
   - CSV with `id,text` columns
   - JSON list of `{"id": ..., "text": ...}` (or plain strings)
   - JSONL, one `{"id", "text"}` per line
   - Plain text or Markdown, with chunks separated by blank lines (ids
     become `c1`, `c2`, …)

   Open **Preview corpus** to check how it was split.
2. **Questions.** Edit the table directly: add rows with the + at the
   bottom, delete with the row checkbox. You can also load a CSV, JSON or
   JSONL file of `{query, reference_chunk_ids}`. *Reference chunk ids* are
   the ground truth: the ids of chunks that should be retrieved, comma
   separated. They are optional. Without them only faithfulness is scored.
   Anchor warns you if an id isn't in the corpus.

   **No questions yet?** Open **Generate questions from this corpus**.
   Choose how many, which types (single-hop, multi-hop, unanswerable) and
   the generator (offline templates, or any model you added), then click
   **Generate questions**. Each question comes with its reference ids and
   a difficulty label. Untick or edit rows, then **Replace** the question
   table with them, **Add** them to it, or **Download** them as JSON.
   *Skip easy questions* drops ones keyword overlap already gets right.
   See [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md) for how it works.
3. **Pipeline.**
   - **Retriever:** BM25, TF-IDF, keyword overlap (deliberately weak), or
     hybrid (BM25 and TF-IDF fused with reciprocal-rank fusion). All
     built-in retrievers are offline and deterministic.
   - **Chunks retrieved (k):** how many chunks go to the generator. It is
     also the cutoff for Recall@k and nDCG@k.
   - **Answer generator:** *Extractive* copies the best-matching sentences
     from the retrieved chunks. It is offline, and faithful by
     construction. Pick one of your models instead to have it write an
     answer from the chunks. The offline option is the default here
     because every question costs a call.
   - **Faithfulness judge:** *Heuristic* (offline) or any model you added.
     A model is preselected once you add one.
   - **Version name:** the label runs are stored under. Reusing a name
     replaces that version's runs.
   - **Also run every other retriever** evaluates all four retrievers with
     the same settings in one click, ready for Compare.
4. Click **Run evaluation.** A progress bar runs, then the mean scores
   appear with links to Results and Compare.

LLM runs are capped at 25 cases per click (`ANCHOR_MAX_LLM_CASES`), so a
mistaken click can't run up a bill.

#### Upload your pipeline's outputs

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

#### Score a single answer

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

Pick a baseline (A) and a candidate (B), and a significance level (0.05
by default). The page defaults to the two versions that share the most
questions and differ the most.

- **Headline deltas** show B's value and B − A for each metric. Under each
  delta, *significant* means a paired test says the change is unlikely to
  be chance; *within noise* means it could be.
- **Is the difference real?** plots every delta with its confidence
  interval on a fixed −1 to +1 axis. An interval that crosses 0 could be
  chance. The table lists the p-values and a verdict: *real improvement*,
  *real regression*, *could be chance* or *no change*. When nothing is
  significant, a note says the smallest change your question set could
  have detected. If that number is large, add questions.
- **What changed, question by question** matches runs by question text and
  labels each one *regressed*, *mixed*, *improved* or *unchanged*. Use this
  to catch a regression that the mean hides.
- **Regression gate:** set floors and an allowed drop with sliders, see
  PASS or FAIL, and copy the matching `anchor gate` command for CI. Turn on
  **Only fail on significant drops** to ignore drops the paired test can't
  call real.

### Inspect

Choose a version and a question to see one run in full:

- the score cards
- the generated answer
- the **claim-level audit**: each claim marked *Supported* with the chunk
  it cites, or *Unsupported* with the reason
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
anchor compare v2 v3                           # means, deltas, regressed questions, significance
anchor gate v3 --min-recall 0.9 --baseline v2 --max-faithfulness-drop 0.05
anchor gate v3 --baseline v2 --max-recall-drop 0.02 --require-significant
anchor synth corpus.json -n 30 -o questions.json  # generate questions with reference ids
```

Every command takes `--db PATH` (default `anchor_traces.db`). `run` and
`score` take `--judge auto|llm|heuristic`; `auto` uses an LLM if any
provider key is set. `demo`, `run`, `score` and `synth` take
`--provider`, `--model` and `--base-url` (for `--provider custom`).
Keys are read from the environment only, never from a flag, so they
don't end up in shell history. `run` and `score` take `--workers N` to
judge N cases in parallel.

```bash
anchor providers               # every provider, its default model, and whether its key is set
anchor providers --check       # also send one tiny request per configured key
GROQ_API_KEY=... anchor run my_rag:pipeline q.json -v v3 --provider groq --model llama-3.3-70b-versatile
anchor score out.jsonl -v local --provider custom --base-url http://localhost:11434/v1 --model llama3.2
```

- **`anchor run`** imports `module:function`, a function taking a question
  and returning `(chunks, answer)`. Add `--factory` if the function builds
  and returns the pipeline instead.
- **`anchor synth`** reads a corpus in any format the dashboard accepts
  and writes a question set. `--generator auto|template|llm` (auto uses
  an LLM when a provider key is set), `--mix single_hop=6,multi_hop=2,unanswerable=2`,
  `--drop-easy`, `--seed`. A one-line summary goes to stderr.
- **`anchor compare`** prints JSON. Its `significance` block has, per
  metric, `n` (shared questions), `mean_delta`, `ci_low`/`ci_high`,
  `p_value`, `significant` and `mde` (smallest detectable change).
  `--alpha` sets the level.
- **`anchor gate`** exits with code 1 on failure, so it can fail a CI job.
  Available metrics are `faithfulness`, `recall`, `precision`, `hit`, `mrr`, `ndcg` and `map`,
  each with `--min-X` and `--max-X-drop`. `--require-significant` lets a
  drop beyond `--max-X-drop` pass when it isn't significant at `--alpha`.

## 4. Python

```python
from anchor import evaluate_pipeline, evaluate_records, summarize, Chunk

def my_pipeline(query: str):
    chunks = my_retriever.search(query)           # list[Chunk] or [{"id", "text"}]
    answer = my_llm.generate(query, chunks)
    return chunks, answer

results = evaluate_pipeline(my_pipeline, dataset, pipeline_version="v3", k=5)
print(summarize(results))

# choose the judge's provider and model explicitly (key from the environment or passed in)
import os
from anchor import Connection, FaithfulnessJudge, Secret
judge = FaithfulnessJudge(Connection("groq", "llama-3.3-70b-versatile", Secret(os.environ["GROQ_API_KEY"])))
results = evaluate_pipeline(my_pipeline, dataset, pipeline_version="v3", judge=judge, max_workers=4)

# generate questions from your chunks
from anchor.synth import generate_questions
dataset = generate_questions(my_chunks, n=30, generator="template")   # or "llm" with connection=...

# is v3 really better than v2?
from anchor import compare_versions
sig = compare_versions("anchor_traces.db", "v2", "v3")["significance"]
print(sig["recall_at_k"]["p_value"], sig["recall_at_k"]["ci_low"], sig["recall_at_k"]["ci_high"])

# or, from logs:
from anchor import load_records
records = load_records(open("outputs.jsonl", "rb").read(), "outputs.jsonl")
evaluate_records(records, pipeline_version="prod-logs")
```

Then open the dashboard to inspect the runs.

## 5. Configuration

| Variable | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`, `MISTRAL_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, `ANTHROPIC_API_KEY` | unset | Provider keys. With none set everything runs offline. In the dashboard, a key set on the server becomes the capped built-in model. |
| `ANCHOR_LLM_PROVIDER` | first provider with a key, in the order above | Provider for the CLI and Python when none is passed |
| `ANCHOR_LLM_MODEL` | the provider's default model | Model id (`ANCHOR_JUDGE_MODEL` is accepted as an alias) |
| `ANCHOR_LLM_API_KEY` | unset | Key used with `ANCHOR_LLM_PROVIDER`, or for a custom endpoint |
| `ANCHOR_LLM_BASE_URL` | unset | Endpoint for the custom provider |
| `ANCHOR_LLM_TIMEOUT` | `60` | Seconds per provider request |
| `ANCHOR_LLM_WORKERS` | `4` | Parallel LLM calls per dashboard run |
| `ANCHOR_DB` | `anchor_traces.db` in the repo | Dashboard workspace path (local mode) |
| `ANCHOR_PUBLIC` | unset | `1` gives every dashboard session a private, demo-seeded workspace and refuses private endpoints |
| `ANCHOR_SERVER_PROVIDER`, `ANCHOR_SERVER_MODEL` | first provider with a key, its default model | The dashboard's built-in model |
| `ANCHOR_SERVER_KEY_SESSION_LIMIT` | `25` | Built-in model calls per browser session |
| `ANCHOR_SERVER_KEY_DAILY_LIMIT` | `500` | Built-in model calls per day across all sessions |
| `ANCHOR_KEY_TTL_MINUTES` | `60` | Idle minutes before a session's keys are forgotten |
| `ANCHOR_MAX_LLM_CASES` | `25` | Most LLM cases one dashboard click may run |

## 6. Troubleshooting

- **"LLM judge failed (auth); used the heuristic instead."** The key is
  wrong or revoked. The run still completed, using the heuristic. Other
  reasons: `rate_limit` (free tier exhausted for now), `not_found` (the
  model id isn't served by that provider), `network`, `parse` (the model
  didn't return valid JSON even after one repair attempt). Click **Test**
  on the model card, or run `anchor providers --check`.
- **Retrieval columns are blank.** Those questions have no reference chunk
  ids, so only faithfulness was scored.
- **Every change says "could be chance".** Your question set is too
  small to detect changes that size. The note under the table says how
  big a change it can detect; add questions until that is smaller than
  the changes you care about.
- **Compare says no questions in common.** The two versions were run on
  different question sets. Questions are matched by exact text.
- **Scores look identical across retrievers.** On a small corpus where the
  answer words appear in only one chunk, every lexical retriever finds it.
  Try the keyword retriever, a smaller k, or your own corpus.
