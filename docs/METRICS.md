# How Anchor scores a RAG pipeline

Anchor splits evaluation into two tracks that fail for different reasons.
That lets you tell a **retrieval failure** apart from a **generation failure**:

- A **retrieval failure** means the right chunk never reached the model. Low
  recall and a low MRR show this. Faithfulness can still be high, because the
  model may stick to whatever it was given.
- A **generation failure** means the right chunk was retrieved but the
  answer says something it doesn't support. Retrieval scores look fine and
  faithfulness drops.

Every score below lies in $[0, 1]$, where 1 is perfect, so every chart in
the dashboard uses a fixed 0–1 axis.

All examples use the same case: a query whose relevant chunks are
$R = \{a, b\}$, and a retriever that returned the ranking
`x, a, y, b, z` (best first).

---

## Retrieval metrics: pure functions

These need ground truth, the `reference_chunk_ids` for each question. They
call no model and use no randomness, so the same ranking always gives the
same number. That is what makes them safe as a CI gate. They are computed
in `anchor/retrieval_metrics.py`, and `tests/test_metrics.py` checks each one
against a hand-computed value.

### Recall@k: did we find the relevant chunks?

$$
\text{Recall@}k = \frac{|\,\text{top-}k \cap R\,|}{|R|}
$$

This is the share of relevant chunks that appear anywhere in the first $k$
results. It ignores order inside the top $k$.

*Example:* with $k=3$, the top 3 are `x, a, y`, which contain one of the
two relevant chunks, so Recall@3 = 1/2 = **0.50**. With $k=5$, both
relevant chunks are found, so Recall@5 = **1.00**.

**Use it when** the generator reads all $k$ chunks equally, so what matters
is whether the evidence is in the context at all.

### MRR: how soon does the first relevant chunk appear?

$$
\text{RR} = \frac{1}{\text{rank of the first relevant chunk}}, \qquad
\text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \text{RR}_q
$$

*Example:* the first relevant chunk, `a`, is at rank 2, so RR = **0.50**.

**Use it when** only the top answer matters, such as single-chunk QA or a
"best source" link. It ignores every relevant chunk after the first.

### nDCG@k: is the ranking well ordered?

Each relevant chunk earns a gain discounted by its position:

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i+1)}, \qquad
\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

$\text{rel}_i$ is 1 if the chunk at rank $i$ is relevant and 0 otherwise.
IDCG is the DCG of the best possible ranking, with every relevant chunk
first. Dividing by IDCG normalizes the score, so 1.0 always means a
perfect order.

*Example (k=5):* relevant chunks sit at ranks 2 and 4.

- DCG = $1/\log_2 3 + 1/\log_2 5 = 0.631 + 0.431 = 1.062$
- The ideal order puts them at ranks 1 and 2: IDCG = $1/\log_2 2 + 1/\log_2 3 = 1 + 0.631 = 1.631$
- nDCG@5 = 1.062 / 1.631 = **0.65**

The $\log_2(i+1)$ discount is gentle. Rank 2 is worth 63% of rank 1, and
rank 10 is still worth 29%. That fits a reader who skims down a list.

### MAP: precision at every hit

$$
\text{AP} = \frac{1}{|R|} \sum_{i:\,\text{rel}_i = 1} \text{Precision@}i, \qquad
\text{MAP} = \frac{1}{|Q|}\sum_{q} \text{AP}_q
$$

*Example:* the hits are at rank 2, where precision is 1/2, and rank 4,
where precision is 2/4. AP = (0.5 + 0.5) / 2 = **0.50**.

The sum is divided by $|R|$, not by the number of hits. So a relevant chunk
that is never retrieved counts as zero precision, and MAP penalizes misses.

### Which one should I watch?

| Symptom | Metric that moves |
|---|---|
| Evidence missing from the context | Recall@k drops |
| Evidence present but buried | nDCG@k and MAP drop, recall stays |
| The single best source got worse | MRR drops |

In the bundled demo, the keyword retriever loses Recall@3 on *"What rights
do citizens of Nepal have?"* (it finds one of two relevant chunks) while
its MRR stays at 1.0. That is why Anchor's compare view works per question
and doesn't only compare means.

---

## Faithfulness: a bounded, audited judgment

Whether an answer is supported by its context can't be computed exactly,
so this is the one place Anchor uses judgment. The judgment is constrained
in three ways so the number stays auditable:

1. **The answer is split into atomic claims.**
2. **Each claim must cite one retrieved chunk id that supports it, or be
   marked unsupported.** The LLM judge returns JSON constrained to this
   schema, so it can't skip the step.
3. **The score is derived, never asked for:**

$$
\text{faithfulness} = \frac{\#\,\text{supported claims}}{\#\,\text{claims}}
$$

A judge saying "8/10" can't be checked. A judge saying "claim 2 is
supported by chunk `art-084`" can be checked by opening the chunk.

**Citation enforcement:** if the LLM cites a chunk id that wasn't
actually retrieved, Anchor downgrades that claim to unsupported. A made-up
citation can't inflate the score.

### The two judge backends

| | LLM judge | Heuristic judge |
|---|---|---|
| Needs | An Anthropic API key | Nothing |
| Claims | Atomic facts extracted by the model | Sentences of the answer |
| Support test | The model reads the chunk | Word overlap ≥ 0.5 (below) |
| Catches paraphrase | Yes | No |
| Deterministic | Nearly (structured output) | Exactly |
| Cost | Real token usage, shown per run | Free |

**The heuristic, exactly.** Let $C$ be the set of content words in a claim
sentence (lower-cased, stop-words removed) and $D$ the content words of a
chunk. The claim's support from that chunk is the share of the claim's
words the chunk covers:

$$
\text{overlap}(C, D) = \frac{|C \cap D|}{|C|}
$$

A claim is supported if some retrieved chunk reaches overlap ≥ 0.5, and the
chunk with the highest overlap is the one cited. The measure is
asymmetric on purpose: a long chunk isn't penalized for containing more
than the claim needs.

*Example:* the claim "Each province elects a governor." has content words
$C$ = {each, province, elects, governor}. The chunk "Nepal is divided into
seven provinces, each with its own provincial assembly…" shares only
"each". "province" ≠ "provinces", because there is no stemming. Overlap is
1/4 = 0.25 < 0.5, so the claim is **unsupported**.

The heuristic is deliberately crude. Its job is to make the whole pipeline
run offline and for free, for CI and first-time users, not to replace the
LLM judge. Use the LLM judge for numbers you'll report.

---

## Traces, latency and cost

Every evaluated case is stored as a run in SQLite (`anchor_traces.db`), with
timed spans:

- `retrieval+generation` is your pipeline, or `precomputed_output` for
  uploaded logs.
- `retrieval_metrics` is the deterministic scoring.
- `faithfulness_judge` is the judge call. Its metadata records which
  backend ran.

The judge's cost is computed from the token counts the API reports, at
list prices. Your pipeline's own LLM calls are not metered.

## The regression gate

`anchor gate` turns stored means into pass or fail. It checks two things:

- **Floors:** for example, mean Recall@k ≥ 0.9.
- **Allowed drop versus a baseline:** for example, mean faithfulness may
  not fall more than 0.05 below `main`.

With the heuristic judge, every input to the gate is deterministic, so a
red build always means something changed. The Compare page has the same
gate with sliders, and prints the matching CLI command.
