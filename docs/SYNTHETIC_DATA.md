# Synthetic test sets

Retrieval metrics need ground truth: for each question, the ids of the
chunks that answer it. Labelling that by hand is the slowest part of
setting up an evaluation. Anchor removes it: if a question is *written
from* specific chunks, those chunks are its references by construction.

```bash
anchor synth corpus.json -n 30 -o questions.json
anchor run my_rag:pipeline questions.json -v v1
```

In the dashboard: **Evaluate → Build and run a pipeline → Questions →
Generate questions from this corpus**.

## Question types

| Type | References | What it tests |
|---|---|---|
| **Single-hop** | the one chunk it was written from | Basic retrieval and answering |
| **Multi-hop** | two chunks that share a distinctive term | Whether the retriever brings back *both* pieces; recall@k drops to 0.5 if it finds only one |
| **Unanswerable** | none | Whether the pipeline says "I don't know" instead of inventing an answer |

The default mix is 60% single-hop, 20% multi-hop, 20% unanswerable
(`--mix single_hop=6,multi_hop=2,unanswerable=2`).

Multi-hop pairs are chunks linked by a term that appears in only two or
three chunks of the corpus, so the pair is about a common topic rather than
sharing a word like "the".

## Generators

### Templates (offline, deterministic)

Each sentence is scanned for one fact to blank out, in this order:

| Rule | Sentence | Question | Answer |
|---|---|---|---|
| date | …promulgated on 20 September 2015 by… | …promulgated on what date by…? | 20 September 2015 |
| count | Nepal is divided into seven provinces, each… | Nepal is divided into how many provinces? | seven |
| numbered | Article 7 makes Nepali the official language… | Which article makes Nepali the official language…? | Article 7 |
| definition | The National Assembly is a permanent house… | What is the National Assembly? | a permanent house… |

Clauses after the blank (", each with…") are cut, and passives such as
"was promulgated" are skipped by the definition rule. A multi-hop template
question joins two single-hop questions ("…, and which article…?").

Unanswerable template questions ask about the corpus's main subject (the
most frequent proper noun) using a fixed list of topics such as currency,
national anthem or rainfall. A topic is used only if none of its key words
appear anywhere in the corpus, so the corpus can't answer it.

**Limitation:** template questions reuse the source wording, so lexical
retrievers find them easily and almost all are labelled *easy*. Use them
for smoke tests, CI and learning the tool, not as a stand-in for real user
questions.

### Claude

Claude gets the chunk (or the pair, or a sample of the corpus for
unanswerable questions) and writes a question the way a user would, without
copying phrases. Structured outputs guarantee a `{question, answer}` shape.
These questions differ in wording from the source, so they actually test
retrieval. Each candidate is one API call, and attempts are capped at twice
the number requested.

## Quality checks

Every candidate goes through the same deterministic checks, whichever
generator wrote it:

- **De-duplication.** A question whose content words overlap an earlier
  one with Jaccard ≥ 0.8 is dropped.
- **No leaked answers.** A template question that still contains its
  answer is discarded.
- **Difficulty**, from how the built-in retrievers rank the reference
  chunks for that question:
  - *easy:* even plain keyword overlap ranks every reference first
  - *medium:* BM25 finds every reference in its top k (default 3)
  - *hard:* BM25 misses at least one reference in its top k
  - *n/a:* unanswerable
- **Skip easy questions** (`--drop-easy`) removes the easy ones, so the set
  concentrates on questions where retrievers differ.

The same corpus, settings and `--seed` always give the same template
questions.

## Output format

A JSON list that `anchor run`, the dashboard and `parse_dataset` all read:

```json
{
  "query": "Nepal is divided into how many provinces?",
  "reference_chunk_ids": ["prov-007"],
  "reference_answer": "seven",
  "type": "single_hop",
  "difficulty": "easy",
  "generator": "template",
  "rule": "count"
}
```

Only `query` and `reference_chunk_ids` are used for scoring today.
`reference_answer` and `type` are kept for answer-completeness and
abstention scoring, which are next on the roadmap.

## Review before you trust it

Generated questions are a starting point. In the dashboard every question
has a **Use** checkbox and editable text and reference ids: untick weak
ones, fix wording, then add them to the question table or download them.
Spot-check a few reference ids before relying on a set for a release
decision.
