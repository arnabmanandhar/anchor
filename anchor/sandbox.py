"""
A small, dependency-free RAG sandbox so people can try Anchor from the
dashboard without writing any code: load a corpus, pick a retriever, pick a
generator, and Anchor evaluates the resulting pipeline like any other.

This is sample input for the evaluator, not part of the evaluator. Anchor
itself never retrieves anything; it scores rankings and answers that a
pipeline produced. Everything here talks to the same (chunks, answer)
contract a real pipeline would.

Retrievers (all pure Python, deterministic, ties broken by corpus order):
  - "bm25"     Okapi BM25 (k1=1.5, b=0.75), the classic lexical baseline
  - "tfidf"    TF-IDF vectors + cosine similarity
  - "keyword"  count of distinct query words present in the chunk (no IDF),
               deliberately weak so comparisons have something to show
  - "hybrid"   reciprocal-rank fusion of bm25 and tfidf

Generators:
  - "extractive"  picks the chunk sentences that best overlap the query;
                  offline, free, faithful by construction (useful baseline)
  - "llm"         asks Claude to answer from the chunks (needs an API key)
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .judge import DEFAULT_MODEL, split_sentences
from .schema import Chunk

RETRIEVERS = ("bm25", "tfidf", "keyword", "hybrid")
GENERATORS = ("extractive", "llm")

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "in", "on", "for", "with",
    "it", "that", "this", "as", "by", "be", "at", "or", "from", "has", "have", "had", "its",
    "which", "what", "who", "how", "does", "do", "did", "many", "much",
}


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP] if drop_stopwords else tokens


# ------------------------------------------------------------------ corpus


def parse_corpus(data: bytes | str, filename: str = "corpus.txt") -> List[Chunk]:
    """Accepts CSV (id,text columns), JSON (list of {id,text} or strings),
    JSONL (one {id,text} per line), or plain text (chunks separated by blank
    lines; ids become c1, c2, ...)."""
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    name = filename.lower()

    if name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows or "text" not in rows[0]:
            raise ValueError("CSV corpus needs a 'text' column (and optionally 'id').")
        items = [{"id": r.get("id") or f"c{i}", "text": r["text"]} for i, r in enumerate(rows, start=1)]
    elif name.endswith(".jsonl"):
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif name.endswith(".json"):
        items = json.loads(text)
        if isinstance(items, dict):
            items = items.get("chunks") or items.get("corpus") or []
    else:
        items = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            item = {"id": f"c{i}", "text": item}
        if not str(item.get("text", "")).strip():
            continue
        chunks.append(Chunk(id=str(item.get("id") or f"c{i}"), text=str(item["text"]).strip()))
    ids = [c.id for c in chunks]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate chunk ids in corpus: {', '.join(dupes[:5])}")
    if not chunks:
        raise ValueError("No chunks found in the corpus.")
    return chunks


# -------------------------------------------------------------- retrievers


class _Index:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.docs = [tokenize(c.text) for c in self.chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        df = Counter(t for d in self.docs for t in set(d))
        self.df = df
        # Smoothed IDF (as in scikit-learn) for TF-IDF; BM25 uses its own below.
        self.idf = {t: math.log((1 + self.n) / (1 + f)) + 1 for t, f in df.items()}

    def bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
        q = tokenize(query)
        scores = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                f = tf[t]
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / (self.avgdl or 1)))
            scores.append(s)
        return scores

    def tfidf(self, query: str) -> List[float]:
        qtf = Counter(tokenize(query))
        qvec = {t: f * self.idf.get(t, 0.0) for t, f in qtf.items()}
        qnorm = math.sqrt(sum(v * v for v in qvec.values()))
        scores = []
        for tf in self.tf:
            dvec = {t: f * self.idf[t] for t, f in tf.items()}
            dnorm = math.sqrt(sum(v * v for v in dvec.values()))
            dot = sum(qvec[t] * dvec.get(t, 0.0) for t in qvec)
            scores.append(dot / (qnorm * dnorm) if qnorm and dnorm else 0.0)
        return scores

    def keyword(self, query: str) -> List[float]:
        q = set(tokenize(query, drop_stopwords=False))
        return [float(len(q & set(tokenize(c.text, drop_stopwords=False)))) for c in self.chunks]


def _rank(scores: List[float]) -> List[int]:
    # Stable sort: equal scores keep corpus order, so results are reproducible.
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def make_retriever(chunks: Sequence[Chunk], method: str = "bm25", top_k: int = 3) -> Callable[[str], List[Chunk]]:
    if method not in RETRIEVERS:
        raise ValueError(f"unknown retriever {method!r}; choose from {RETRIEVERS}")
    index = _Index(chunks)

    def retrieve(query: str) -> List[Chunk]:
        if method == "hybrid":
            # Reciprocal-rank fusion: score = sum over rankers of 1 / (60 + rank).
            fused = [0.0] * index.n
            for ranker in (index.bm25, index.tfidf):
                for rank, i in enumerate(_rank(ranker(query)), start=1):
                    fused[i] += 1.0 / (60 + rank)
            scores = fused
        else:
            scores = getattr(index, method)(query)
        order = _rank(scores)[:top_k]
        return [Chunk(id=index.chunks[i].id, text=index.chunks[i].text, score=round(scores[i], 4)) for i in order]

    return retrieve


# -------------------------------------------------------------- generators


def extractive_answer(query: str, chunks: Sequence[Chunk], max_sentences: int = 2) -> str:
    """Pick the sentences from the retrieved chunks with the most query-word
    overlap, in rank order. It never invents text, so with the heuristic
    judge it scores as fully faithful; use it as a grounded baseline."""
    q = set(tokenize(query))
    scored = []
    for rank, chunk in enumerate(chunks):
        for pos, sentence in enumerate(split_sentences(chunk.text)):
            overlap = len(q & set(tokenize(sentence)))
            scored.append((-overlap, rank, pos, sentence))
    if not scored:
        return "I don't know."
    picked = sorted(scored)[:max_sentences]
    picked = [s for s in picked if s[0] < 0] or picked[:1]
    return " ".join(s[3] for s in sorted(picked, key=lambda s: (s[1], s[2])))


_GEN_SYSTEM = (
    "Answer the question using only the provided context chunks. "
    "Be concise (one to three sentences). If the context does not contain the answer, say you don't know."
)


def llm_answer(query: str, chunks: Sequence[Chunk], api_key: str, model: Optional[str] = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    context = "\n".join(f"[{c.id}] {c.text}" for c in chunks)
    resp = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=16000,
        system=_GEN_SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("generation request was refused")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def make_pipeline(
    chunks: Sequence[Chunk],
    retriever: str = "bm25",
    top_k: int = 3,
    generator: str = "extractive",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Callable[[str], Tuple[List[Chunk], str]]:
    """Build a pipeline function in the shape evaluate_pipeline expects."""
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}; choose from {GENERATORS}")
    if generator == "llm" and not api_key:
        raise ValueError("the llm generator needs an API key")
    retrieve = make_retriever(chunks, retriever, top_k)

    def pipeline(query: str):
        retrieved = retrieve(query)
        if generator == "llm":
            return retrieved, llm_answer(query, retrieved, api_key, model)
        return retrieved, extractive_answer(query, retrieved)

    return pipeline


def parse_reference_ids(value) -> Optional[List[str]]:
    """'c1, c3' / ['c1','c3'] / None -> ['c1','c3'] or None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        ids = [str(v).strip() for v in value]
    else:
        ids = [p.strip() for p in re.split(r"[,;\s]+", str(value))]
    ids = [i for i in ids if i]
    return ids or None


def parse_dataset(data: bytes | str, filename: str = "dataset.json") -> List[Dict]:
    """Questions file: JSON list / JSONL of {query, reference_chunk_ids?}, or
    CSV with 'query' and optional 'reference_chunk_ids' (comma separated)."""
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    name = filename.lower()
    if name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
    elif name.endswith(".jsonl"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    out = []
    for r in rows:
        query = str(r.get("query", "")).strip()
        if query:
            out.append({"query": query, "reference_chunk_ids": parse_reference_ids(r.get("reference_chunk_ids"))})
    if not out:
        raise ValueError("No questions found (each row needs a 'query').")
    return out
