"""
Synthetic test sets: generate evaluation questions from a corpus, with the
ground-truth chunk ids known by construction.

The hard part of evaluating retrieval is labelling which chunks answer each
question. If the question is *written from* a chunk, that label is free.

Question types
  single_hop    answered by one chunk
  multi_hop     needs two chunks that share a topic (both are references)
  unanswerable  on the corpus's topic, but nothing in the corpus answers it;
                no reference ids, and the right answer is to abstain

Generators
  "template"  offline and deterministic. Turns a sentence into a question by
              blanking out one fact (a count, a date, a numbered article, the
              predicate of "X is Y"). Questions stay close to the source
              wording, so they are lexically easy: good for smoke tests and
              CI, not a substitute for real user questions.
  "llm"       a language model (any provider anchor.llm supports) writes
              natural questions from the chunk(s), so wording differs from
              the source and retrieval is genuinely tested.

Every question then goes through the same deterministic checks:
  - de-duplication (token Jaccard >= 0.8 with an earlier question)
  - for template questions, the answer must not appear in the question
  - a difficulty label from how the built-in retrievers rank the
    reference chunks (easy: even keyword overlap ranks them first;
    hard: BM25 misses one in its top k)
  - optional drop_easy, which removes questions the weakest retriever
    already gets perfectly
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Sequence
from typing import Optional

from . import llm
from .judge import split_sentences
from .sandbox import _Index, _rank, tokenize
from .schema import Chunk

QUESTION_TYPES = ("single_hop", "multi_hop", "unanswerable")
DEFAULT_MIX = {"single_hop": 0.6, "multi_hop": 0.2, "unanswerable": 0.2}
GENERATORS = ("template", "llm")
DUPLICATE_JACCARD = 0.8

_NUMBER_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                 "thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|hundred")
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_NUMBERED = "Article|Section|Chapter|Clause|Rule|Part|Schedule|Step|Figure|Table|Appendix"

# Questions about the corpus's main subject that a corpus should not answer
# unless it contains one of the key words; checked against the vocabulary.
_UNANSWERABLE_TEMPLATES: list[tuple[str, tuple[str, ...]]] = [
    ("Who composed the national anthem of {e}?", ("anthem", "composed")),
    ("What is the official currency of {e}?", ("currency",)),
    ("How many airports does {e} have?", ("airport", "airports")),
    ("What was the population of {e} in 1950?", ("population",)),
    ("Which sport is the most popular in {e}?", ("sport", "sports")),
    ("What is the tallest building in {e}?", ("building", "tallest")),
    ("What is the minimum wage in {e}?", ("wage", "wages")),
    ("When did {e} first join the United Nations?", ("united", "nations")),
    ("What is the main export of {e}?", ("export", "exports")),
    ("Who designed the flag of {e}?", ("flag", "designed")),
    ("What is the speed limit on highways in {e}?", ("speed", "highway", "highways")),
    ("How many universities are there in {e}?", ("university", "universities")),
    ("What is the average annual rainfall in {e}?", ("rainfall", "rain")),
    ("Who was the first astronaut from {e}?", ("astronaut",)),
]


# ------------------------------------------------------------- templates


def _finish(question: str, gap_end: int) -> str:
    """Cut trailing clauses after the blank (', each with…'), tidy the end."""
    tail = question[gap_end:]
    cut = re.search(r"[,;:]| which | who | with ", tail)
    if cut:
        question = question[: gap_end + cut.start()]
    return question.strip().rstrip(".") + "?"


def cloze_questions(sentence: str) -> list[dict[str, str]]:
    """Template questions from one sentence, best first. Each is
    {"question", "answer", "rule"}; empty when no rule applies."""
    s = sentence.strip().rstrip(".")
    out = []

    m = re.search(rf"\b\d{{1,2}} (?:{_MONTHS}) \d{{4}}\b", s)
    if m:
        q = s[: m.start()] + "what date" + s[m.end():]
        out.append({"question": _finish(q, m.start() + 9), "answer": m.group(0), "rule": "date"})
    else:
        m = re.search(r"\b(?:in|since|from|until) ((?:1[5-9]|20)\d{2})\b", s)
        if m:
            q = s[: m.start(1)] + "what year" + s[m.end(1):]
            out.append({"question": _finish(q, m.start(1) + 9), "answer": m.group(1), "rule": "year"})

    numbered = set(_NUMBERED.split("|"))
    m = next((m for m in re.finditer(rf"\b(\d[\d,]*|{_NUMBER_WORDS}) ([a-z]+s)\b", s, flags=re.IGNORECASE)
              if s[: m.start()].split()[-1:] not in ([w] for w in numbered)), None)
    if m:
        gap = f"how many {m.group(2)}"
        q = s[: m.start()] + gap + s[m.end():]
        out.append({"question": _finish(q, m.start() + len(gap)), "answer": m.group(1), "rule": "count"})

    m = re.match(rf"({_NUMBERED}) (\d+[A-Za-z]?)\b", s)
    if m and len(s) > m.end() + 10:
        q = f"Which {m.group(1).lower()}" + s[m.end():]
        out.append({"question": q.rstrip(".") + "?", "answer": m.group(0), "rule": "numbered"})

    m = re.match(r"(.{3,80}?) (is|are|was|were) (.+)$", s)
    # Skip passives ("was promulgated on…"): the blank would be the whole event.
    if (m and len(m.group(1).split()) <= 8 and not re.search(r"\d", m.group(1))
            and not m.group(3).split()[0].endswith("ed")):
        subject = m.group(1)
        subject = subject[0].lower() + subject[1:] if subject.split()[0] in ("The", "A", "An") else subject
        out.append({"question": f"What {m.group(2)} {subject}?", "answer": m.group(3), "rule": "definition"})

    return [o for o in out if o["answer"].lower() not in o["question"].lower()]


def _main_entity(chunks: Sequence[Chunk]) -> str:
    """Most frequent capitalised word that isn't generic or a sentence opener."""
    generic = set(_NUMBERED.split("|")) | {"The", "A", "An", "It", "This", "These", "There", "In", "On", "For",
                                          "State", "Each", "Every", "All", "Some", "When", "What", "How"}
    counts: Counter = Counter()
    for c in chunks:
        for sent in split_sentences(c.text):
            for w in re.findall(r"\b[A-Z][a-z]{2,}\b", sent):
                if w not in generic:
                    counts[w] += 1
    return counts.most_common(1)[0][0] if counts else "this subject"


def _shared_term_pairs(chunks: Sequence[Chunk], index: _Index, rng: random.Random) -> list[tuple[Chunk, Chunk, str]]:
    """Pairs of chunks linked by a distinctive term (in 2 or 3 chunks only)."""
    by_term: dict[str, list[int]] = {}
    for i, doc in enumerate(index.docs):
        for t in set(doc):
            if len(t) > 3 and not t.isdigit() and 2 <= index.df[t] <= 3:
                by_term.setdefault(t, []).append(i)
    pairs, seen = [], set()
    for term in sorted(by_term):
        ids = by_term[term]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                key = (ids[a], ids[b])
                if key not in seen:
                    seen.add(key)
                    pairs.append((chunks[ids[a]], chunks[ids[b]], term))
    rng.shuffle(pairs)
    return pairs


def _template_single(chunk: Chunk) -> Optional[dict]:
    for sent in split_sentences(chunk.text):
        qs = cloze_questions(sent)
        if qs:
            return {"query": qs[0]["question"], "reference_answer": qs[0]["answer"], "rule": qs[0]["rule"]}
    return None


def _template_multi(a: Chunk, b: Chunk) -> Optional[dict]:
    qa, qb = _template_single(a), _template_single(b)
    if not qa or not qb:
        return None
    second = qb["query"]
    if second.split()[0] in ("What", "Which", "How", "When", "Who"):
        second = second[0].lower() + second[1:]
    return {"query": f"{qa['query'][:-1]}, and {second}",
            "reference_answer": f"{qa['reference_answer']}; {qb['reference_answer']}", "rule": "compound"}


def _template_unanswerable(chunks: Sequence[Chunk], vocab: set, rng: random.Random) -> list[dict]:
    entity = _main_entity(chunks)
    usable = [t for t, keys in _UNANSWERABLE_TEMPLATES if not any(k in vocab for k in keys)]
    rng.shuffle(usable)
    return [{"query": t.format(e=entity), "reference_answer": None, "rule": "out-of-corpus"} for t in usable]


# ------------------------------------------------------------------- LLM

_LLM_SYSTEM = (
    "You write evaluation questions for a retrieval-augmented QA system. Write the question the way a real "
    "user would ask it: natural wording, no copying of long phrases from the text, no mention of 'the text', "
    "'the passage' or chunk ids. Return JSON only."
)
_LLM_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"],
    "additionalProperties": False,
}
_LLM_TASKS = {
    "single_hop": "Write one question that this chunk fully answers, and its short answer.\n\nCHUNK:\n{context}",
    "multi_hop": ("Write one question that can only be answered by combining facts from BOTH chunks (neither "
                  "alone is enough), and its short answer.\n\nCHUNKS:\n{context}"),
    "unanswerable": ("These chunks are a sample from a document collection. Write one question a user of this "
                     "collection might plausibly ask, on the same topic, that the collection does NOT answer. "
                     "Set the answer to \"not in the corpus\".\n\nCHUNKS:\n{context}"),
}


def _llm_question(kind: str, context_chunks: Sequence[Chunk], connection: llm.Connection) -> Optional[dict]:
    context = "\n".join(f"[{c.id}] {c.text}" for c in context_chunks)
    try:
        data, _ = llm.complete_json(connection, _LLM_SYSTEM, _LLM_TASKS[kind].format(context=context), _LLM_SCHEMA,
                                    max_tokens=2048, name="question")
    except llm.LLMError as e:
        if e.kind in ("auth", "not_found", "blocked"):
            raise  # a broken key or model won't fix itself on the next candidate
        return None
    question = data["question"].strip()
    if not question:
        return None
    answer = None if kind == "unanswerable" else data["answer"].strip()
    return {"query": question, "reference_answer": answer, "rule": "llm"}


# ---------------------------------------------------------------- checks


def _jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x | y else 1.0


def difficulty(query: str, reference_ids: Sequence[str], index: _Index, k: int = 3) -> str:
    """easy: keyword overlap alone ranks every reference in the top len(refs);
    hard: BM25 misses a reference in its top k; otherwise medium."""
    if not reference_ids:
        return "n/a"
    ids = [c.id for c in index.chunks]
    refs = set(reference_ids)
    kw_top = {ids[i] for i in _rank(index.keyword(query))[: len(refs)]}
    if refs <= kw_top:
        return "easy"
    bm_top = {ids[i] for i in _rank(index.bm25(query))[:k]}
    return "medium" if refs <= bm_top else "hard"


def _quotas(n: int, mix: dict[str, float]) -> dict[str, int]:
    total = sum(mix.values()) or 1.0
    raw = {t: n * w / total for t, w in mix.items() if w > 0}
    quotas = {t: int(v) for t, v in raw.items()}
    for t in sorted(raw, key=lambda t: raw[t] - quotas[t], reverse=True)[: n - sum(quotas.values())]:
        quotas[t] += 1
    return quotas


# ------------------------------------------------------------------- API


def generate_questions(
    chunks: Sequence[Chunk],
    n: int = 20,
    mix: Optional[dict[str, float]] = None,
    generator: str = "template",
    connection: Optional[llm.Connection] = None,
    seed: int = 0,
    drop_easy: bool = False,
    k: int = 3,
) -> list[dict]:
    """Generate up to n questions from chunks.

    mix maps question types to weights (default 60% single_hop, 20% multi_hop,
    20% unanswerable). Each returned dict has query, reference_chunk_ids
    (empty for unanswerable), reference_answer, type, difficulty, generator
    and rule (which template, or "llm"). Fewer than n come back when the
    corpus can't supply enough distinct questions of a type.
    """
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}; choose from {GENERATORS}")
    if generator == "llm" and connection is None:
        raise ValueError("the llm generator needs a model connection")
    mix = mix if mix is not None else DEFAULT_MIX
    unknown = set(mix) - set(QUESTION_TYPES)
    if unknown:
        raise ValueError(f"unknown question types {sorted(unknown)}; choose from {QUESTION_TYPES}")
    chunks = list(chunks)
    if not chunks:
        return []
    rng = random.Random(seed)
    index = _Index(chunks)
    vocab = set(index.df)
    quotas = _quotas(n, mix)

    def candidates(kind: str):
        if kind == "single_hop":
            order = chunks[:]
            rng.shuffle(order)
            for c in order:
                q = _llm_question(kind, [c], connection) if generator == "llm" else _template_single(c)
                if q:
                    yield q, [c.id]
        elif kind == "multi_hop":
            for a, b, _term in _shared_term_pairs(chunks, index, rng):
                q = _llm_question(kind, [a, b], connection) if generator == "llm" else _template_multi(a, b)
                if q:
                    yield q, [a.id, b.id]
        else:
            if generator == "llm":
                for _ in range(quotas.get(kind, 0) * 2):
                    sample = rng.sample(chunks, min(len(chunks), 12))
                    q = _llm_question(kind, sample, connection)
                    if q:
                        yield q, []
            else:
                for q in _template_unanswerable(chunks, vocab, rng):
                    yield q, []

    out: list[dict] = []
    for kind in QUESTION_TYPES:
        want = quotas.get(kind, 0)
        got = 0
        # An LLM call is made per candidate, so cap attempts at twice the quota.
        attempts = 0
        for q, refs in candidates(kind) if want else ():
            attempts += 1
            if got >= want or (generator == "llm" and attempts > 2 * want):
                break
            if any(_jaccard(q["query"], o["query"]) >= DUPLICATE_JACCARD for o in out):
                continue
            level = difficulty(q["query"], refs, index, k=k)
            if drop_easy and level == "easy":
                continue
            out.append({
                "query": q["query"],
                "reference_chunk_ids": refs,
                "reference_answer": q["reference_answer"],
                "type": kind,
                "difficulty": level,
                "generator": generator,
                "rule": q["rule"],
            })
            got += 1
    return out
