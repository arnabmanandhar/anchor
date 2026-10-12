"""
Faithfulness judge.

The judge may not just rate an answer. It has to break the answer into
atomic claims and, for each claim, give one of three verdicts:

  supported     a specific retrieved chunk states it (the chunk id is cited)
  contradicted  a retrieved chunk states something incompatible with it
  unsupported   no retrieved chunk says it either way

faithfulness_score = supported claims / total claims. The score is derived
from the audit, never a number the model invents, so every score can be
checked by opening the cited chunks.

Backends
  "llm"        any model reachable through anchor.llm (Gemini, Groq,
               OpenRouter, Hugging Face, OpenAI, xAI, Anthropic, Mistral, a
               local server). The reply must match a JSON schema, and a
               citation of a chunk that wasn't retrieved is downgraded.
  "heuristic"  offline lexical check: claims are the answer's sentences, a
               claim is supported when enough of its content words (lightly
               stemmed) appear in one chunk, and a claim whose numbers
               disagree with that chunk's numbers is contradicted.

backend="auto" (the default) uses the LLM when a connection is available,
from the arguments or the environment (see anchor.llm.connection_from_env),
and the heuristic otherwise. If an LLM call fails, the heuristic runs
instead and the verdict says why.

Caching: pass cache= (the Tracer implements it) and identical judge inputs,
for the same model and prompt version, are judged once. Re-running an
evaluation costs nothing for unchanged answers and gives identical scores.

Keys never enter a verdict, a cache key, a trace or a repr.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional, Protocol

from . import llm
from .llm import Connection, LLMError, Secret
from .schema import Chunk, ClaimVerdict, JudgeVerdict

# Bump when the prompt or schema changes, so cached verdicts aren't reused.
PROMPT_VERSION = "judge-v3"

# Heuristic: a sentence counts as supported when at least this share of its
# (stemmed) content words appear in a single retrieved chunk.
HEURISTIC_THRESHOLD = 0.6

VERDICTS = ("supported", "contradicted", "unsupported")

_JUDGE_SYSTEM_PROMPT = """You are a strict faithfulness auditor for a retrieval-augmented QA system.

You get a QUESTION, RETRIEVED CHUNKS (each with an id) and a GENERATED ANSWER.

1. Split the GENERATED ANSWER into short, atomic factual claims. Skip filler
   such as "Here is what I found". If the answer only says it doesn't know,
   return an empty claims list.
2. Give each claim exactly one verdict, judged ONLY against the chunks (no
   outside knowledge, no assumptions):
   - "supported": one chunk states it. Set chunk_id to that chunk's id,
     copied exactly.
   - "contradicted": a chunk states something incompatible (a different
     number, name, date or negation). Set chunk_id to that chunk's id.
   - "unsupported": no chunk says it either way. Set chunk_id to null.
   Use note for a few words of explanation when the verdict is not supported.
3. End with one short sentence in reasoning summarising the audit.
"""

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "chunk_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["claim", "verdict", "chunk_id", "note"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["claims", "reasoning"],
    "additionalProperties": False,
}


class VerdictCache(Protocol):
    def get_verdict(self, key: str) -> Optional[dict]: ...
    def put_verdict(self, key: str, verdict: dict) -> None: ...


class FaithfulnessJudge:
    """
    FaithfulnessJudge()                                   # LLM if a key is in the environment, else heuristic
    FaithfulnessJudge(backend="heuristic")                # always offline
    FaithfulnessJudge(provider="groq", api_key="gsk_…")   # a specific provider (model defaults per provider)
    FaithfulnessJudge(connection=Connection(...))         # full control
    """

    def __init__(
        self,
        connection: Optional[Connection] = None,
        backend: str = "auto",
        cache: Optional[VerdictCache] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if backend not in ("auto", "llm", "heuristic"):
            raise ValueError(f"backend must be 'auto', 'llm' or 'heuristic', got {backend!r}")
        self.backend = backend
        self.cache = cache
        if connection is None and backend != "heuristic":
            if provider or api_key:
                provider = provider or llm.DEFAULT_PROVIDER
                spec = llm.PROVIDERS[provider]
                connection = Connection(provider, model or spec.models[0],
                                        Secret(api_key) if api_key else None, base_url)
            else:
                connection = llm.connection_from_env()
        self.connection = connection

    def __repr__(self) -> str:
        return f"FaithfulnessJudge(backend={self.active_backend!r}, model={self.model!r})"

    @property
    def model(self) -> Optional[str]:
        """provider/model of the LLM backend, or None for the heuristic."""
        return f"{self.connection.provider}/{self.connection.model}" if self.connection else None

    @property
    def active_backend(self) -> str:
        """The backend score() will try first."""
        if self.backend == "heuristic" or self.connection is None:
            return "heuristic"
        if self.connection.key is None and self.connection.provider != "custom":
            return "heuristic"
        return "llm"

    def score(self, query: str, chunks: list[Chunk], answer: str) -> JudgeVerdict:
        if self.active_backend == "llm":
            key = self._cache_key(query, chunks, answer)
            cached = self.cache.get_verdict(key) if self.cache else None
            if cached:
                verdict = JudgeVerdict(**cached)
                verdict.cached, verdict.input_tokens, verdict.output_tokens = True, 0, 0
                return verdict
            try:
                verdict = self._score_llm(query, chunks, answer)
            except Exception as e:  # network/auth/parse failure -> degrade, don't crash a run
                reason = e.kind if isinstance(e, LLMError) else type(e).__name__
                verdict = self._score_heuristic(query, chunks, answer)
                verdict.reasoning = f"LLM judge failed ({reason}); used the heuristic instead."
                verdict.fallback_reason = reason
                return verdict
            if self.cache:
                self.cache.put_verdict(key, verdict.model_dump())
            return verdict
        return self._score_heuristic(query, chunks, answer)

    def estimate_cost(self, verdict: JudgeVerdict) -> float:
        if verdict.judge_backend != "llm" or not self.connection:
            return 0.0
        return self.connection.cost(verdict.input_tokens, verdict.output_tokens)

    def _cache_key(self, query: str, chunks: list[Chunk], answer: str) -> str:
        material = json.dumps([PROMPT_VERSION, self.model, query, [[c.id, c.text] for c in chunks], answer])
        return hashlib.sha256(material.encode()).hexdigest()

    # ---------------------------------------------------------------- LLM

    def _score_llm(self, query: str, chunks: list[Chunk], answer: str) -> JudgeVerdict:
        chunk_block = "\n".join(f"[{c.id}] {c.text}" for c in chunks)
        user_msg = f"QUESTION:\n{query}\n\nRETRIEVED CHUNKS:\n{chunk_block}\n\nGENERATED ANSWER:\n{answer}"
        data, completion = llm.complete_json(self.connection, _JUDGE_SYSTEM_PROMPT, user_msg, _VERDICT_SCHEMA,
                                             max_tokens=8192, name="faithfulness_audit")

        # Enforce the audit contract: a verdict that points at a chunk only
        # stands if that chunk was actually retrieved. A made-up citation is
        # downgraded, so the score can't be inflated by invented ids.
        retrieved_ids = {c.id for c in chunks}
        claims: list[ClaimVerdict] = []
        for c in data.get("claims", []):
            verdict, chunk_id, note = c["verdict"], c.get("chunk_id"), c.get("note")
            if verdict in ("supported", "contradicted") and chunk_id not in retrieved_ids:
                note = f"judge cited chunk {chunk_id!r}, which was not retrieved"
                verdict, chunk_id = "unsupported", None
            claims.append(ClaimVerdict(claim=c["claim"], supported=verdict == "supported", verdict=verdict,
                                       chunk_id=chunk_id if verdict != "unsupported" else None, note=note))

        return JudgeVerdict(
            faithfulness_score=_score(claims, answer),
            claims=claims,
            reasoning=data.get("reasoning"),
            judge_backend="llm",
            judge_model=self.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )

    # ----------------------------------------------------------- heuristic

    def _score_heuristic(self, query: str, chunks: list[Chunk], answer: str) -> JudgeVerdict:
        """Offline fallback. Crude on purpose: it keeps the pipeline running
        with no key and no network, it does not replace the LLM judge."""
        claims: list[ClaimVerdict] = []
        for sentence in split_sentences(answer):
            best_chunk, best_overlap = None, 0.0
            for chunk in chunks:
                overlap = claim_overlap(sentence, chunk.text)
                if overlap > best_overlap:
                    best_overlap, best_chunk = overlap, chunk

            claim_nums = numbers(sentence)
            chunk_nums = numbers(best_chunk.text) if best_chunk else set()
            if best_chunk and claim_nums and best_overlap >= HEURISTIC_THRESHOLD / 2 and chunk_nums \
                    and not claim_nums & chunk_nums:
                claims.append(ClaimVerdict(
                    claim=sentence, supported=False, verdict="contradicted", chunk_id=best_chunk.id,
                    note=f"numbers differ: claim says {', '.join(sorted(claim_nums))}, "
                         f"chunk says {', '.join(sorted(chunk_nums))}"))
            elif best_overlap >= HEURISTIC_THRESHOLD and not (claim_nums - chunk_nums):
                claims.append(ClaimVerdict(claim=sentence, supported=True, verdict="supported", chunk_id=best_chunk.id))
            else:
                why = (f"number {', '.join(sorted(claim_nums - chunk_nums))} not in the best chunk"
                       if best_overlap >= HEURISTIC_THRESHOLD else
                       f"best word overlap {best_overlap:.2f} < {HEURISTIC_THRESHOLD}")
                claims.append(ClaimVerdict(claim=sentence, supported=False, verdict="unsupported", note=why))

        return JudgeVerdict(
            faithfulness_score=_score(claims, answer),
            claims=claims,
            reasoning="Heuristic word-overlap scoring (no model used).",
            judge_backend="heuristic",
        )


def _score(claims: list[ClaimVerdict], answer: str) -> float:
    # An answer with no factual claims (an empty answer or a plain "I don't
    # know") asserts nothing unsupported, so it is fully faithful only if it
    # is an abstention; an empty answer scores 0.
    if not claims:
        return 1.0 if is_abstention(answer) else 0.0
    return sum(c.supported for c in claims) / len(claims)


_ABSTAIN = re.compile(
    r"\b(i (do not|don't|cannot|can't) (know|find|answer|tell)|not (mentioned|stated|specified|covered|available|provided)"
    r"|no (information|mention|answer)|(cannot|can't|unable to) (be )?(determined|answer|find)|does not (say|mention|specify)"
    r"|insufficient (information|context))\b", re.IGNORECASE)


def is_abstention(answer: str) -> bool:
    """True when the answer declines to answer ("I don't know", "not mentioned in the context")."""
    return bool(answer.strip()) and bool(_ABSTAIN.search(answer))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def stem(word: str) -> str:
    """Light suffix stripping so 'provinces' matches 'province'. Deliberately
    conservative: it only merges plural and simple verb forms."""
    for suffix, repl, min_len in (("ies", "y", 5), ("sses", "ss", 6), ("es", "e", 5), ("s", "", 4),
                                  ("ing", "", 6), ("ed", "", 5)):
        if word.endswith(suffix) and len(word) >= min_len and not word.endswith("ss"):
            return word[: -len(suffix)] + repl
    return word


def tokenize(text: str) -> set:
    return {stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                 "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
                 "hundred": "100", "thousand": "1000"}


def numbers(text: str) -> set:
    """Numbers mentioned in text, normalised ("seven" and "7" are the same)."""
    out = {n.replace(",", "") for n in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)}
    out |= {_NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in _NUMBER_WORDS}
    return out


def claim_overlap(claim: str, chunk_text: str) -> float:
    """|claim ∩ chunk| / |claim| over stemmed content words: the share of the
    claim's vocabulary that the chunk covers. Asymmetric on purpose, so a
    long chunk isn't penalised for containing more than the claim needs."""
    claim_tokens, chunk_tokens = tokenize(claim), tokenize(chunk_text)
    if not claim_tokens or not chunk_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "in",
    "on", "for", "with", "it", "that", "this", "as", "by", "be", "at", "or",
    "from", "has", "have", "had", "its", "which", "these", "those", "can",
}
