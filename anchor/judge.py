"""
Faithfulness judge.

Philosophy: the judge is not allowed to just vibe-score the answer. It has
to break the answer into atomic claims and, for each claim, either point at
the exact retrieved chunk that supports it, or mark it unsupported. The
faithfulness_score is a *derived* number (supported claims / total claims),
never a number the model invents directly — that's what keeps this
deterministic-ish and auditable instead of "trust the LLM's vibes".

Two backends:
  - "llm":       calls Anthropic's Messages API with a strict JSON schema
                 (structured outputs), so the audit always parses.
  - "heuristic": zero-dependency lexical-overlap fallback so the whole
                 project runs end-to-end with no API key, no network call,
                 and no cost. Useful for CI and for the demo.

Backend selection (backend="auto", the default): use the LLM if an API key
was passed in or ANTHROPIC_API_KEY is set, otherwise the heuristic. The
result always records which backend actually produced it.

Security: the API key is only ever held on the judge instance and handed to
the Anthropic client. It is never written to a verdict, a trace, or a log,
and LLM failures are reported by exception *type* only, so a provider error
message can never echo credentials into the trace database or the UI.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from .schema import Chunk, ClaimVerdict, JudgeVerdict

DEFAULT_MODEL = "claude-sonnet-5-5"

# Rough list prices (USD per million tokens) used only for the cost estimate
# shown in summaries. Override via FaithfulnessJudge(price_per_mtok=...).
DEFAULT_PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}

# Heuristic: a sentence counts as supported when at least this fraction of
# its content words appear in a single retrieved chunk.
HEURISTIC_THRESHOLD = 0.5

_JUDGE_SYSTEM_PROMPT = """You are a strict faithfulness auditor for a RAG system.

You will be given a QUESTION, a set of RETRIEVED CHUNKS (each with an id),
and a GENERATED ANSWER.

Break the GENERATED ANSWER into a short list of atomic factual claims.
For each claim, decide whether it is directly supported by the text of one
specific retrieved chunk. A claim is "supported" only if that chunk actually
contains the information — do not use outside knowledge, do not assume.

For a supported claim, set chunk_id to the id of the supporting chunk,
exactly as written in the RETRIEVED CHUNKS. For an unsupported claim, set
chunk_id to null and say briefly in note why it is unsupported.
Finish with one short sentence in reasoning summarizing the audit.
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
                    "supported": {"type": "boolean"},
                    "chunk_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["claim", "supported", "chunk_id", "note"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["claims", "reasoning"],
    "additionalProperties": False,
}


class FaithfulnessJudge:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        backend: str = "auto",
        price_per_mtok: Optional[dict] = None,
    ):
        if backend not in ("auto", "llm", "heuristic"):
            raise ValueError(f"backend must be 'auto', 'llm' or 'heuristic', got {backend!r}")
        self.model = model or os.environ.get("ANCHOR_JUDGE_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or None
        self.backend = backend
        self.price_per_mtok = price_per_mtok or DEFAULT_PRICE_PER_MTOK

    def __repr__(self) -> str:  # never show the key
        return f"FaithfulnessJudge(model={self.model!r}, backend={self.active_backend!r})"

    @property
    def active_backend(self) -> str:
        """The backend score() will try first."""
        if self.backend == "heuristic" or not self._api_key:
            return "heuristic"
        return "llm"

    def score(self, query: str, chunks: List[Chunk], answer: str) -> JudgeVerdict:
        if self.active_backend == "llm":
            try:
                return self._score_llm(query, chunks, answer)
            except Exception as e:  # network/auth/parse failure -> degrade, don't crash a run
                verdict = self._score_heuristic(query, chunks, answer)
                verdict.reasoning = f"LLM judge failed ({type(e).__name__}); used heuristic fallback."
                verdict.fallback_reason = type(e).__name__
                return verdict
        return self._score_heuristic(query, chunks, answer)

    def estimate_cost(self, verdict: JudgeVerdict) -> float:
        if verdict.judge_backend != "llm":
            return 0.0
        return (
            verdict.input_tokens * self.price_per_mtok["input"]
            + verdict.output_tokens * self.price_per_mtok["output"]
        ) / 1_000_000

    # ---------------------------------------------------------------- LLM

    def _score_llm(self, query: str, chunks: List[Chunk], answer: str) -> JudgeVerdict:
        import anthropic  # local import so the package doesn't hard-require it

        client = anthropic.Anthropic(api_key=self._api_key)
        chunk_block = "\n".join(f"[{c.id}] {c.text}" for c in chunks)
        user_msg = f"QUESTION:\n{query}\n\nRETRIEVED CHUNKS:\n{chunk_block}\n\nGENERATED ANSWER:\n{answer}"

        resp = client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=_JUDGE_SYSTEM_PROMPT,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
            messages=[{"role": "user", "content": user_msg}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("judge request was refused")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("judge output was truncated")

        raw = next(block.text for block in resp.content if block.type == "text")
        data = json.loads(raw)

        # Enforce the audit contract: a claim only counts as supported if it
        # cites a chunk that was actually retrieved. A hallucinated citation
        # is downgraded, so the score can't be inflated by made-up ids.
        retrieved_ids = {c.id for c in chunks}
        claims: List[ClaimVerdict] = []
        for c in data.get("claims", []):
            claim = ClaimVerdict(**c)
            if claim.supported and claim.chunk_id not in retrieved_ids:
                claim = ClaimVerdict(
                    claim=claim.claim,
                    supported=False,
                    chunk_id=None,
                    note=f"judge cited chunk {claim.chunk_id!r}, which was not retrieved",
                )
            claims.append(claim)

        supported = sum(1 for c in claims if c.supported)
        score = supported / len(claims) if claims else 0.0

        return JudgeVerdict(
            faithfulness_score=score,
            claims=claims,
            reasoning=data.get("reasoning"),
            judge_backend="llm",
            judge_model=self.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    # ----------------------------------------------------------- heuristic

    def _score_heuristic(self, query: str, chunks: List[Chunk], answer: str) -> JudgeVerdict:
        """Zero-dependency fallback: split the answer into sentences, and
        mark a sentence 'supported' if enough of its content words appear in
        some chunk. Crude on purpose — it exists so the pipeline always
        runs, not to replace the LLM judge."""

        claims: List[ClaimVerdict] = []
        for sentence in split_sentences(answer):
            best_chunk_id, best_overlap = None, 0.0
            for chunk in chunks:
                overlap = claim_overlap(sentence, chunk.text)
                if overlap > best_overlap:
                    best_overlap, best_chunk_id = overlap, chunk.id

            supported = best_overlap >= HEURISTIC_THRESHOLD
            claims.append(
                ClaimVerdict(
                    claim=sentence,
                    supported=supported,
                    chunk_id=best_chunk_id if supported else None,
                    note=None if supported else f"best lexical overlap {best_overlap:.2f} < {HEURISTIC_THRESHOLD}",
                )
            )

        score = sum(1 for c in claims if c.supported) / len(claims) if claims else 0.0
        return JudgeVerdict(
            faithfulness_score=score,
            claims=claims,
            reasoning="Heuristic lexical-overlap scoring (no API key available).",
            judge_backend="heuristic",
        )


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS


def claim_overlap(claim: str, chunk_text: str) -> float:
    """|claim ∩ chunk| / |claim| over content words: the share of the claim's
    vocabulary that the chunk covers. Asymmetric on purpose — a long chunk
    shouldn't be penalised for containing more than the claim needs."""
    claim_tokens, chunk_tokens = tokenize(claim), tokenize(chunk_text)
    if not claim_tokens or not chunk_tokens:
        return 0.0
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "in",
    "on", "for", "with", "it", "that", "this", "as", "by", "be", "at", "or",
    "from", "has", "have", "had", "its", "which", "these", "those", "can",
}
