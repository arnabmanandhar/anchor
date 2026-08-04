"""
Faithfulness judge.

Philosophy: the judge is not allowed to just vibe-score the answer. It has
to break the answer into atomic claims and, for each claim, either point at
the exact retrieved chunk that supports it, or mark it unsupported. The
faithfulness_score is a *derived* number (supported claims / total claims),
never a number the model invents directly — that's what keeps this
deterministic-ish and auditable instead of "trust the LLM's vibes".

Two backends:
  - "llm":       calls Anthropic's Messages API with a strict JSON-only
                 system prompt (temperature=0).
  - "heuristic": zero-dependency lexical-overlap fallback so the whole
                 project runs end-to-end with no API key, no network call,
                 and no cost. Useful for CI and for the demo.

Set ANTHROPIC_API_KEY in the environment to use the LLM backend; otherwise
Anchor automatically falls back to the heuristic and says so in the result.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from .schema import Chunk, ClaimVerdict, JudgeVerdict

_JUDGE_SYSTEM_PROMPT = """You are a strict faithfulness auditor for a RAG system.

You will be given a QUESTION, a set of RETRIEVED CHUNKS (each with an id),
and a GENERATED ANSWER.

Break the GENERATED ANSWER into a short list of atomic factual claims.
For each claim, decide whether it is directly supported by the text of one
specific retrieved chunk. A claim is "supported" only if that chunk actually
contains the information — do not use outside knowledge, do not assume.

Respond with ONLY valid JSON, no markdown fences, no preamble, in this exact
shape:

{
  "claims": [
    {"claim": "...", "supported": true, "chunk_id": "c1", "note": null},
    {"claim": "...", "supported": false, "chunk_id": null, "note": "not in any chunk"}
  ],
  "reasoning": "one short sentence summarizing the audit"
}
"""


class FaithfulnessJudge:
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def score(self, query: str, chunks: List[Chunk], answer: str) -> JudgeVerdict:
        if self.api_key:
            try:
                return self._score_llm(query, chunks, answer)
            except Exception as e:  # network/parse failure -> degrade, don't crash a run
                verdict = self._score_heuristic(query, chunks, answer)
                verdict.reasoning = f"LLM judge failed ({e}); used heuristic fallback."
                return verdict
        return self._score_heuristic(query, chunks, answer)

    # ---------------------------------------------------------------- LLM

    def _score_llm(self, query: str, chunks: List[Chunk], answer: str) -> JudgeVerdict:
        import anthropic  # local import so the package doesn't hard-require it

        client = anthropic.Anthropic(api_key=self.api_key)
        chunk_block = "\n".join(f"[{c.id}] {c.text}" for c in chunks)
        user_msg = f"QUESTION:\n{query}\n\nRETRIEVED CHUNKS:\n{chunk_block}\n\nGENERATED ANSWER:\n{answer}"

        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
        raw = raw.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        claims = [ClaimVerdict(**c) for c in data.get("claims", [])]
        supported = sum(1 for c in claims if c.supported)
        score = supported / len(claims) if claims else 0.0

        return JudgeVerdict(
            faithfulness_score=score,
            claims=claims,
            reasoning=data.get("reasoning"),
            judge_backend="llm",
        )

    # ----------------------------------------------------------- heuristic

    def _score_heuristic(self, query: str, chunks: List[Chunk], answer: str) -> JudgeVerdict:
        """Zero-dependency fallback: split the answer into sentences, and
        mark a sentence 'supported' if it shares enough vocabulary with some
        chunk. Crude on purpose — it exists so the pipeline always runs,
        not to replace the LLM judge."""

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        claims: List[ClaimVerdict] = []

        def tokenize(text: str) -> set:
            return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS

        for sentence in sentences:
            sent_tokens = tokenize(sentence)
            best_chunk_id, best_overlap = None, 0.0
            for chunk in chunks:
                chunk_tokens = tokenize(chunk.text)
                if not sent_tokens or not chunk_tokens:
                    continue
                overlap = len(sent_tokens & chunk_tokens) / len(sent_tokens)
                if overlap > best_overlap:
                    best_overlap, best_chunk_id = overlap, chunk.id

            supported = best_overlap >= 0.5
            claims.append(
                ClaimVerdict(
                    claim=sentence,
                    supported=supported,
                    chunk_id=best_chunk_id if supported else None,
                    note=None if supported else f"best lexical overlap {best_overlap:.2f} < 0.5",
                )
            )

        score = sum(1 for c in claims if c.supported) / len(claims) if claims else 0.0
        return JudgeVerdict(
            faithfulness_score=score,
            claims=claims,
            reasoning="Heuristic lexical-overlap scoring (no ANTHROPIC_API_KEY set).",
            judge_backend="heuristic",
        )


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "in",
    "on", "for", "with", "it", "that", "this", "as", "by", "be", "at", "or",
    "from", "has", "have", "had", "its", "which", "these", "those", "can",
}
