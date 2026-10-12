"""
Core data models. Every RAG pipeline in the world can be squeezed into this
shape: a query goes in, chunks come back ranked, an answer gets generated.
That's the only contract Anchor needs to evaluate and trace a pipeline it
has never seen before.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Chunk(BaseModel):
    """A single retrieved chunk, ranked by position in the list it came in."""

    id: str
    text: str
    score: Optional[float] = None


class RunInput(BaseModel):
    """One evaluation case: what went in, what came back."""

    run_id: str
    pipeline_version: str = "v1"
    query: str
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    generated_answer: str

    # Optional ground truth. If provided, deterministic retrieval metrics
    # are computed. If omitted, only faithfulness scoring runs.
    reference_chunk_ids: Optional[List[str]] = None
    reference_answer: Optional[str] = None

    metadata: Dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)


class ClaimVerdict(BaseModel):
    """A single atomic claim extracted from the generated answer, and
    whether the judge could find it grounded in a specific retrieved chunk."""

    claim: str
    supported: bool
    chunk_id: Optional[str] = None
    note: Optional[str] = None


class JudgeVerdict(BaseModel):
    """Structured output of the faithfulness judge for one run."""

    faithfulness_score: float  # 0.0 - 1.0, fraction of claims grounded
    claims: List[ClaimVerdict] = Field(default_factory=list)
    reasoning: Optional[str] = None
    judge_backend: str = "heuristic"  # "llm" or "heuristic"
    judge_model: Optional[str] = None
    fallback_reason: Optional[str] = None  # set when the LLM judge failed and the heuristic ran instead
    input_tokens: int = 0
    output_tokens: int = 0


class RetrievalMetrics(BaseModel):
    recall_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    map_score: Optional[float] = None
    k: int = 5


class RunResult(BaseModel):
    """Everything computed for one RunInput: metrics + judge verdict + trace id."""

    run_id: str
    pipeline_version: str
    query: str
    generated_answer: str = ""
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    reference_chunk_ids: Optional[List[str]] = None
    retrieval: Optional[RetrievalMetrics] = None
    judge: JudgeVerdict
    latency_ms: float
    cost_usd: float = 0.0
    created_at: str = Field(default_factory=_now)
