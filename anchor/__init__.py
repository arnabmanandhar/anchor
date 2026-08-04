"""
Anchor — a deterministic-first evaluation and tracing layer for RAG pipelines.

Retrieval metrics are pure math (no model calls, fully reproducible).
Faithfulness scoring uses an LLM judge only where judgment is genuinely required,
and every judge verdict must cite the exact chunk it is grounded in.
Every run is captured as a trace (spans with latency + cost) so regressions
are visible before they ship, not after.
"""

from .schema import Chunk, RunInput, JudgeVerdict, ClaimVerdict, RunResult
from .retrieval_metrics import recall_at_k, mrr, ndcg_at_k, map_score, all_retrieval_metrics
from .judge import FaithfulnessJudge
from .tracer import Tracer, Span
from .runner import evaluate_pipeline

__all__ = [
    "Chunk",
    "RunInput",
    "JudgeVerdict",
    "ClaimVerdict",
    "RunResult",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "map_score",
    "all_retrieval_metrics",
    "FaithfulnessJudge",
    "Tracer",
    "Span",
    "evaluate_pipeline",
]

__version__ = "0.1.0"
