"""
Anchor — a deterministic-first evaluation and tracing layer for RAG pipelines.

Retrieval metrics are pure math (no model calls, fully reproducible).
Faithfulness scoring uses an LLM judge only where judgment is genuinely required,
and every judge verdict must cite the exact chunk it is grounded in.
Every run is captured as a trace (spans with latency, plus judge token cost) so regressions
are visible before they ship, not after.
"""

from .schema import Chunk, RunInput, JudgeVerdict, ClaimVerdict, RetrievalMetrics, RunResult
from .retrieval_metrics import recall_at_k, mrr, ndcg_at_k, map_score, all_retrieval_metrics
from .judge import FaithfulnessJudge
from .tracer import Tracer, Span
from .runner import evaluate_pipeline, evaluate_records, summarize
from .records import load_records
from .compare import compare_versions, significance
from .stats import paired_test, bootstrap_ci, min_detectable_effect
from .gate import run_gate, GateResult

__all__ = [
    "Chunk",
    "RunInput",
    "JudgeVerdict",
    "ClaimVerdict",
    "RetrievalMetrics",
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
    "evaluate_records",
    "load_records",
    "summarize",
    "compare_versions",
    "significance",
    "paired_test",
    "bootstrap_ci",
    "min_detectable_effect",
    "run_gate",
    "GateResult",
]

__version__ = "0.4.0"
