"""
Anchor — a deterministic-first evaluation and tracing layer for RAG pipelines.

Retrieval metrics are pure math (no model calls, fully reproducible).
Faithfulness scoring uses an LLM judge (any provider, or an offline heuristic) only
where judgment is genuinely required, and every verdict must cite the exact chunk
it is grounded in.
Every run is captured as a trace (spans with latency, plus judge token cost) so regressions
are visible before they ship, not after.
"""

from .compare import compare_versions, significance
from .gate import GateResult, run_gate
from .judge import FaithfulnessJudge
from .llm import PROVIDERS, Connection, LLMError, Secret
from .records import load_records
from .retrieval_metrics import all_retrieval_metrics, hit_at_k, map_score, mrr, ndcg_at_k, precision_at_k, recall_at_k
from .runner import evaluate_pipeline, evaluate_records, summarize
from .schema import Chunk, ClaimVerdict, JudgeVerdict, RetrievalMetrics, RunInput, RunResult
from .stats import bootstrap_ci, min_detectable_effect, paired_test
from .synth import generate_questions
from .tracer import Span, Tracer

__all__ = [
    "Chunk",
    "RunInput",
    "JudgeVerdict",
    "ClaimVerdict",
    "RetrievalMetrics",
    "RunResult",
    "recall_at_k",
    "precision_at_k",
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "map_score",
    "all_retrieval_metrics",
    "FaithfulnessJudge",
    "Connection",
    "Secret",
    "LLMError",
    "PROVIDERS",
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
    "generate_questions",
]

__version__ = "0.6.0"
