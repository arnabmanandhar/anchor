"""
The orchestrator. Point this at any pipeline function and a dataset, and it
computes retrieval metrics + faithfulness score for every case, traces every
step, and persists the results — so you can diff two pipeline versions
before you ship either one.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, List, Optional, Tuple

from .judge import FaithfulnessJudge
from .retrieval_metrics import all_retrieval_metrics
from .schema import Chunk, JudgeVerdict, RunResult
from .tracer import Tracer

PipelineFn = Callable[[str], Tuple[List[Chunk], str]]
# A pipeline function takes a query string and returns (retrieved_chunks, answer).


ProgressFn = Callable[[int, int], None]
# Optional progress callback: called as progress(done, total) after each case.


def _evaluate_case(
    case: dict,
    produce: Callable[[], Tuple[List[Chunk], str]],
    produce_span: str,
    pipeline_version: str,
    k: int,
    tracer: Tracer,
    judge: FaithfulnessJudge,
) -> RunResult:
    run_id = str(uuid.uuid4())[:8]
    query = case["query"]
    t0 = time.perf_counter()

    with tracer.span(run_id, produce_span) as s:
        retrieved_chunks, answer = produce()
        retrieved_chunks = [c if isinstance(c, Chunk) else Chunk(**c) for c in retrieved_chunks]
        s.metadata = {"num_chunks": len(retrieved_chunks)}

    retrieval_metrics = None
    if case.get("reference_chunk_ids"):
        with tracer.span(run_id, "retrieval_metrics"):
            retrieved_ids = [c.id for c in retrieved_chunks]
            retrieval_metrics = all_retrieval_metrics(retrieved_ids, case["reference_chunk_ids"], k=k)

    with tracer.span(run_id, "faithfulness_judge") as s:
        judge_verdict: JudgeVerdict = judge.score(query, retrieved_chunks, answer)
        s.metadata = {"backend": judge_verdict.judge_backend}

    latency_ms = (time.perf_counter() - t0) * 1000
    # Judge cost from real token usage (0 for the heuristic backend).
    cost_usd = judge.estimate_cost(judge_verdict)

    result = RunResult(
        run_id=run_id,
        pipeline_version=pipeline_version,
        query=query,
        generated_answer=answer,
        retrieved_chunks=retrieved_chunks,
        reference_chunk_ids=case.get("reference_chunk_ids"),
        retrieval=retrieval_metrics,
        judge=judge_verdict,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )
    tracer.record_run(result)
    return result


def evaluate_pipeline(
    pipeline_fn: PipelineFn,
    dataset: List[dict],
    pipeline_version: str = "v1",
    k: int = 5,
    tracer: Optional[Tracer] = None,
    judge: Optional[FaithfulnessJudge] = None,
    db_path: str = "anchor_traces.db",
    progress: Optional[ProgressFn] = None,
) -> List[RunResult]:
    """
    dataset: list of dicts like:
        {"query": "...", "reference_chunk_ids": ["c1","c3"], "reference_answer": "..."}
    reference_chunk_ids and reference_answer are optional. If reference_chunk_ids
    is missing, retrieval metrics are skipped for that case (judge score still runs).
    """
    tracer = tracer or Tracer(db_path)
    judge = judge or FaithfulnessJudge()
    results: List[RunResult] = []

    for i, case in enumerate(dataset, start=1):
        results.append(
            _evaluate_case(
                case,
                lambda: pipeline_fn(case["query"]),
                "retrieval+generation",
                pipeline_version,
                k,
                tracer,
                judge,
            )
        )
        if progress:
            progress(i, len(dataset))

    return results


def evaluate_records(
    records: List[dict],
    pipeline_version: str = "v1",
    k: int = 5,
    tracer: Optional[Tracer] = None,
    judge: Optional[FaithfulnessJudge] = None,
    db_path: str = "anchor_traces.db",
    progress: Optional[ProgressFn] = None,
) -> List[RunResult]:
    """Score outputs your pipeline already produced, without calling it.

    records: list of dicts (see anchor.records.normalize_record for accepted aliases):
        {"query": "...",
         "retrieved_chunks": [{"id": "doc_12", "text": "..."}, ...],   # ranked, best first
         "answer": "...",
         "reference_chunk_ids": ["doc_12"]}                            # optional
    Latency in these runs measures Anchor's scoring only, not your pipeline.
    """
    from .records import normalize_record

    tracer = tracer or Tracer(db_path)
    judge = judge or FaithfulnessJudge()
    results: List[RunResult] = []

    for i, raw in enumerate(records, start=1):
        rec = normalize_record(raw, index=i)
        results.append(
            _evaluate_case(
                rec,
                lambda: (rec["retrieved_chunks"], rec["answer"]),
                "precomputed_output",
                pipeline_version,
                k,
                tracer,
                judge,
            )
        )
        if progress:
            progress(i, len(records))

    return results


def summarize(results: List[RunResult]) -> dict:
    """Aggregate a list of RunResults into mean metrics — the numbers you'd
    actually put in a PR description or a Slack message."""
    n = len(results)
    if n == 0:
        return {}

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_runs": n,
        "mean_faithfulness": mean([r.judge.faithfulness_score for r in results]),
        "mean_recall_at_k": mean([r.retrieval.recall_at_k for r in results if r.retrieval]),
        "mean_mrr": mean([r.retrieval.mrr for r in results if r.retrieval]),
        "mean_ndcg_at_k": mean([r.retrieval.ndcg_at_k for r in results if r.retrieval]),
        "mean_map": mean([r.retrieval.map_score for r in results if r.retrieval]),
        "mean_latency_ms": mean([r.latency_ms for r in results]),
        "total_cost_usd": sum(r.cost_usd for r in results),
    }
