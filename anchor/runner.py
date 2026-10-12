"""
The orchestrator. Point this at any pipeline function and a dataset, and it
computes retrieval metrics + faithfulness score for every case, traces every
step, and persists the results — so you can diff two pipeline versions
before you ship either one.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .judge import FaithfulnessJudge
from .retrieval_metrics import all_retrieval_metrics
from .schema import Chunk, JudgeVerdict, RunResult
from .tracer import Tracer

PipelineFn = Callable[[str], tuple[list[Chunk], str]]
# A pipeline function takes a query string and returns (retrieved_chunks, answer).


ProgressFn = Callable[[int, int], None]
# Optional progress callback: called as progress(done, total) after each case.


def _evaluate_case(
    case: dict,
    produce: Callable[[], tuple[list[Chunk], str]],
    produce_span: str,
    pipeline_version: str,
    k: int,
    tracer: Tracer,
    judge: FaithfulnessJudge,
) -> RunResult:
    run_id = uuid.uuid4().hex[:16]  # 64 bits: no realistic collision (8 hex chars collided on large sets)
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
        s.metadata = {"backend": judge_verdict.judge_backend, "model": judge_verdict.judge_model,
                      "cached": judge_verdict.cached}

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
    dataset: list[dict],
    pipeline_version: str = "v1",
    k: int = 5,
    tracer: Optional[Tracer] = None,
    judge: Optional[FaithfulnessJudge] = None,
    db_path: str = "anchor_traces.db",
    progress: Optional[ProgressFn] = None,
    max_workers: int = 1,
) -> list[RunResult]:
    """
    dataset: list of dicts like:
        {"query": "...", "reference_chunk_ids": ["c1","c3"], "reference_answer": "..."}
    reference_chunk_ids and reference_answer are optional. If reference_chunk_ids
    is missing, retrieval metrics are skipped for that case (judge score still runs).
    max_workers > 1 evaluates cases concurrently; only use it if pipeline_fn is thread-safe.
    """
    tracer = tracer or Tracer(db_path)
    judge = judge or FaithfulnessJudge(cache=tracer)
    jobs = [(case, (lambda case=case: pipeline_fn(case["query"]))) for case in dataset]
    return _run_all(jobs, "retrieval+generation", pipeline_version, k, tracer, judge, progress, max_workers)


def evaluate_records(
    records: list[dict],
    pipeline_version: str = "v1",
    k: int = 5,
    tracer: Optional[Tracer] = None,
    judge: Optional[FaithfulnessJudge] = None,
    db_path: str = "anchor_traces.db",
    progress: Optional[ProgressFn] = None,
    max_workers: int = 1,
) -> list[RunResult]:
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
    judge = judge or FaithfulnessJudge(cache=tracer)
    normalized = [normalize_record(raw, index=i) for i, raw in enumerate(records, start=1)]
    jobs = [(rec, (lambda rec=rec: (rec["retrieved_chunks"], rec["answer"]))) for rec in normalized]
    return _run_all(jobs, "precomputed_output", pipeline_version, k, tracer, judge, progress, max_workers)


def _run_all(jobs, produce_span, pipeline_version, k, tracer, judge, progress, max_workers) -> list[RunResult]:
    """Evaluate every (case, produce) job, in parallel when max_workers > 1
    (LLM judge calls are I/O bound, so threads give a near-linear speed-up).
    Results come back in input order whatever order they finish in."""
    total = len(jobs)
    if max_workers <= 1 or total <= 1:
        results = []
        for i, (case, produce) in enumerate(jobs, start=1):
            results.append(_evaluate_case(case, produce, produce_span, pipeline_version, k, tracer, judge))
            if progress:
                progress(i, total)
        return results
    results: list[Optional[RunResult]] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_evaluate_case, case, produce, produce_span, pipeline_version, k, tracer, judge): i
                   for i, (case, produce) in enumerate(jobs)}
        for done, fut in enumerate(as_completed(futures), start=1):
            results[futures[fut]] = fut.result()
            if progress:
                progress(done, total)
    return results


def summarize(results: list[RunResult]) -> dict:
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
        "mean_precision_at_k": mean([r.retrieval.precision_at_k for r in results if r.retrieval]),
        "hit_rate_at_k": mean([r.retrieval.hit_at_k for r in results if r.retrieval]),
        "mean_mrr": mean([r.retrieval.mrr for r in results if r.retrieval]),
        "mean_ndcg_at_k": mean([r.retrieval.ndcg_at_k for r in results if r.retrieval]),
        "mean_map": mean([r.retrieval.map_score for r in results if r.retrieval]),
        "contradiction_rate": mean([1.0 if r.judge.contradicted else 0.0 for r in results]),
        "mean_latency_ms": mean([r.latency_ms for r in results]),
        "total_cost_usd": sum(r.cost_usd for r in results),
    }
