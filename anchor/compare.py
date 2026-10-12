"""
Side-by-side comparison of two pipeline versions, read straight back out of
the trace database. This is the "should I ship this" answer.

Besides the aggregate means, the comparison lines up runs by query text so
you can see *which* questions got better or worse — a flat mean can hide a
regression on one query that another query's improvement cancels out.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .stats import paired_test
from .tracer import Tracer

METRIC_KEYS = ("mean_faithfulness", "mean_recall_at_k", "mean_mrr", "mean_ndcg_at_k", "mean_map", "mean_latency_ms")

# Per-query metrics used to flag regressions/improvements. Latency is left
# out on purpose: it is noisy run to run and would flag every query.
_PER_QUERY_KEYS = ("faithfulness", "recall_at_k", "mrr", "ndcg_at_k", "map_score")


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _run_metrics(run: Dict) -> Dict[str, Optional[float]]:
    retrieval = json.loads(run["retrieval_json"]) if run["retrieval_json"] else None
    out = {"faithfulness": json.loads(run["judge_json"])["faithfulness_score"]}
    for key in ("recall_at_k", "mrr", "ndcg_at_k", "map_score"):
        out[key] = retrieval[key] if retrieval else None
    out["latency_ms"] = run["latency_ms"]
    return out


def summarize_runs(runs: List[Dict]) -> Dict:
    if not runs:
        return {"n_runs": 0}
    metrics = [_run_metrics(r) for r in runs]
    return {
        "n_runs": len(runs),
        "mean_faithfulness": _mean([m["faithfulness"] for m in metrics]),
        "mean_recall_at_k": _mean([m["recall_at_k"] for m in metrics]),
        "mean_mrr": _mean([m["mrr"] for m in metrics]),
        "mean_ndcg_at_k": _mean([m["ndcg_at_k"] for m in metrics]),
        "mean_map": _mean([m["map_score"] for m in metrics]),
        "mean_latency_ms": _mean([m["latency_ms"] for m in metrics]),
    }


def per_query_diff(runs_a: List[Dict], runs_b: List[Dict], tolerance: float = 1e-9) -> List[Dict]:
    """Match runs by query text (the latest run per query wins, so re-runs
    don't double count) and report the change in each metric."""
    latest_a = {r["query"]: _run_metrics(r) for r in runs_a}
    latest_b = {r["query"]: _run_metrics(r) for r in runs_b}

    rows = []
    for query in sorted(set(latest_a) & set(latest_b)):
        a, b = latest_a[query], latest_b[query]
        deltas = {
            k: (b[k] - a[k]) if (a[k] is not None and b[k] is not None) else None
            for k in _PER_QUERY_KEYS
        }
        changed = [d for d in deltas.values() if d is not None and abs(d) > tolerance]
        if not changed:
            status = "unchanged"
        elif all(d > 0 for d in changed):
            status = "improved"
        elif all(d < 0 for d in changed):
            status = "regressed"
        else:
            status = "mixed"
        rows.append({"query": query, "status": status, "a": a, "b": b, "delta": deltas})
    return rows


def significance(runs_a: List[Dict], runs_b: List[Dict], alpha: float = 0.05) -> Dict[str, Dict]:
    """Paired significance test per metric over the questions both versions
    answered (latest run per query, as in per_query_diff). Keyed like
    _PER_QUERY_KEYS; see stats.paired_test for the fields."""
    latest_a = {r["query"]: _run_metrics(r) for r in runs_a}
    latest_b = {r["query"]: _run_metrics(r) for r in runs_b}
    shared = sorted(set(latest_a) & set(latest_b))
    return {
        key: paired_test([latest_a[q][key] for q in shared], [latest_b[q][key] for q in shared], alpha=alpha)
        for key in _PER_QUERY_KEYS
    }


def compare_versions(db_path: str, version_a: str, version_b: str, alpha: float = 0.05) -> Dict:
    tracer = Tracer(db_path)
    runs_a = tracer.fetch_runs(version_a)
    runs_b = tracer.fetch_runs(version_b)
    tracer.close()

    summary_a, summary_b = summarize_runs(runs_a), summarize_runs(runs_b)
    diff = {}
    for key in METRIC_KEYS:
        a, b = summary_a.get(key), summary_b.get(key)
        diff[key] = (b - a) if (a is not None and b is not None) else None

    queries = per_query_diff(runs_a, runs_b)
    return {
        version_a: summary_a,
        version_b: summary_b,
        "diff (b - a)": diff,
        "regressed_queries": [q["query"] for q in queries if q["status"] in ("regressed", "mixed")],
        "improved_queries": [q["query"] for q in queries if q["status"] == "improved"],
        "significance": significance(runs_a, runs_b, alpha=alpha),
    }
