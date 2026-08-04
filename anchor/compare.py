"""
Side-by-side comparison of two pipeline versions, read straight back out of
the trace database. This is the "should I ship this" answer.
"""

from __future__ import annotations

import json
from typing import Dict

from .tracer import Tracer


def compare_versions(db_path: str, version_a: str, version_b: str) -> Dict:
    tracer = Tracer(db_path)
    runs_a = tracer.fetch_runs(version_a)
    runs_b = tracer.fetch_runs(version_b)
    tracer.close()

    def summarize(runs):
        if not runs:
            return {"n_runs": 0}
        faith = [json.loads(r["judge_json"])["faithfulness_score"] for r in runs]
        recalls, mrrs, ndcgs, maps_ = [], [], [], []
        for r in runs:
            retrieval = json.loads(r["retrieval_json"]) if r["retrieval_json"] else None
            if retrieval:
                recalls.append(retrieval["recall_at_k"])
                mrrs.append(retrieval["mrr"])
                ndcgs.append(retrieval["ndcg_at_k"])
                maps_.append(retrieval["map_score"])
        lat = [r["latency_ms"] for r in runs]

        def mean(vals):
            return sum(vals) / len(vals) if vals else None

        return {
            "n_runs": len(runs),
            "mean_faithfulness": mean(faith),
            "mean_recall_at_k": mean(recalls),
            "mean_mrr": mean(mrrs),
            "mean_ndcg_at_k": mean(ndcgs),
            "mean_map": mean(maps_),
            "mean_latency_ms": mean(lat),
        }

    summary_a, summary_b = summarize(runs_a), summarize(runs_b)
    diff = {}
    for key in ("mean_faithfulness", "mean_recall_at_k", "mean_mrr", "mean_ndcg_at_k", "mean_map", "mean_latency_ms"):
        a, b = summary_a.get(key), summary_b.get(key)
        diff[key] = (b - a) if (a is not None and b is not None) else None

    return {version_a: summary_a, version_b: summary_b, "diff (b - a)": diff}
