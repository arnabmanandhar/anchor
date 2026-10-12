"""
Regression gate: turn a pipeline version's stored metrics into a pass/fail
answer a CI job can act on.

Two kinds of check, both optional:
  - absolute floors   e.g. mean recall@k must be >= 0.8
  - relative to a baseline version   e.g. mean faithfulness may not drop
                                     more than 0.05 below "main"

Retrieval floors are only as deterministic as your pipeline is; with the
heuristic judge, faithfulness is deterministic too, so a gate run with no
API key is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .compare import summarize_runs
from .tracer import Tracer

# CLI flag name -> summary key
GATE_METRICS = {
    "faithfulness": "mean_faithfulness",
    "recall": "mean_recall_at_k",
    "mrr": "mean_mrr",
    "ndcg": "mean_ndcg_at_k",
    "map": "mean_map",
}


@dataclass
class GateResult:
    version: str
    passed: bool
    checks: List[Dict] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"Gate for {self.version!r}: {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            mark = "ok  " if c["passed"] else "FAIL"
            lines.append(f"  [{mark}] {c['description']}")
        return "\n".join(lines)


def run_gate(
    db_path: str,
    version: str,
    floors: Optional[Dict[str, float]] = None,
    baseline: Optional[str] = None,
    max_drop: Optional[Dict[str, float]] = None,
) -> GateResult:
    """floors and max_drop are keyed by GATE_METRICS names (faithfulness, recall, ...)."""
    floors = floors or {}
    max_drop = max_drop or {}
    for name in list(floors) + list(max_drop):
        if name not in GATE_METRICS:
            raise ValueError(f"unknown gate metric {name!r}; choose from {sorted(GATE_METRICS)}")

    tracer = Tracer(db_path)
    try:
        summary = summarize_runs(tracer.fetch_runs(version))
        base_summary = summarize_runs(tracer.fetch_runs(baseline)) if baseline else None
    finally:
        tracer.close()

    checks: List[Dict] = []
    if summary["n_runs"] == 0:
        checks.append({"passed": False, "description": f"no runs found for version {version!r}"})
        return GateResult(version, False, checks)

    for name, floor in floors.items():
        value = summary.get(GATE_METRICS[name])
        ok = value is not None and value >= floor
        shown = "n/a" if value is None else f"{value:.3f}"
        checks.append({"passed": ok, "description": f"{name} {shown} >= {floor}"})

    if baseline:
        if base_summary["n_runs"] == 0:
            checks.append({"passed": False, "description": f"no runs found for baseline {baseline!r}"})
        for name, allowed in max_drop.items():
            key = GATE_METRICS[name]
            value, base = summary.get(key), base_summary.get(key)
            if value is None or base is None:
                checks.append({"passed": False, "description": f"{name}: missing on {version!r} or {baseline!r}"})
                continue
            drop = base - value
            checks.append(
                {
                    "passed": drop <= allowed,
                    "description": f"{name} {value:.3f} vs {baseline} {base:.3f} (drop {drop:+.3f}, allowed {allowed})",
                }
            )

    return GateResult(version, all(c["passed"] for c in checks), checks)
