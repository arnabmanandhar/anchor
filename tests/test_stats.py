import math

import pytest

from anchor import evaluate_pipeline, run_gate
from anchor.compare import compare_versions
from anchor.judge import FaithfulnessJudge
from anchor.schema import Chunk
from anchor.stats import bootstrap_ci, describe, min_detectable_effect, paired_test, permutation_p_value

HEURISTIC = FaithfulnessJudge(backend="heuristic")


def test_exact_permutation_p_value_by_hand():
    # Three positive deltas: only the all-plus and all-minus sign patterns
    # reach |sum| = 3, so p = 2 / 2^3.
    assert permutation_p_value([1, 1, 1]) == pytest.approx(2 / 8)
    # Opposite deltas cancel: nothing to explain.
    assert permutation_p_value([0.5, -0.5]) == 1.0
    assert permutation_p_value([]) == 1.0


def test_large_consistent_improvement_is_significant():
    a = [0.2] * 20
    b = [0.8] * 20
    r = paired_test(a, b)
    assert r["n"] == 20 and r["mean_delta"] == pytest.approx(0.6)
    assert r["significant"] and r["p_value"] < 0.001
    assert r["ci_low"] == pytest.approx(0.6) and r["ci_high"] == pytest.approx(0.6)
    assert describe(r).endswith("real improvement")


def test_noisy_small_change_is_not_significant_and_reports_mde():
    a = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    b = [1.0, 0.0, 1.0, 0.0, 1.0, 0.25]
    r = paired_test(a, b)
    assert not r["significant"]
    assert r["ci_low"] < 0 < r["ci_high"]
    assert r["mde"] > abs(r["mean_delta"])
    assert "can only detect" in describe(r)


def test_results_are_deterministic_and_drop_missing_pairs():
    a = [0.1, 0.4, None, 0.3, 0.9, 0.2, 0.5, 0.6, 0.1, 0.0, 0.7, 0.3, 0.2, 0.4, 0.5, 0.8]
    b = [0.3, 0.5, 0.9, 0.2, 1.0, 0.4, 0.6, 0.6, 0.3, 0.1, 0.9, 0.2, 0.5, 0.4, 0.7, 0.9]
    r1, r2 = paired_test(a, b), paired_test(a, b)
    assert r1 == r2 and r1["n"] == 15


def test_bootstrap_ci_contains_the_mean():
    lo, hi = bootstrap_ci([0.0, 0.5, 1.0, 0.25, 0.75])
    assert lo < 0.5 < hi
    assert bootstrap_ci([]) is None


def test_mde_formula():
    deltas = [0.0, 1.0, 0.0, 1.0]
    sd = 0.5773502691896258
    assert min_detectable_effect(deltas) == pytest.approx((1.959964 + 0.841621) * sd / math.sqrt(4))
    assert min_detectable_effect([0.3]) is None


def test_paired_test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_test([0.1], [0.1, 0.2])


def _store(db, version, ranking_for):
    def pipeline(query):
        ids = ranking_for(query)
        return [Chunk(id=i, text=f"{query} {i}") for i in ids], f"{query} {ids[0]}."

    dataset = [{"query": f"q{i}", "reference_chunk_ids": ["a"]} for i in range(8)]
    evaluate_pipeline(pipeline, dataset, pipeline_version=version, judge=HEURISTIC, db_path=db, k=1)


def test_compare_reports_significance(tmp_path):
    db = str(tmp_path / "t.db")
    _store(db, "good", lambda q: ["a", "b"])
    _store(db, "bad", lambda q: ["b", "a"])
    sig = compare_versions(db, "good", "bad")["significance"]
    assert sig["recall_at_k"]["mean_delta"] == -1.0
    assert sig["recall_at_k"]["significant"]


def test_gate_require_significant_ignores_noise_but_not_real_drops(tmp_path):
    db = str(tmp_path / "t.db")
    _store(db, "base", lambda q: ["a", "b"])
    # One question of eight misses: a 0.125 drop that a paired test can't call real.
    _store(db, "noisy", lambda q: ["b", "a"] if q == "q3" else ["a", "b"])
    _store(db, "broken", lambda q: ["b", "a"])

    strict = run_gate(db, "noisy", baseline="base", max_drop={"recall": 0.05})
    assert not strict.passed
    lenient = run_gate(db, "noisy", baseline="base", max_drop={"recall": 0.05}, require_significant=True)
    assert lenient.passed and "not significant" in lenient.report()
    assert not run_gate(db, "broken", baseline="base", max_drop={"recall": 0.05}, require_significant=True).passed
    # Floors are never relaxed.
    assert not run_gate(db, "noisy", floors={"recall": 0.95}, require_significant=True).passed
