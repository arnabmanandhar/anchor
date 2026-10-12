"""
Is a difference between two pipeline versions real, or noise?

A mean that moves by +0.04 on 12 questions often means nothing: swap a
couple of questions and it flips. Everything here works on *paired*
per-question differences (B − A on the same question), which removes the
question-to-question variation and makes small test sets far more useful.

  paired_test           mean Δ, a bootstrap confidence interval for it, and
                        a sign-flip permutation p-value
  bootstrap_ci          percentile bootstrap CI for a single mean
  min_detectable_effect smallest Δ this many questions can reliably detect

Pure Python and seeded, so the same runs always give the same numbers:
safe to use in a CI gate.
"""

from __future__ import annotations

import itertools
import math
import random
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_SEED = 0
DEFAULT_RESAMPLES = 5000
# Below this many pairs the permutation test enumerates every sign pattern
# (2^n of them) and the p-value is exact; above it, it samples.
EXACT_PERMUTATION_MAX_N = 14

# z-scores for a two-sided 5% test with 80% power, used by min_detectable_effect.
_Z_ALPHA, _Z_POWER = 1.959964, 0.841621


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted list, q in [0, 1]."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap confidence interval for the mean of values."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    if len(set(values)) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    tail = (1 - confidence) / 2
    return (_percentile(means, tail), _percentile(means, 1 - tail))


def permutation_p_value(
    deltas: Sequence[float], n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> float:
    """Two-sided sign-flip permutation test of "mean Δ = 0".

    Under the null hypothesis A and B are interchangeable on each question,
    so each Δ is equally likely to have either sign. The p-value is the
    share of sign patterns whose |mean| is at least the observed |mean|.
    """
    deltas = list(deltas)
    n = len(deltas)
    if n == 0:
        return 1.0
    observed = abs(sum(deltas)) - 1e-12
    if observed <= 0:
        return 1.0
    if n <= EXACT_PERMUTATION_MAX_N:
        hits = sum(
            abs(sum(d if s else -d for d, s in zip(deltas, signs))) >= observed
            for signs in itertools.product((True, False), repeat=n)
        )
        return hits / 2 ** n
    rng = random.Random(seed)
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in deltas)) >= observed for _ in range(n_resamples))
    # +1 on both sides keeps a sampled p-value from ever being exactly 0.
    return (hits + 1) / (n_resamples + 1)


def min_detectable_effect(deltas: Sequence[float]) -> Optional[float]:
    """Smallest true mean Δ a paired test on this many questions detects 80%
    of the time at the 5% level, given the spread of these differences.

    Normal approximation: MDE = (z_0.975 + z_0.8) · sd(Δ) / √n. It tells you
    whether "no significant change" means "no change" or "too few questions
    to tell".
    """
    deltas = [d for d in deltas if d is not None]
    if len(deltas) < 2:
        return None
    return (_Z_ALPHA + _Z_POWER) * stdev(deltas) / math.sqrt(len(deltas))


def paired_test(
    a: Sequence[Optional[float]],
    b: Sequence[Optional[float]],
    alpha: float = 0.05,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Compare paired scores a[i] (baseline) and b[i] (candidate) on the same questions.

    Pairs where either side is None are dropped. Returns n, mean_a, mean_b,
    mean_delta (b − a), ci_low/ci_high for mean_delta, p_value, significant
    (p < alpha), and mde.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be the same length (one entry per shared question)")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    out = {"n": len(pairs), "alpha": alpha}
    if not pairs:
        return {**out, "mean_a": None, "mean_b": None, "mean_delta": None, "ci_low": None,
                "ci_high": None, "p_value": None, "significant": False, "mde": None}
    deltas = [y - x for x, y in pairs]
    ci = bootstrap_ci(deltas, confidence=1 - alpha, n_resamples=n_resamples, seed=seed)
    p = permutation_p_value(deltas, n_resamples=n_resamples, seed=seed)
    return {
        **out,
        "mean_a": mean(x for x, _ in pairs),
        "mean_b": mean(y for _, y in pairs),
        "mean_delta": mean(deltas),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": p,
        "significant": p < alpha,
        "mde": min_detectable_effect(deltas),
    }


def describe(result: Dict) -> str:
    """One plain-language line for a paired_test result."""
    if not result["n"]:
        return "no shared questions to compare"
    d, lo, hi, p = result["mean_delta"], result["ci_low"], result["ci_high"], result["p_value"]
    ci = f"{int(round((1 - result['alpha']) * 100))}% CI [{lo:+.3f}, {hi:+.3f}]"
    if result["significant"]:
        verdict = "real improvement" if d > 0 else "real regression"
    elif result["mde"] is not None and result["mde"] > abs(d):
        verdict = f"not significant; {result['n']} questions can only detect changes of about ±{result['mde']:.2f}"
    else:
        verdict = "not significant"
    return f"Δ {d:+.3f}, {ci}, p = {p:.3f}: {verdict}"
