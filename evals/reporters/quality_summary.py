"""Aggregate user-mode RAG evaluation quality without exposing raw examples."""
from __future__ import annotations

from math import ceil, isfinite
from statistics import fmean, median
from typing import Any


def summarize_qa_quality(
    judged_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    *,
    pass_threshold: float,
) -> dict[str, Any]:
    """Build an inspectable summary from the *same* evaluated answers.

    p95 uses nearest-rank (ceil(0.95 * n)), not linear interpolation.
    A judge pass paired with EM < 1 is recorded as a disagreement, not
    silently converted to a model failure.
    """
    n = len(scored_rows)
    if n == 0 or len(judged_rows) != n:
        raise ValueError("quality summary requires a nonempty aligned evaluation batch")
    if (
        isinstance(pass_threshold, bool)
        or not isinstance(pass_threshold, (int, float))
        or not isfinite(pass_threshold)
        or not 0 <= pass_threshold <= 1
    ):
        raise ValueError("pass_threshold must be a finite number from 0 to 1")

    latencies: list[int] = []
    exact: list[float] = []
    f1_scores: list[float] = []
    passed = 0
    judge_pass_not_exact = 0

    for judged, scored in zip(judged_rows, scored_rows, strict=True):
        decision = judged.get("judge", {}).get("decision")
        if decision not in ("pass", "fail"):
            raise ValueError("quality summary encountered an invalid judge decision")
        latency = scored.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
            raise ValueError("quality summary requires nonnegative integer latencies")
        if judged.get("latency_ms") != latency:
            raise ValueError("quality summary encountered inconsistent latency values")

        em = scored.get("em")
        f1 = scored.get("f1")
        for score in (em, f1):
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("quality summary requires finite EM/F1 values from 0 to 1")
        passed += decision == "pass"
        judge_pass_not_exact += decision == "pass" and em < 1
        latencies.append(latency)
        exact.append(float(em))
        f1_scores.append(float(f1))

    ordered = sorted(latencies)
    pass_rate = passed / n
    return {
        "schema_version": 1,
        "source": "user_mode_single_pass",
        "examples": n,
        "judge": {
            "passed": passed,
            "failed": n - passed,
            "pass_rate": pass_rate,
            "pass_threshold": float(pass_threshold),
            "threshold_pass": pass_rate >= pass_threshold,
        },
        "answer_quality": {
            "mean_exact_match": fmean(exact),
            "mean_token_f1": fmean(f1_scores),
            "judge_pass_not_exact": judge_pass_not_exact,
        },
        "latency_ms": {
            "mean": fmean(latencies),
            "p50": median(ordered),
            "p95": ordered[ceil(0.95 * n) - 1],
            "max": ordered[-1],
            "percentile_method": "nearest_rank",
        },
    }
