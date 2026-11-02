"""Offline regression tests for privacy-safe QA quality aggregates."""
from __future__ import annotations

from math import nan

import pytest

from evals.reporters.quality_summary import summarize_qa_quality


def _rows():
    judged = [
        {"judge": {"decision": "pass"}, "latency_ms": 10,
         "question": "sensitive question 123", "answer": "sensitive answer 456"},
        {"judge": {"decision": "pass"}, "latency_ms": 20},
        {"judge": {"decision": "fail"}, "latency_ms": 100},
        {"judge": {"decision": "fail"}, "latency_ms": 200},
    ]
    scored = [
        {"em": 1.0, "f1": 1.0, "latency_ms": 10, "answer": "do not persist"},
        {"em": 0.0, "f1": 0.5, "latency_ms": 20},
        {"em": 0.0, "f1": 0.0, "latency_ms": 100},
        {"em": 0.0, "f1": 0.5, "latency_ms": 200},
    ]
    return judged, scored


def test_summary_reconciles_judge_em_f1_and_nearest_rank_latency():
    judged, scored = _rows()
    result = summarize_qa_quality(judged, scored, pass_threshold=0.6)
    assert result["source"] == "user_mode_single_pass"
    assert result["examples"] == 4
    assert result["judge"] == {
        "passed": 2, "failed": 2, "pass_rate": 0.5,
        "pass_threshold": 0.6, "threshold_pass": False,
    }
    assert result["answer_quality"] == {
        "mean_exact_match": 0.25,
        "mean_token_f1": 0.5,
        "judge_pass_not_exact": 1,
    }
    assert result["latency_ms"] == {
        "mean": 82.5,
        "p50": 60.0,
        "p95": 200,
        "max": 200,
        "percentile_method": "nearest_rank",
    }
    assert "sensitive" not in str(result)
    assert "do not persist" not in str(result)


def test_single_example_has_identical_p50_and_p95():
    judged, scored = _rows()
    summary = summarize_qa_quality(judged[:1], scored[:1], pass_threshold=1)
    assert summary["latency_ms"]["p50"] == 10
    assert summary["latency_ms"]["p95"] == 10
    assert summary["judge"]["threshold_pass"] is True


def test_invalid_alignment_and_payload_fail_closed():
    judged, scored = _rows()
    with pytest.raises(ValueError, match="nonempty aligned"):
        summarize_qa_quality([], [], pass_threshold=0.5)
    with pytest.raises(ValueError, match="nonempty aligned"):
        summarize_qa_quality(judged, scored[:-1], pass_threshold=0.5)
    bad = [dict(x) for x in judged]
    bad[0] = dict(bad[0], judge={"decision": "unknown"})
    with pytest.raises(ValueError, match="invalid judge decision"):
        summarize_qa_quality(bad, scored, pass_threshold=0.5)
    bad = [dict(x) for x in scored]
    bad[0] = dict(bad[0], latency_ms=-1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        summarize_qa_quality(judged, bad, pass_threshold=0.5)
    bad = [dict(x) for x in scored]
    bad[0] = dict(bad[0], f1=nan)
    with pytest.raises(ValueError, match="finite EM/F1"):
        summarize_qa_quality(judged, bad, pass_threshold=0.5)
    bad = [dict(x) for x in scored]
    bad[0] = dict(bad[0], latency_ms=12)
    with pytest.raises(ValueError, match="inconsistent latency"):
        summarize_qa_quality(judged, bad, pass_threshold=0.5)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, nan, True, "0.5"])
def test_invalid_threshold_rejected(threshold):
    judged, scored = _rows()
    with pytest.raises(ValueError, match="pass_threshold"):
        summarize_qa_quality(judged, scored, pass_threshold=threshold)
