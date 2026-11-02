"""Offline regression tests for evaluation-only custom retrieval capture."""
from __future__ import annotations

import json

import pytest

from evals.metrics.retrieval_capture import (
    RetrievalTraceCollector,
    summarize_captured_retrieval,
)
from evals.reporters.retrieval_metrics import write_retrieval_reports


def test_single_retrieval_call_preserves_rank_and_only_exposes_needed_fields():
    collector = RetrievalTraceCollector()
    collector.on_chunks([
        {"doc_id": None, "text": "other", "metadata": {"secret": "private"}},
        {"doc_id": "gold", "text": "answer", "score": 0.95},
        {"doc_id": "gold", "text": "repeated", "score": 0.6},
    ])
    assert collector.trace() == {
        "status": "captured",
        "retrieved": [
            {"doc_id": None, "text": "other"},
            {"doc_id": "gold", "text": "answer"},
            {"doc_id": "gold", "text": "repeated"},
        ],
    }


def test_missing_or_multiple_calls_cannot_be_misrepresented_as_one_ranking():
    never_called = RetrievalTraceCollector()
    assert never_called.trace() == {"status": "no_retrieval_call"}
    many_calls = RetrievalTraceCollector()
    many_calls.on_chunks([{"doc_id": "gold", "text": "one"}])
    many_calls.on_chunks([{"doc_id": "gold", "text": "two"}])
    assert many_calls.trace() == {"status": "multiple_retrieval_calls"}


@pytest.mark.parametrize("bad_chunks", [
    [None],
    [{"doc_id": 123, "text": "source"}],
    [{"doc_id": "gold", "text": ["not a string"]}],
])
def test_invalid_retrieval_chunks_never_receive_scores(bad_chunks):
    collector = RetrievalTraceCollector()
    collector.on_chunks(bad_chunks)
    assert collector.trace() == {"status": "invalid_retrieval_chunks"}


def test_reporting_only_scores_labeled_unambiguous_examples_without_source_text(tmp_path):
    sensitive = "confidential customer text not for reports"
    good = RetrievalTraceCollector()
    good.on_chunks([
        {"doc_id": None, "text": "irrelevant"},
        {"doc_id": "gold", "text": sensitive},
    ])
    report, coverage = summarize_captured_retrieval([
        {"id": "example-1", "source_id": "gold", "citation_text": sensitive, "retrieval_trace": good.trace()},
        {"id": "no-label", "source_id": None, "retrieval_trace": good.trace()},
        {"id": "multi-call", "source_id": "gold", "retrieval_trace": {"status": "multiple_retrieval_calls"}},
        {"id": "unsupported", "source_id": "gold", "retrieval_trace": {"status": "unsupported_backend"}},
        {"id": "missing-trace", "source_id": "gold"},
    ], top_k=2)
    assert report is not None
    assert coverage["total_examples"] == 5
    assert coverage["scored_examples"] == 1
    assert coverage["skipped_examples"] == 4
    assert coverage["skipped_reasons"] == {
        "missing_gold_document_id": 1,
        "multiple_retrieval_calls": 1,
        "trace_unavailable": 1,
        "unsupported_backend": 1,
    }
    assert report["averages"]["hit@k"] == 1.0
    assert report["averages"]["precision@k"] == 0.5
    assert report["averages"]["MRR"] == 0.5
    jpath, mpath = write_retrieval_reports(report, tmp_path)
    assert sensitive not in jpath.read_text(encoding="utf-8")
    assert sensitive not in mpath.read_text(encoding="utf-8")
    assert json.loads(jpath.read_text(encoding="utf-8"))["num_examples"] == 1


def test_no_scored_rows_produce_only_coverage_not_fake_zero_average():
    report, coverage = summarize_captured_retrieval([
        {"id": "empty", "source_id": "gold", "retrieval_trace": {"status": "no_retrieval_call"}},
    ], top_k=8)
    assert report is None
    assert coverage["scored_examples"] == 0
    assert coverage["skipped_reasons"] == {"no_retrieval_call": 1}


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5, "8"])
def test_capture_rejects_invalid_top_k(bad_k):
    with pytest.raises(ValueError, match="positive integer"):
        summarize_captured_retrieval([], top_k=bad_k)


def test_empty_real_retrieval_call_scores_no_hit_not_missing():
    trace = RetrievalTraceCollector()
    trace.on_chunks([])
    report, coverage = summarize_captured_retrieval(
        [{"id": "empty-retrieved", "source_id": "gold", "retrieval_trace": trace.trace()}],
        top_k=3,
    )
    assert report["averages"]["hit@k"] == 0.0
    assert coverage["scored_examples"] == 1
