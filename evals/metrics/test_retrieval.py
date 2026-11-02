"""Offline regression tests for retrieval ranking metrics."""

from evals.metrics.retrieval import compute_retrieval_metrics


def test_missing_document_id_does_not_shift_gold_into_top_k():
    results = [
        {"doc_id": None, "text": "unidentified first search result"},
        {"doc_id": "gold", "text": "gold document at actual rank two"},
    ]
    metrics = compute_retrieval_metrics(results, "gold", "", top_k=1)
    assert metrics["hit@k"] == 0.0
    assert metrics["recall@k"] == 0.0
    assert metrics["MRR"] == 0.5


def test_multiple_unidentified_results_preserve_reciprocal_rank():
    results = [
        {"text": "unidentified one"},
        {"doc_id": "", "text": "unidentified two"},
        {"doc_id": "gold", "text": "actual third result"},
    ]
    metrics = compute_retrieval_metrics(results, "gold", "", top_k=2)
    assert metrics["hit@k"] == 0.0
    assert metrics["recall@k"] == 0.0
    assert metrics["MRR"] == 1 / 3


def test_fully_identified_results_keep_existing_scores():
    results = [
        {"doc_id": "irrelevant", "text": "first"},
        {"doc_id": "gold", "text": "second"},
    ]
    metrics = compute_retrieval_metrics(results, "gold", "", top_k=2)
    assert metrics["hit@k"] == 1.0
    assert metrics["recall@k"] == 1.0
    assert metrics["MRR"] == 0.5
