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

def test_retrieval_missing_context_text_is_handled_as_empty():
    results = [
        {"doc_id": "gold", "text": None},
        {"doc_id": "other"},
    ]
    metrics = compute_retrieval_metrics(results, "gold", "citation", top_k=2)
    assert metrics["hit@k"] == 1.0
    assert metrics["MRR"] == 1.0
    assert metrics["context_precision"] == 0.0
    assert metrics["support_coverage"] == 0.0


def test_retrieval_valid_k_still_preserves_actual_ranking():
    results = [
        {"doc_id": "other", "text": "not relevant"},
        {"doc_id": "gold", "text": "relevant"},
    ]
    assert compute_retrieval_metrics(results, "gold", "", top_k=1)["hit@k"] == 0.0
    assert compute_retrieval_metrics(results, "gold", "", top_k=2)["hit@k"] == 1.0


def test_retrieval_rejects_invalid_top_k_values():
    import pytest
    from evals.metrics.retrieval import hit_at_k, recall_at_k

    for invalid in (0, -1, -5, True, 1.5, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            compute_retrieval_metrics([{"doc_id": "gold", "text": "text"}],
                                      "gold", "text", top_k=invalid)
        with pytest.raises(ValueError, match="positive integer"):
            hit_at_k(["gold"], "gold", k=invalid)
        with pytest.raises(ValueError, match="positive integer"):
            recall_at_k(["gold"], ["gold"], k=invalid)

def test_precision_at_k_counts_unique_gold_documents_and_preserves_rank():
    from evals.metrics.retrieval import precision_at_k

    assert precision_at_k(["gold", "irrelevant"], ["gold"], k=2) == 0.5
    assert precision_at_k(["", "gold"], ["gold"], k=1) == 0.0
    assert precision_at_k(["", "gold"], ["gold"], k=2) == 0.5
    assert precision_at_k(["gold", "gold", "other"], ["gold"], k=3) == 1 / 3
    assert precision_at_k(["x", "y"], ["x", "y"], k=2) == 1.0
    assert precision_at_k([], ["gold"], k=2) == 0.0


def test_precision_at_k_rejects_invalid_k_and_empty_gold():
    from evals.metrics.retrieval import precision_at_k
    import pytest

    assert precision_at_k(["gold"], [], k=1) == 0.0
    for invalid in [0, -1, True, 1.5, "2"]:
        with pytest.raises(ValueError, match="positive integer"):
            precision_at_k(["gold"], ["gold"], k=invalid)


def test_computed_metrics_include_precision_at_k_without_shifting_unidentified_results():
    results = [
        {"doc_id": None, "text": "unidentified"},
        {"doc_id": "gold", "text": "the answer"},
        {"doc_id": "gold", "text": "duplicate gold record"},
    ]
    result = compute_retrieval_metrics(results, "gold", "", top_k=3)
    assert result["precision@k"] == 1 / 3
    assert result["hit@k"] == 1.0
    assert result["recall@k"] == 1.0
    assert result["MRR"] == 0.5
