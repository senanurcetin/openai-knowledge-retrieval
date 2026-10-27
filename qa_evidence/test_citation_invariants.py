"""Independent, offline QA for per-document citation limits in PR #9.

Checks stable cross-document order, match-only quotas, output metadata, and
read-only treatment of input chunks. This is not a benchmark or live API test.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict

import pytest
from rapidfuzz import fuzz

from retrieval.citations import extract_spans


def make_chunks(seed: int) -> list[dict[str, object]]:
    randomizer = random.Random(seed)
    document_ids = ["alpha", "beta", "gamma", "delta", "epsilon"]
    rows: list[dict[str, object]] = []
    for page in range(1, 36):
        document_id = randomizer.choice(document_ids)
        matching = randomizer.randrange(5) != 0
        rows.append(
            {
                "doc_id": document_id,
                "text": "citation needle" if matching else "zzzzzzzzzzzzzzzz",
                "page": page,
                "metadata": {"title": f"Metadata {document_id}"},
                "title": f"Top {document_id}" if page % 3 == 0 else None,
            }
        )
    randomizer.shuffle(rows)
    return rows


def independent_reference(
    answer: str, rows: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    """Select earliest matched indexes per document, then restore global order."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    if limit <= 0:
        return []

    for index, row in enumerate(rows):
        # Group all matching positions before selecting each source's quota.
        if fuzz.partial_ratio(answer, row["text"]) > 50:
            by_doc[row["doc_id"]].append(index)
    chosen = {idx for indexes in by_doc.values() for idx in indexes[:limit]}
    return [
        {
            "doc_id": row["doc_id"],
            "title": row.get("title") or row["metadata"].get("title"),
            "page": row.get("page"),
            "span_start": -1,
            "span_end": -1,
        }
        for idx, row in enumerate(rows)
        if idx in chosen
    ]


@pytest.mark.parametrize("seed", range(24))
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 5, 99])
def test_interleaved_source_quotas_match_independent_reference(seed, limit):
    rows = make_chunks(seed)
    before = copy.deepcopy(rows)
    actual = extract_spans("citation needle", rows, max_per_source=limit)
    expected = independent_reference("citation needle", rows, limit)
    assert actual == expected
    assert rows == before


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_very_long_matching_source_cannot_starve_late_sources(limit):
    chunks = [{"doc_id": "first", "text": "needle", "page": index} for index in range(800)]
    chunks.extend(
        {"doc_id": "late", "text": "needle", "page": index}
        for index in range(1000, 1000 + limit + 2)
    )
    citations = extract_spans("needle", chunks, max_per_source=limit)
    assert [citation["page"] for citation in citations] == [
        *range(limit),
        *range(1000, 1000 + limit),
    ]


@pytest.mark.parametrize("limit", [0, -1, -50])
def test_nonpositive_limit_returns_without_inspecting_invalid_chunks(limit):
    assert extract_spans("needle", [{"not_even_a_document": True}], limit) == []


def test_chunk_without_a_title_or_metadata_returns_null_title():
    result = extract_spans("needle", [{"doc_id": "source", "text": "needle"}], max_per_source=1)
    assert result == [
        {
            "doc_id": "source",
            "title": None,
            "page": None,
            "span_start": -1,
            "span_end": -1,
        }
    ]


def test_nonmatching_chunks_do_not_spend_the_quota():
    rows = [
        {"doc_id": "alpha", "text": "zzzzzzzz", "page": 1},
        {"doc_id": "beta", "text": "needle", "page": 2},
        {"doc_id": "alpha", "text": "needle", "page": 3},
        {"doc_id": "alpha", "text": "needle", "page": 4},
    ]
    out = extract_spans("needle", rows, max_per_source=1)
    assert [(row["doc_id"], row["page"]) for row in out] == [
        ("beta", 2),
        ("alpha", 3),
    ]
