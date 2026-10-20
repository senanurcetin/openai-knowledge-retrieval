from retrieval.citations import extract_spans


def _chunk(doc_id, text="answer", page=1, **kwargs):
    return {"doc_id": doc_id, "text": text, "page": page, **kwargs}


def test_max_per_source_does_not_truncate_other_documents():
    chunks = [
        _chunk("a", page=1),
        _chunk("a", page=2),
        _chunk("a", page=3),
        _chunk("b", page=4),
        _chunk("b", page=5),
    ]

    citations = extract_spans("answer", chunks, max_per_source=2)

    assert [(c["doc_id"], c["page"]) for c in citations] == [
        ("a", 1),
        ("a", 2),
        ("b", 4),
        ("b", 5),
    ]


def test_nonmatching_chunk_does_not_consume_source_quota():
    chunks = [
        _chunk("a", text="zzzzzzzzzzzz"),
        _chunk("a", page=2, metadata={"title": "Source title"}),
        _chunk("b", page=3, title="Other title"),
    ]

    citations = extract_spans("answer", chunks, max_per_source=1)

    assert [(c["doc_id"], c["page"]) for c in citations] == [("a", 2), ("b", 3)]
    assert [c["title"] for c in citations] == ["Source title", "Other title"]


def test_nonpositive_per_source_limit_returns_no_citations():
    for limit in (0, -1):
        assert extract_spans("answer", [_chunk("a")], max_per_source=limit) == []
