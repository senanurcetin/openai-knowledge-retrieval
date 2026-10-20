from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz


def extract_spans(
    answer: str, chunks: list[dict[str, Any]], max_per_source: int = 3
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    if max_per_source <= 0:
        return citations

    counts_by_source: dict[str, int] = {}
    for c in chunks:
        doc_id = c["doc_id"]
        if counts_by_source.get(doc_id, 0) >= max_per_source:
            continue
        text = c["text"]
        # fuzzy match a sentence from chunk to the answer
        score = fuzz.partial_ratio(answer, text)
        if score > 50:
            citations.append(
                {
                    "doc_id": doc_id,
                    "title": c.get("title") or c.get("metadata", {}).get("title"),
                    "page": c.get("page"),
                    "span_start": -1,
                    "span_end": -1,
                }
            )
            counts_by_source[doc_id] = counts_by_source.get(doc_id, 0) + 1
    return citations
