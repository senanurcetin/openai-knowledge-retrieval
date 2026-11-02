"""Opt-in, in-memory retrieval capture and coverage accounting.

A ranking is only scored when one retrieval tool invocation produced it.
Multiple calls cannot be combined into one meaningful rank list.
"""
from __future__ import annotations

from statistics import fmean
from typing import Any

from evals.metrics.retrieval import compute_retrieval_metrics
from evals.reporters.retrieval_metrics import METRIC_KEYS


class RetrievalTraceCollector:
    def __init__(self) -> None:
        self._calls: list[list[dict[str, str | None]] | None] = []

    def on_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not isinstance(chunks, list):
            self._calls.append(None)
            return
        captured: list[dict[str, str | None]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                self._calls.append(None)
                return
            doc_id, chunk_text = chunk.get("doc_id"), chunk.get("text")
            if not (doc_id is None or isinstance(doc_id, str)):
                self._calls.append(None)
                return
            if not (chunk_text is None or isinstance(chunk_text, str)):
                self._calls.append(None)
                return
            captured.append({"doc_id": doc_id, "text": chunk_text or ""})
        self._calls.append(captured)

    def trace(self) -> dict[str, Any]:
        if not self._calls:
            return {"status": "no_retrieval_call"}
        if len(self._calls) != 1:
            return {"status": "multiple_retrieval_calls"}
        if self._calls[0] is None:
            return {"status": "invalid_retrieval_chunks"}
        return {"status": "captured", "retrieved": self._calls[0]}


def summarize_captured_retrieval(
    examples: list[dict[str, Any]], top_k: int
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Score trustworthy traces only; record every excluded case explicitly.

    Input examples: id, source_id, citation_text, retrieval_trace. Returned
    metrics contain IDs and scores only, never retrieved document contents.
    """
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for example in examples:
        gold = example.get("source_id")
        trace = example.get("retrieval_trace")
        reason: str | None = None
        if not isinstance(gold, str) or not gold.strip():
            reason = "missing_gold_document_id"
        elif not isinstance(trace, dict):
            reason = "trace_unavailable"
        elif trace.get("status") != "captured":
            reason = str(trace.get("status") or "trace_unavailable")
        else:
            retrieved = trace.get("retrieved")
            if not isinstance(retrieved, list) or not all(
                isinstance(chunk, dict)
                and (chunk.get("doc_id") is None or isinstance(chunk.get("doc_id"), str))
                and isinstance(chunk.get("text"), str)
                for chunk in retrieved
            ):
                reason = "invalid_retrieval_chunks"
            else:
                metrics = compute_retrieval_metrics(
                    retrieved,
                    gold,
                    example.get("citation_text") or "",
                    top_k=top_k,
                )
                rows.append({"id": str(example.get("id") or ""), "metrics": metrics})
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1

    coverage = {
        "total_examples": len(examples),
        "scored_examples": len(rows),
        "skipped_examples": len(examples) - len(rows),
        "skipped_reasons": dict(sorted(skipped.items())),
        "note": "Only one observed custom retrieval call with a source_id is scored; absent traces do not receive zero scores.",
    }
    if not rows:
        return None, coverage
    report = {
        "num_examples": len(rows),
        "top_k": top_k,
        "averages": {
            key: fmean(row["metrics"][key] for row in rows) for key in METRIC_KEYS
        },
        "rows": rows,
    }
    return report, coverage
