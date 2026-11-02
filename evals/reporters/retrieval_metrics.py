"""Evaluate saved retrieval results without models, credentials or network calls.

Input JSONL: one object per line with id, gold_doc_id, retrieved (ranked
list of {doc_id, text}), and optional citation_text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from evals.metrics.retrieval import compute_retrieval_metrics
from evals.reporters.md import _safe_markdown


METRIC_KEYS = (
    "hit@k", "recall@k", "precision@k", "MRR",
    "context_precision", "support_coverage",
)


def evaluate_retrieval_jsonl(source: str | Path, top_k: int = 8) -> dict[str, Any]:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    rows: list[dict[str, Any]] = []
    with Path(source).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"Line {line_number}: expected JSON object")
            retrieved = entry.get("retrieved")
            gold = entry.get("gold_doc_id")
            citation = entry.get("citation_text", "")
            if not isinstance(retrieved, list) or not all(
                isinstance(doc, dict)
                and (doc.get("doc_id") is None or isinstance(doc.get("doc_id"), str))
                and (doc.get("text") is None or isinstance(doc.get("text"), str))
                for doc in retrieved
            ):
                raise ValueError(f"Line {line_number}: retrieved must be a list of document objects")
            if not isinstance(gold, str) or not gold.strip():
                raise ValueError(f"Line {line_number}: nonempty gold_doc_id is required")
            if not isinstance(citation, str):
                raise ValueError(f"Line {line_number}: citation_text must be a string")
            metrics = compute_retrieval_metrics(retrieved, gold, citation, top_k=top_k)
            row_id = entry.get("id", str(line_number))
            rows.append({"id": str(row_id), "metrics": metrics})
    if not rows:
        raise ValueError("Retrieval input has no evaluation records")
    return {
        "num_examples": len(rows),
        "top_k": top_k,
        "averages": {
            key: fmean(row["metrics"][key] for row in rows)
            for key in METRIC_KEYS
        },
        "rows": rows,
    }


def write_retrieval_reports(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    json_path = dest / "retrieval_metrics.json"
    md_path = dest / "retrieval_metrics.md"

    md_lines = [
        "# Offline RAG Retrieval Evaluation",
        "",
        f"Examples: {report['num_examples']}  ",
        f"Top-k: {report['top_k']}",
        "",
        "## Average metrics",
        "",
        "| Metric | Average |",
        "|---|---:|",
    ]
    for key in METRIC_KEYS:
        md_lines.append(f"| {key} | {report['averages'][key]:.3f} |")
    md_lines.extend([
        "",
        "## Individual examples",
        "",
        "| ID | " + " | ".join(METRIC_KEYS) + " |",
        "|---|" + "---:|" * len(METRIC_KEYS),
    ])
    for row in report["rows"]:
        values = " | ".join(f"{row['metrics'][key]:.3f}" for key in METRIC_KEYS)
        md_lines.append(f"| {_safe_markdown(row['id'])} | {values} |")

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score saved ranked retrieval results without API calls."
    )
    parser.add_argument("input", type=Path, help="JSONL with ranked retrieved documents and gold_doc_id")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))
    args = parser.parse_args(argv)
    try:
        report = evaluate_retrieval_jsonl(args.input, top_k=args.top_k)
        json_path, md_path = write_retrieval_reports(report, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Retrieval report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
