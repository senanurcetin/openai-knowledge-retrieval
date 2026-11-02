"""Offline tests: report artifacts share one ID per evaluation invocation."""
from __future__ import annotations

import json
import re
from pathlib import Path

from evals.reporters.md import write_markdown_report
from evals.reporters.html import write_html_report
from evals.reporters.retrieval_metrics import (
    METRIC_KEYS,
    evaluate_retrieval_jsonl,
    write_retrieval_reports,
    write_unavailable_retrieval_reports,
)


def test_qa_and_retrieval_writers_expose_the_same_explicit_id(tmp_path):
    run_id = "7fd388d70f124fdca851920c7402961a"
    qa = {
        "project": "example", "run_id": run_id, "num_examples": 1,
        "avg_latency_ms": 5, "avg_em": 1, "avg_f1": 1,
        "rows": [{"id": "x", "latency_ms": 5, "em": 1, "f1": 1, "answer": "ok"}],
    }
    md_path = Path(write_markdown_report(qa, out_dir=str(tmp_path)))
    html_path = tmp_path / "qa.html"
    write_html_report({"project": "example", "run_id": run_id}, html_path)
    assert f"Run ID: \`{run_id}\`" in md_path.read_text(encoding="utf-8")
    assert f"Run ID: {run_id}" in html_path.read_text(encoding="utf-8")

    metrics = {
        "run_id": run_id, "num_examples": 1, "top_k": 2,
        "averages": {metric: 0.0 for metric in METRIC_KEYS},
        "rows": [{"id": "sample", "metrics": {
            metric: 0.0 for metric in METRIC_KEYS
        }}],
    }
    json_path, retrieval_md = write_retrieval_reports(metrics, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == run_id
    assert f"Run ID: \`{run_id}\`" in retrieval_md.read_text(encoding="utf-8")

    json_path, retrieval_md = write_unavailable_retrieval_reports(
        tmp_path, top_k=2, reason="no_scored_examples", run_id=run_id,
    )
    assert json.loads(json_path.read_text(encoding="utf-8"))["run_id"] == run_id
    assert f"Run ID: \`{run_id}\`" in retrieval_md.read_text(encoding="utf-8")


def test_offline_evaluations_generate_different_identifiers(tmp_path):
    source = tmp_path / "hits.jsonl"
    source.write_text(
        json.dumps({"gold_doc_id": "gold", "retrieved": [
            {"doc_id": "gold", "text": "passage"},
        ]}) + "\n", encoding="utf-8",
    )
    first = evaluate_retrieval_jsonl(source, top_k=1)
    second = evaluate_retrieval_jsonl(source, top_k=1)
    assert re.fullmatch(r"[0-9a-f]{32}", first["run_id"])
    assert re.fullmatch(r"[0-9a-f]{32}", second["run_id"])
    assert first["run_id"] != second["run_id"]
    assert first["averages"] == second["averages"]


def test_run_id_is_escaped_in_rendered_reports(tmp_path):
    injected = "<script>x</script>|new\nline"
    report = {
        "project": "test", "run_id": injected, "num_examples": 0,
        "avg_latency_ms": 0, "avg_em": 0, "avg_f1": 0, "rows": [],
    }
    markdown = Path(write_markdown_report(report, out_dir=str(tmp_path))).read_text("utf-8")
    html_path = tmp_path / "qa.html"
    write_html_report({"run_id": injected}, html_path)
    html = html_path.read_text("utf-8")
    assert "<script>" not in markdown + html
    assert "&lt;script&gt;" in markdown + html
    assert "&#124;" in markdown
