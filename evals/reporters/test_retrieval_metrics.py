"""No-service tests for offline retrieval metrics and report files."""
from __future__ import annotations

import json

import pytest

from evals.reporters.retrieval_metrics import (
    evaluate_retrieval_jsonl,
    main,
    write_retrieval_reports,
)


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def test_offline_metrics_preserve_ranks_and_do_not_reward_duplicates(tmp_path):
    source = tmp_path / "results.jsonl"
    write_jsonl(source, [
        {
            "id": "query|1\n<script>x</script>",
            "gold_doc_id": "gold",
            "citation_text": "answer",
            "retrieved": [
                {"doc_id": None, "text": "unknown"},
                {"doc_id": "gold", "text": "answer"},
                {"doc_id": "gold", "text": "duplicate"},
            ],
        },
        {
            "id": "query-2",
            "gold_doc_id": "not-found",
            "retrieved": [{"doc_id": "irrelevant", "text": ""}],
        },
    ])
    report = evaluate_retrieval_jsonl(source, top_k=3)
    assert report["num_examples"] == 2
    assert report["top_k"] == 3
    assert report["rows"][0]["metrics"]["precision@k"] == pytest.approx(1 / 3)
    assert report["rows"][0]["metrics"]["MRR"] == 0.5
    assert report["rows"][1]["metrics"]["hit@k"] == 0
    assert report["averages"]["precision@k"] == pytest.approx(1 / 6)
    assert report["averages"]["MRR"] == 0.25

    json_path, markdown_path = write_retrieval_reports(report, tmp_path / "reports")
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["averages"] == report["averages"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "precision@k" in markdown
    assert "MRR" in markdown
    assert "query&#124;1" in markdown
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert len([line for line in markdown.splitlines() if "query&#124;1" in line]) == 1


@pytest.mark.parametrize("record,error", [
    ({"gold_doc_id": "gold"}, "retrieved"),
    ({"retrieved": [], "gold_doc_id": ""}, "gold_doc_id"),
    ({"retrieved": [{"doc_id": 42}], "gold_doc_id": "gold"}, "retrieved"),
    ({"retrieved": [], "gold_doc_id": "gold", "citation_text": None}, "citation_text"),
    ([], "expected JSON object"),
])
def test_invalid_records_fail_loudly_with_source_line(tmp_path, record, error):
    source = tmp_path / "invalid.jsonl"
    write_jsonl(source, [record])
    with pytest.raises(ValueError, match="Line 1.*" + error):
        evaluate_retrieval_jsonl(source)


def test_empty_and_malformed_jsonl_are_rejected(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no evaluation records"):
        evaluate_retrieval_jsonl(source)
    source.write_text('{"retrieved": [}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON on line 1"):
        evaluate_retrieval_jsonl(source)


@pytest.mark.parametrize("k", [0, -1, True, 1.2, "3"])
def test_invalid_top_k_is_rejected_even_for_empty_file(tmp_path, k):
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_retrieval_jsonl(source, top_k=k)


def test_cli_writes_offline_reports_without_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [
        {"id": "demo", "gold_doc_id": "doc-1",
         "retrieved": [{"doc_id": "doc-1", "text": "citation"}]}
    ])
    dest = tmp_path / "outputs"
    assert main([str(source), "--top-k", "1", "--output-dir", str(dest)]) == 0
    assert (dest / "retrieval_metrics.json").exists()
    assert (dest / "retrieval_metrics.md").exists()
    assert "Retrieval report:" in capsys.readouterr().out
