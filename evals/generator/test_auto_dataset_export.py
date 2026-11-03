"""Regression tests for lossless auto-evaluation JSONL exports."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.datasets.schema import EvalItem, EvalRow, write_jsonl
from evals.generator import auto_pipeline


def _row(question):
    return EvalRow(item=EvalItem(
        question=question, correct_answer="alpha", citation_text="alpha source",
    ))


def test_atomic_exclusive_jsonl_write_preserves_existing_data(tmp_path):
    path = tmp_path / "already-present.jsonl"
    write_jsonl([_row("first")], path)
    initial = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_jsonl([_row("second")], path, exclusive=True)
    assert path.read_bytes() == initial


def test_auto_filename_collision_uses_new_file_without_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(auto_pipeline.time, "strftime", lambda _: "20261103_034000")
    rows = [_row("first evaluation")]
    first = auto_pipeline._write_auto_dataset(rows, output_dir=tmp_path)
    previous = first.read_bytes()
    second = auto_pipeline._write_auto_dataset([_row("new evaluation")], output_dir=tmp_path)
    third = auto_pipeline._write_auto_dataset([_row("third evaluation")], output_dir=tmp_path)

    assert [p.name for p in (first, second, third)] == [
        "auto_20261103_034000.jsonl",
        "auto_20261103_034000_1.jsonl",
        "auto_20261103_034000_2.jsonl",
    ]
    assert first.read_bytes() == previous
    assert "first evaluation" in first.read_text(encoding="utf-8")
    assert "new evaluation" in second.read_text(encoding="utf-8")
    assert "third evaluation" in third.read_text(encoding="utf-8")
    for path in (first, second, third):
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(entries) == 1
        assert entries[0]["item"]["correct_answer"] == "alpha"


def test_autogenerate_pipeline_uses_collision_safe_export(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(auto_pipeline.time, "strftime", lambda _: "20261103_034000")
    monkeypatch.setattr(
        auto_pipeline, "iterate_chunks",
        lambda cfg, max_docs: iter([{"doc_id": "doc-1", "text": "Alpha citation"}]),
    )
    monkeypatch.setattr(
        auto_pipeline, "make_context_candidates",
        lambda chunks, **kwargs: list(chunks),
    )
    monkeypatch.setattr(
        auto_pipeline, "generate_many_from_context",
        lambda cfg, context, n_questions: [{
            "question": "What is the reference?", "correct_answer": "Alpha",
            "citation_text": "Alpha citation", "source_id": "doc-1",
            "difficulty": "easy", "tags": ["knowledge"],
        }],
    )
    cfg = SimpleNamespace(evals=SimpleNamespace(auto=SimpleNamespace(
        items_target=1, scan_docs=1, context_chars_min=5,
        context_chars_max=100, per_doc_context_cap=1,
        questions_per_context=1, qgen_workers=1, quality_regen_limit=0,
        mix={"easy": 1.0, "medium": 0.0, "hard": 0.0},
        per_tag_cap=5, per_source_cap=5,
    )))

    first_summary, first_rows = auto_pipeline.auto_generate_dataset(cfg)
    second_summary, second_rows = auto_pipeline.auto_generate_dataset(cfg)
    assert first_summary.sampled == second_summary.sampled == 1
    assert first_rows == second_rows
    assert Path(first_summary.dataset_path).exists()
    assert Path(second_summary.dataset_path).exists()
    assert first_summary.dataset_path != second_summary.dataset_path
    assert len(list((tmp_path / "evals" / "datasets").glob("auto_*.jsonl"))) == 2
