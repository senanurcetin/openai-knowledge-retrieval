"""Offline tests for curated RAG JSONL schema and record integrity."""
from __future__ import annotations

import json

import pytest

from evals.datasets.schema import EvalItem, EvalRow, load_and_validate_jsonl, write_jsonl


def save_records(path, records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def sample(**fields):
    return {
        "id": "eval-1",
        "question": "What is a useful response?",
        "correct_answer": "A substantive answer",
        "citation_text": "A substantive answer appears here",
        **fields,
    }


def test_flat_and_wrapped_jsonl_records_are_equivalent(tmp_path):
    source = tmp_path / "mixed.jsonl"
    first = sample(id="flat")
    second = sample(id="wrapped", source_id="source-1", tags=["relevant"])
    save_records(source, [first, {"item": second}])
    result = load_and_validate_jsonl(source)
    assert [row.item.id for row in result] == ["flat", "wrapped"]
    assert result[0].item.question == result[1].item.question
    assert result[1].item.source_id == "source-1"
    assert result[1].item.tags == ["relevant"]


def test_exported_rows_roundtrip_and_optional_missing_ids_are_supported(tmp_path):
    source = tmp_path / "no-ids.jsonl"
    save_records(source, [
        {"item": sample(id=None)},
        {"item": sample(id=None, question="A different question")},
    ])
    original = load_and_validate_jsonl(source)
    assert [r.item.id for r in original] == [None, None]
    written = tmp_path / "roundtrip.jsonl"
    write_jsonl(original, written)
    assert load_and_validate_jsonl(written) == original


@pytest.mark.parametrize("flat", [True, False])
def test_duplicate_ids_rejected_with_both_source_line_numbers(tmp_path, flat):
    path = tmp_path / "duplicates.jsonl"
    first = sample(id="reused")
    second = sample(id="reused", question="Other unique question")
    if not flat:
        first, second = {"item": first}, {"item": second}
    path.write_text(
        json.dumps(first) + "\n\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Duplicate evaluation id.*reused.*line 3.*line 1"):
        load_and_validate_jsonl(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("question", ""),
        ("question", " \t\n "),
        ("correct_answer", ""),
        ("correct_answer", " \r\n "),
        ("id", " \t "),
    ],
)
def test_whitespace_only_required_fields_and_provided_id_are_rejected(tmp_path, key, value):
    path = tmp_path / "blank.jsonl"
    save_records(path, [{"item": sample(**{key: value})}])
    with pytest.raises(ValueError, match=r"Schema error on line 1"):
        load_and_validate_jsonl(path)


def test_ambiguous_flat_and_wrapped_keys_cannot_be_silently_ignored(tmp_path):
    source = tmp_path / "ambiguous.jsonl"
    save_records(source, [{"item": sample(), "question": "Different top-level question"}])
    with pytest.raises(ValueError, match=r"Mixed flat and wrapped evaluation fields on line 1"):
        load_and_validate_jsonl(source)


@pytest.mark.parametrize("record", [[], None, "string"])
def test_invalid_nonobject_rows_still_rejected_with_line_number(tmp_path, record):
    path = tmp_path / "bad.jsonl"
    save_records(path, [record])
    with pytest.raises(ValueError, match=r"Invalid JSON object on line 1"):
        load_and_validate_jsonl(path)


def test_empty_citation_is_allowed_when_question_and_answer_exist(tmp_path):
    path = tmp_path / "no-support.jsonl"
    save_records(path, [{"item": sample(citation_text="")}])
    rows = load_and_validate_jsonl(path)
    assert rows[0].item.citation_text == ""


def test_direct_model_validation_matches_file_validation():
    with pytest.raises(ValueError):
        EvalItem(**sample(question="  "))
    assert EvalRow(item=EvalItem(**sample(id=None))).item.id is None
