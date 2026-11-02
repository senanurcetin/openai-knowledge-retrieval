"""Quality-gate regression: invalid generated QA rows never break schema."""
from __future__ import annotations

from types import SimpleNamespace

from evals.datasets.schema import EvalItem
from evals.generator.auto_pipeline import _quality_gate


def test_auto_generation_gate_drops_blank_question_and_answer_before_export():
    cfg = SimpleNamespace(
        evals=SimpleNamespace(
            auto=SimpleNamespace(quality_regen_limit=0),
        ),
    )
    rows = [
        {
            "question": "A valid first question",
            "correct_answer": "Alpha",
            "citation_text": "Alpha is found in this citation",
        },
        {
            "question": " \n\t ",
            "correct_answer": "Alpha",
            "citation_text": "Alpha is found in this citation",
        },
        {
            "question": "A question with no answer",
            "correct_answer": "  ",
            "citation_text": "No useful reference answer exists",
        },
        {
            "question": "A question with a missing answer",
            "correct_answer": "",
            "citation_text": "No useful reference answer exists",
        },
        {
            "question": "A valid second question",
            "correct_answer": "Beta",
            "citation_text": "Beta is found in this citation",
        },
    ]

    filtered = _quality_gate(rows, cfg)
    assert len(filtered) == 2
    assert [item["question"] for item in filtered] == [
        "A valid first question",
        "A valid second question",
    ]
    for item in filtered:
        EvalItem(**item)
