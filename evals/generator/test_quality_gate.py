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



def test_answerability_normalizes_units_in_both_answer_and_citation():
    cfg = SimpleNamespace(
        evals=SimpleNamespace(auto=SimpleNamespace(quality_regen_limit=0))
    )
    rows = [
        {"question": "How long is the warranty?", "correct_answer": "24 months",
         "citation_text": "Warranty coverage is 24 months."},
        {"question": "How much current does it draw?", "correct_answer": "4.5 amperes",
         "citation_text": "Operating current is 4.5 A."},
        {"question": "How many years?", "correct_answer": "2 yr",
         "citation_text": "Maintenance interval is 2 years."},
        {"question": "What is the wrong duration?", "correct_answer": "24 months",
         "citation_text": "Warranty coverage is 12 months."},
    ]
    filtered = _quality_gate(rows, cfg)
    assert [row["question"] for row in filtered] == [
        "How long is the warranty?",
        "How much current does it draw?",
        "How many years?",
    ]
    for row in filtered:
        EvalItem(**row)


def test_answerability_requires_complete_numeric_or_word_token():
    cfg = SimpleNamespace(
        evals=SimpleNamespace(auto=SimpleNamespace(quality_regen_limit=0))
    )
    rows = [
        {"question": "Does it last exactly 24 units?", "correct_answer": "24",
         "citation_text": "The measured duration is 124 units."},
        {"question": "What is the exact duration?", "correct_answer": "24",
         "citation_text": "The measured duration is 24 units."},
        {"question": "Is the tool a cat?", "correct_answer": "cat",
         "citation_text": "This is a catalog of tools."},
        {"question": "Which animal is named?", "correct_answer": "cat",
         "citation_text": "The listed animal is a cat."},
    ]
    filtered = _quality_gate(rows, cfg)
    assert [row["question"] for row in filtered] == [
        "What is the exact duration?", "Which animal is named?",
    ]
