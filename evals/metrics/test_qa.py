"""Offline QA metric regression tests; token frequency must affect F1."""
import pytest
from evals.metrics.qa import em_f1, exact_match, f1_score


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("red red red blue", "red blue", 2 / 3),
        ("red blue", "red red red blue", 2 / 3),
        ("red red", "red red red", 0.8),
        ("Alpha BETA", "alpha beta", 1.0),
        ("red blue", "yellow green", 0.0),
        ("", "red", 0.0),
        ("", "", 0.0),
    ],
)
def test_f1_uses_token_occurrence_counts(pred, gold, expected):
    assert f1_score(pred, gold) == pytest.approx(expected)


def test_exact_match_stays_independent_of_token_f1():
    assert exact_match("red red blue", "red blue") == 0.0
    assert em_f1("red red blue", "red blue") == (0.0, pytest.approx(0.8))
