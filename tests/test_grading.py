"""Robust answer grading tests (PRD §7.3, §14) — Wave 2."""

from __future__ import annotations

from mem_eval.grading.judge import ExactJudge, LLMJudge, NormalizingJudge, normalize


def test_normalize_strips_punctuation_and_case():
    assert normalize("  Berlin! ") == "berlin"
    assert normalize("New-York, NY") == "new york ny"


def test_normalizing_judge_containment_and_case():
    j = NormalizingJudge()
    assert j.judge("The answer is Berlin.", "berlin")
    assert j.judge("user123@example.com", "USER123@example.com")
    assert not j.judge("I don't know.", "Berlin")
    assert not j.judge(None, "Berlin")


def test_normalizing_judge_numeric_tolerance():
    j = NormalizingJudge(numeric_abs_tol=0.5)
    assert j.judge("the count is 42", "42")
    assert j.judge("42.4", "42")          # within abs tol
    assert not j.judge("99", "42")


def test_normalizing_judge_rejects_substring_false_positives():
    """Whole-token containment, NOT raw substring: a labs-facing grader must not
    mark these correct (regression guard for the Wave 2 reviewer findings)."""
    j = NormalizingJudge()
    assert not j.judge("I have 4200 items", "42")
    assert not j.judge("the catalog has 142 entries", "42")
    assert not j.judge("classify the records", "class")
    assert not j.judge("she is an engineer", "engine")
    assert not j.judge("blueberry pie", "blue")


def test_normalizing_judge_ambiguous_multimonth_not_date_matched():
    """A date answer mentioning multiple distinct months is ambiguous; the date
    canonicalizer must not force-match the first one."""
    from mem_eval.grading.judge import _canon_date

    assert _canon_date("changed from August to July") is None


def test_normalizing_judge_dates():
    j = NormalizingJudge()
    assert j.judge("It was March.", "march")
    assert j.judge("2025-03-19", "2025-3-19")
    assert j.judge("March 2025", "2025 march")
    assert not j.judge("April", "March")


def test_normalizing_judge_aliases():
    j = NormalizingJudge()
    assert j.judge("I live in NYC now", "New York City", aliases=("NYC",))
    assert not j.judge("I live in NYC now", "New York City")  # no alias -> miss


def test_exact_judge_is_strict():
    j = ExactJudge()
    assert j.judge("Berlin", "berlin")
    assert not j.judge("the answer is Berlin", "berlin")  # containment NOT enough


def test_llm_judge_offline_path_is_deterministic_bool():
    j = LLMJudge()  # offline stub
    assert j.model == "offline-stub"
    v1 = j.judge("the answer is Berlin", "Berlin")
    v2 = j.judge("the answer is Berlin", "Berlin")
    assert v1 is True and v2 is True
    assert j.judge("nope", "Berlin") is False


def test_llm_judge_accepts_real_complete_fn():
    calls = []

    def fake_model(prompt: str) -> str:
        calls.append(prompt)
        return "YES"

    j = LLMJudge(complete_fn=fake_model, model="claude-test")
    assert j.judge("anything", "Berlin") is True
    assert calls and "Berlin" in calls[0]  # gold is passed to the model
    assert j.model == "claude-test"
