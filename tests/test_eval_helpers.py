"""Unit tests for the grading helpers in evals/eval.py.

These exercise the harness scoring logic (schema validity and the per-case
grade) against hand-built result objects, with no agent, Pinecone, or OpenAI
call. The full harness in eval.py still needs live keys and is not run in CI.
"""
from dataclasses import dataclass, field

import eval as evalmod


@dataclass
class FakeResult:
    """Stand-in for a PipelineResult, holding only the fields grading reads."""

    answer: str = ""
    sources: list = field(default_factory=list)
    tool_used: str = "docs"
    confidence: float | None = 0.9
    retriever_calls: int = 1
    verdict: str = "APPROVE"


def test_validates_as_schema_true_for_complete_result():
    assert evalmod.validates_as_schema(FakeResult(answer="net sales were 254 billion"))


def test_validates_as_schema_false_when_confidence_is_none():
    # A fallback result carries confidence=None, which must not validate.
    assert not evalmod.validates_as_schema(FakeResult(confidence=None))


def test_validates_as_schema_false_when_confidence_out_of_range():
    assert not evalmod.validates_as_schema(FakeResult(confidence=1.5))


def test_grade_passes_a_well_formed_grounded_case():
    case = {
        "expected_keywords_any": ["net sales", "$"],
        "must_use_retriever": True,
        "expect_verdict": "APPROVE",
    }
    result = FakeResult(answer="Total net sales were 254 billion.", retriever_calls=2)
    ok, reasons = evalmod.grade(case, result)
    assert ok, reasons


def test_grade_flags_missing_keyword():
    case = {"expected_keywords_any": ["operating revenue"]}
    result = FakeResult(answer="The company sells groceries.")
    ok, reasons = evalmod.grade(case, result)
    assert not ok
    assert any("missing" in r for r in reasons)


def test_grade_flags_retriever_use_mismatch():
    case = {"expected_keywords_any": ["price"], "must_use_retriever": False}
    result = FakeResult(answer="The price is $50.", retriever_calls=3)
    ok, reasons = evalmod.grade(case, result)
    assert not ok
    assert any("retriever use mismatch" in r for r in reasons)


def test_grade_flags_verdict_mismatch():
    case = {"expected_keywords_any": ["net sales"], "expect_verdict": "APPROVE"}
    result = FakeResult(answer="net sales rose", verdict="REVISE")
    ok, reasons = evalmod.grade(case, result)
    assert not ok
    assert any("verdict mismatch" in r for r in reasons)
