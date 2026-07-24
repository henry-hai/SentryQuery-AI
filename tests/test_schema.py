"""Contract tests for the AnswerSchema and CriticVerdict Pydantic models.

Pure validation, no API calls. These lock the structured-output contract that
both the UI and the Critic depend on.
"""
import pytest
from pydantic import ValidationError

from schema import AnswerSchema, CriticVerdict


def test_answer_schema_accepts_valid_payload():
    a = AnswerSchema(
        answer="Net sales were about 254 billion dollars.",
        sources=["costco-10k.pdf p.5"],
        tool_used="docs",
        confidence=0.9,
    )
    assert a.tool_used == "docs"
    assert a.confidence == 0.9


def test_answer_schema_defaults_sources_to_empty_list():
    a = AnswerSchema(answer="x", tool_used="none", confidence=0.5)
    assert a.sources == []


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 2.0])
def test_answer_schema_rejects_confidence_out_of_range(bad_confidence):
    with pytest.raises(ValidationError):
        AnswerSchema(answer="x", tool_used="docs", confidence=bad_confidence)


def test_answer_schema_rejects_unknown_tool_used():
    with pytest.raises(ValidationError):
        AnswerSchema(answer="x", tool_used="database", confidence=0.5)


@pytest.mark.parametrize("verdict", ["APPROVE", "REVISE"])
def test_critic_verdict_accepts_valid_verdicts(verdict):
    v = CriticVerdict(verdict=verdict, reason="because")
    assert v.verdict == verdict


def test_critic_verdict_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        CriticVerdict(verdict="MAYBE", reason="because")
