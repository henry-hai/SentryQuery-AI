"""Offline tests for the Critic wiring in graph.critique.

IMPORTANT SCOPE NOTE. These do NOT test the Critic's groundedness *judgment*.
That judgment is the gpt-4o-mini call, so verifying that an ungrounded answer is
actually ruled REVISE genuinely requires a live model and cannot run in CI
without a key. Mocking the model would only assert the canned verdict back to
itself, which proves nothing. That semantic check stays in evals/eval.py, which
needs live keys.

What these tests DO cover, with a stub model, is the plumbing that CI can verify
deterministically: the Critic is handed the exact retrieved chunks as evidence,
each chunk is labelled with its source, the answer under review is included, and
the structured verdict is returned faithfully to the caller.
"""
from langchain_core.documents import Document

from graph import critique
from schema import CriticVerdict


class StubCritic:
    """Stand-in for the structured-output LLM: records the prompt it was given
    and returns a preset verdict, so we can inspect the evidence and the passthrough."""

    def __init__(self, verdict: CriticVerdict):
        self._verdict = verdict
        self.last_prompt: str | None = None

    def invoke(self, prompt: str) -> CriticVerdict:
        self.last_prompt = prompt
        return self._verdict


CHUNKS = [
    Document(
        page_content="Total net sales were 254.5 billion dollars.",
        metadata={"source": "costco-10k.pdf", "page": 4.0},
    ),
    Document(
        page_content="Membership fees were 4.8 billion dollars.",
        metadata={"source": "costco-10k.pdf", "page": 5.0},
    ),
]


def test_critique_feeds_every_chunk_and_the_answer_as_evidence():
    stub = StubCritic(CriticVerdict(verdict="APPROVE", reason="ok"))
    answer = "Net sales were 254.5 billion dollars."
    critique(stub, answer, CHUNKS)

    prompt = stub.last_prompt
    assert answer in prompt
    for chunk in CHUNKS:
        assert chunk.page_content in prompt
    # Each chunk is attributed with its 1-based source label.
    assert "costco-10k.pdf p.5" in prompt
    assert "costco-10k.pdf p.6" in prompt


def test_critique_returns_the_models_verdict_unchanged():
    revise = CriticVerdict(verdict="REVISE", reason="claim X is unsupported")
    stub = StubCritic(revise)
    out = critique(stub, "The CEO is a professional boxer.", CHUNKS)
    assert out.verdict == "REVISE"
    assert out.reason == "claim X is unsupported"
