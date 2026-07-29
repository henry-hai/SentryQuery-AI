"""Offline tests for the MCP interface layer: cache, rate limiter, tool shape.

IMPORTANT SCOPE NOTE, in the same spirit as test_critic_plumbing.py. Every test
here stubs run_pipeline, so nothing in this file makes an OpenAI, Pinecone, or
Tavily call and none of it needs a key. That is the point: these cover what CI
can verify deterministically, which is the MCP wiring and the two cost controls.

What these tests do NOT cover is the Critic's groundedness judgment or any live
model behaviour. Whether an ungrounded answer really is ruled REVISE is a live
gpt-4o-mini call, and stubbing it would only assert a canned verdict back to
itself. That check stays in evals/eval.py, which needs real keys. The verdict
assertions below are about faithful passthrough of the verdict field, not about
whether the verdict was correct.
"""
import pytest

import mcp_server


class StubResult:
    """Stand-in for a PipelineResult, carrying the fields the tool reads."""

    def __init__(self, answer="Net sales were 254.5 billion dollars."):
        self.answer = answer
        self.sources = ["costco-10k.pdf p.5", "costco-10k.pdf p.6"]
        self.tool_used = "docs"
        self.confidence = 0.92
        self.verdict = "APPROVE"
        self.critic_reason = "Every claim is supported by the sources."
        self.revisions = 0


class SpyPipeline:
    """Records every call, so a test can assert a paid run did not happen."""

    def __init__(self, result=None):
        self.calls: list[str] = []
        self._result = result or StubResult()

    def __call__(self, system, query):
        self.calls.append(query)
        return self._result


@pytest.fixture
def spy(monkeypatch):
    """Isolate each test: a fresh cache and limiter, and a stubbed pipeline.

    get_system is stubbed too, so the graph is never built and no OpenAI or
    Pinecone client is ever constructed.
    """
    pipeline = SpyPipeline()
    monkeypatch.setattr(mcp_server, "run_pipeline", pipeline)
    monkeypatch.setattr(mcp_server, "get_system", lambda: object())
    monkeypatch.setattr(mcp_server, "cache", mcp_server.AnswerCache(max_entries=8))
    monkeypatch.setattr(mcp_server, "limiter", mcp_server.RateLimiter(max_per_minute=100))
    return pipeline


# -----------------------------------------------------------------------------
# (a) Cache
# -----------------------------------------------------------------------------
def test_repeated_question_is_served_from_cache_without_a_second_paid_run(spy):
    first = mcp_server.answer_question("What were total net sales?")
    second = mcp_server.answer_question("What were total net sales?")

    assert len(spy.calls) == 1, "the repeat must not re-embed or re-query Pinecone"
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["sources"] == first["sources"]


def test_cache_key_normalizes_case_and_whitespace(spy):
    mcp_server.answer_question("  What Were   TOTAL net sales? ")
    hit = mcp_server.answer_question("what were total net sales?")

    assert len(spy.calls) == 1, "differing case and spacing must not cost a second run"
    assert hit["cached"] is True


def test_normalize_question_trims_lowercases_and_collapses_whitespace():
    assert mcp_server.normalize_question("  What\tWere   TOTAL\nsales? ") == (
        "what were total sales?"
    )


def test_cache_evicts_the_oldest_entry_at_the_cap():
    cache = mcp_server.AnswerCache(max_entries=2)
    for key in ("q1", "q2"):
        cache.put(key, {"answer": key})
    cache.get("q1")  # a read must not refresh position: entries age by insertion
    cache.put("q3", {"answer": "q3"})

    assert len(cache) == 2
    assert cache.get("q1") is None, "the oldest entry should have been evicted"
    assert cache.get("q2") is not None
    assert cache.get("q3") is not None


def test_cached_values_are_copies_so_a_caller_cannot_mutate_the_cache(spy):
    first = mcp_server.answer_question("What were total net sales?")
    first["sources"].append("injected.pdf p.1")

    second = mcp_server.answer_question("What were total net sales?")
    assert "injected.pdf p.1" not in second["sources"]


# -----------------------------------------------------------------------------
# (b) Rate limiter
# -----------------------------------------------------------------------------
def test_over_limit_call_is_rejected_without_invoking_the_pipeline(spy, monkeypatch):
    monkeypatch.setattr(mcp_server, "limiter", mcp_server.RateLimiter(max_per_minute=2))

    mcp_server.answer_question("first question")
    mcp_server.answer_question("second question")
    assert len(spy.calls) == 2

    with pytest.raises(mcp_server.RateLimitExceeded) as excinfo:
        mcp_server.answer_question("third question")

    assert len(spy.calls) == 2, "the rejected call must not fire a paid run"
    assert "No paid model or Pinecone call was made" in str(excinfo.value)


def test_cache_hits_do_not_spend_rate_limit_budget(spy, monkeypatch):
    monkeypatch.setattr(mcp_server, "limiter", mcp_server.RateLimiter(max_per_minute=1))

    mcp_server.answer_question("the same question")
    # Only one paid call was allowed, but repeats are free and must still serve.
    for _ in range(5):
        assert mcp_server.answer_question("the same question")["cached"] is True

    assert len(spy.calls) == 1


def test_limiter_is_per_client_and_window_slides():
    limiter = mcp_server.RateLimiter(max_per_minute=1)

    limiter.check("client-a", now=1000.0)
    with pytest.raises(mcp_server.RateLimitExceeded):
        limiter.check("client-a", now=1010.0)

    # A different client has its own budget.
    limiter.check("client-b", now=1010.0)

    # Once the first call falls out of the 60-second window, client-a is allowed
    # again without any manual reset.
    limiter.check("client-a", now=1061.0)


# -----------------------------------------------------------------------------
# (c) Tool response shape
# -----------------------------------------------------------------------------
def test_ask_corpus_returns_the_answer_schema_fields_plus_the_critic_verdict(spy):
    response = mcp_server.ask_corpus("What were total net sales?")

    # The AnswerSchema fields.
    assert response["answer"] == "Net sales were 254.5 billion dollars."
    assert response["sources"] == ["costco-10k.pdf p.5", "costco-10k.pdf p.6"]
    assert response["tool_used"] == "docs"
    assert response["confidence"] == 0.92

    # The Critic's groundedness result, so a client is not left with bare text.
    # This asserts faithful passthrough of the verdict, not that it is correct.
    assert response["verdict"] == "APPROVE"
    assert response["critic_reason"] == "Every claim is supported by the sources."
    assert response["revisions"] == 0

    assert set(response) == {
        "answer",
        "sources",
        "tool_used",
        "confidence",
        "verdict",
        "critic_reason",
        "revisions",
        "cached",
    }


def test_ask_corpus_passes_a_revise_verdict_through_unchanged(spy, monkeypatch):
    revised = StubResult(answer="A narrower, better supported answer.")
    revised.verdict = "REVISE"
    revised.critic_reason = "The growth projection is not in the sources."
    revised.revisions = 1
    monkeypatch.setattr(mcp_server, "run_pipeline", SpyPipeline(revised))

    response = mcp_server.ask_corpus("What will next year's sales be?")
    assert response["verdict"] == "REVISE"
    assert response["critic_reason"] == "The growth projection is not in the sources."
    assert response["revisions"] == 1


def test_blank_question_is_rejected_before_any_paid_call(spy):
    for blank in ("", "   ", "\n\t"):
        with pytest.raises(ValueError):
            mcp_server.answer_question(blank)
    assert spy.calls == []


# -----------------------------------------------------------------------------
# MCP registration
# -----------------------------------------------------------------------------
def test_ask_corpus_is_registered_as_an_mcp_tool_taking_one_question_argument():
    """The injected context parameter must stay out of the public tool schema."""
    import asyncio

    tools = asyncio.run(mcp_server.mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}

    assert "ask_corpus" in by_name
    schema = by_name["ask_corpus"].input_schema
    assert schema["required"] == ["question"]
    assert set(schema["properties"]) == {"question"}
