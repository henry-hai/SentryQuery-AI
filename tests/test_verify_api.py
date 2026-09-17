"""Offline tests for the HTTP claim-verification layer.

IMPORTANT SCOPE NOTE, the same one that governs test_critic_plumbing.py and
test_mcp_server.py. Every test here stubs run_pipeline and critique, so nothing
in this file makes an OpenAI, Pinecone, or Tavily call and none of it needs a
key.

What these DO cover is what CI can verify deterministically: that APPROVE maps
to PASS and REVISE to FAIL, that the Critic is handed the claim itself and the
exact retrieved chunks, that reason_code is derived from graph state, that the
evidence in the response is the chunks that were ruled against rather than
anything re-queried, that the cache and the rate limiter sit in front of every
paid call, and the HTTP status codes.

What they do NOT cover is whether the Critic's verdict is CORRECT. That is a
live gpt-4o-mini judgment. Stubbing it would only assert a canned verdict back
to itself. It stays in evals/eval.py, which needs real keys, and in the recorded
catch in the README, which was produced by a real run.
"""
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import api
from schema import CriticVerdict

CHUNKS = [
    Document(
        page_content="Total net sales were $269,706 million for fiscal 2025.",
        metadata={"source": "/docs/costco-10k-2025.pdf", "page": 33.0},
    ),
    Document(
        page_content="Membership fees were $5,200 million.",
        metadata={"source": "/docs/costco-10k-2025.pdf", "page": 34.0},
    ),
]


class StubResult:
    """Stand-in for a PipelineResult, carrying only the fields the API reads."""

    def __init__(self, retrieved=None, tool_used="docs"):
        self.retrieved = CHUNKS if retrieved is None else retrieved
        self.tool_used = tool_used


class SpyPipeline:
    """Records every call, so a test can assert a paid run did not happen."""

    def __init__(self, result=None):
        self.calls: list[str] = []
        self._result = result if result is not None else StubResult()

    def __call__(self, system, query):
        self.calls.append(query)
        return self._result


class StubSystem:
    """Stand-in for the compiled System. Only critic_llm is ever read here, and
    the stubbed Critic ignores it."""

    critic_llm = object()


class SpyCritic:
    """Records the claim and chunks it was handed, returns a preset verdict."""

    def __init__(self, verdict="APPROVE", reason="Every part is supported."):
        self._verdict = CriticVerdict(verdict=verdict, reason=reason)
        self.calls: list[tuple] = []

    def __call__(self, critic_llm, answer, chunks):
        self.calls.append((answer, chunks))
        return self._verdict


@pytest.fixture
def env(monkeypatch):
    """Isolate each test: fresh cache and limiter, stubbed pipeline and Critic.

    get_system is stubbed too, so the graph is never built and no OpenAI or
    Pinecone client is ever constructed.
    """
    pipeline = SpyPipeline()
    critic = SpyCritic()
    monkeypatch.setattr(api, "run_pipeline", pipeline)
    monkeypatch.setattr(api, "critique", critic)
    monkeypatch.setattr(api, "get_system", StubSystem)
    monkeypatch.setattr(api, "cache", api.AnswerCache(max_entries=8))
    monkeypatch.setattr(api, "limiter", api.RateLimiter(max_per_minute=100))
    return pipeline, critic


@pytest.fixture
def client():
    return TestClient(api.app)


# -----------------------------------------------------------------------------
# (a) Verdict mapping
# -----------------------------------------------------------------------------
def test_approve_maps_to_pass_with_no_reason_code(env):
    response = api.verify_claim("Net sales were $269,706 million.")
    assert response["verdict"] == "PASS"
    assert response["reason_code"] is None


def test_revise_maps_to_fail_with_critic_rejected(env, monkeypatch):
    monkeypatch.setattr(
        api, "critique", SpyCritic("REVISE", "The $500 billion figure is not in the sources.")
    )
    response = api.verify_claim("Net sales were $500 billion.")
    assert response["verdict"] == "FAIL"
    assert response["reason_code"] == "critic_rejected"
    # The Critic's own words, not a rewrite of them.
    assert response["reason"] == "The $500 billion figure is not in the sources."


def test_no_retrieved_chunks_is_no_evidence_and_never_calls_the_critic(env, monkeypatch):
    _, critic = env
    monkeypatch.setattr(api, "run_pipeline", SpyPipeline(StubResult(retrieved=[])))
    response = api.verify_claim("The moon is made of cheese.")
    assert response["verdict"] == "FAIL"
    assert response["reason_code"] == "no_evidence"
    assert response["evidence"] == []
    assert critic.calls == [], "there is nothing to rule against, so do not pay for a ruling"


def test_tool_used_none_is_also_no_evidence(env, monkeypatch):
    monkeypatch.setattr(
        api, "run_pipeline", SpyPipeline(StubResult(tool_used="none"))
    )
    response = api.verify_claim("Write me a sorting function.")
    assert response["verdict"] == "FAIL"
    assert response["reason_code"] == "no_evidence"


def test_reason_code_has_only_the_two_documented_values(env, monkeypatch):
    seen = {api.verify_claim("a claim")["reason_code"]}
    monkeypatch.setattr(api, "critique", SpyCritic("REVISE", "unsupported"))
    seen.add(api.verify_claim("another claim")["reason_code"])
    monkeypatch.setattr(api, "run_pipeline", SpyPipeline(StubResult(retrieved=[])))
    seen.add(api.verify_claim("a third claim")["reason_code"])
    assert seen == {None, "critic_rejected", "no_evidence"}


# -----------------------------------------------------------------------------
# (b) The Critic gets the claim and the exact chunks
# -----------------------------------------------------------------------------
def test_the_critic_rules_on_the_claim_itself_not_the_researchers_answer(env):
    _, critic = env
    claim = "Costco reported total net sales of $500 billion in fiscal 2025."
    api.verify_claim(claim)

    assert len(critic.calls) == 1
    judged, chunks = critic.calls[0]
    assert judged == claim, "the Critic must rule on the claim, not on a drafted answer"
    assert chunks is CHUNKS, "and against the exact chunks the Researcher retrieved"


def test_the_researcher_is_asked_for_evidence_and_is_given_the_claim(env):
    pipeline, _ = env
    api.verify_claim("Net sales were $269,706 million.")
    assert len(pipeline.calls) == 1
    assert "Net sales were $269,706 million." in pipeline.calls[0]


# -----------------------------------------------------------------------------
# (c) Evidence
# -----------------------------------------------------------------------------
def test_evidence_is_the_exact_chunks_with_document_and_1_based_page(env):
    evidence = api.verify_claim("a claim")["evidence"]
    assert evidence[0] == {
        "document": "costco-10k-2025.pdf",
        "page": 34,
        "passage": "Total net sales were $269,706 million for fiscal 2025.",
    }
    assert [e["page"] for e in evidence] == [34, 35]


def test_evidence_drops_repeats_from_multiple_retriever_calls(env, monkeypatch):
    monkeypatch.setattr(
        api, "run_pipeline", SpyPipeline(StubResult(retrieved=CHUNKS + CHUNKS))
    )
    assert len(api.verify_claim("a claim")["evidence"]) == 2


# -----------------------------------------------------------------------------
# (d) Cost controls, in front of every paid call
# -----------------------------------------------------------------------------
def test_repeated_claim_is_served_from_cache_without_a_second_paid_run(env):
    pipeline, critic = env
    first = api.verify_claim("Net sales were $269,706 million.")
    second = api.verify_claim("Net sales were $269,706 million.")

    assert len(pipeline.calls) == 1
    assert len(critic.calls) == 1, "a cache hit must not re-run the Critic either"
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["verdict"] == first["verdict"]
    assert second["checked_at"] == first["checked_at"], "a cache hit reports the original check time"


def test_cache_key_normalizes_case_and_whitespace(env):
    pipeline, _ = env
    api.verify_claim("  Net Sales   Were $269,706 MILLION. ")
    hit = api.verify_claim("net sales were $269,706 million.")
    assert len(pipeline.calls) == 1
    assert hit["cached"] is True


def test_cache_hits_do_not_spend_rate_limit_budget(env, monkeypatch):
    pipeline, _ = env
    monkeypatch.setattr(api, "limiter", api.RateLimiter(max_per_minute=1))
    api.verify_claim("the same claim")
    for _ in range(5):
        assert api.verify_claim("the same claim")["cached"] is True
    assert len(pipeline.calls) == 1


def test_over_limit_claim_is_rejected_without_invoking_the_pipeline(env, monkeypatch):
    pipeline, _ = env
    monkeypatch.setattr(api, "limiter", api.RateLimiter(max_per_minute=2))

    api.verify_claim("first claim")
    api.verify_claim("second claim")
    with pytest.raises(api.RateLimitExceeded):
        api.verify_claim("third claim")

    assert len(pipeline.calls) == 2, "the rejected call must not fire a paid run"


def test_blank_and_oversized_claims_are_rejected_before_any_paid_call(env):
    pipeline, _ = env
    for bad in ("", "   ", "\n\t", "x" * (api.MAX_CLAIM_CHARS + 1)):
        with pytest.raises(ValueError):
            api.verify_claim(bad)
    assert pipeline.calls == []


# -----------------------------------------------------------------------------
# (e) HTTP surface
# -----------------------------------------------------------------------------
def test_verify_returns_the_documented_response_shape(env, client):
    body = client.post("/verify", json={"claim": "Net sales were $269,706 million."}).json()
    assert set(body) == {
        "claim",
        "verdict",
        "reason_code",
        "reason",
        "evidence",
        "cached",
        "checked_at",
    }
    assert set(body["evidence"][0]) == {"document", "page", "passage"}


def test_blank_claim_is_a_400(env, client):
    response = client.post("/verify", json={"claim": "   "})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_claim"


def test_over_limit_is_a_429_with_a_retry_after_header(env, client, monkeypatch):
    monkeypatch.setattr(api, "limiter", api.RateLimiter(max_per_minute=1))
    client.post("/verify", json={"claim": "first claim"})
    response = client.post("/verify", json={"claim": "second claim"})

    assert response.status_code == 429
    assert response.json()["error"] == "rate_limited"
    assert response.json()["retry_after_seconds"] >= 1
    assert "Retry-After" in response.headers


def test_an_upstream_failure_is_a_502_and_leaks_no_upstream_detail(env, client, monkeypatch):
    def boom(system, query):
        raise RuntimeError("openai key sk-secret-value rejected")

    monkeypatch.setattr(api, "run_pipeline", boom)
    response = client.post("/verify", json={"claim": "a claim"})

    assert response.status_code == 502
    assert response.json()["error"] == "upstream_failure"
    assert "sk-secret" not in response.text


def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    # The claim box, not the wording around it, is what makes this the app.
    assert 'id="claim"' in response.text
    assert 'id="check"' in response.text


def test_health_check_answers_without_building_the_graph(client):
    for path in ("/healthz", "/health"):
        assert client.get(path).json() == {"status": "ok"}
    assert api._system is None, "the health check must not construct a paid client"


# -----------------------------------------------------------------------------
# (f) The rate-limit key
# -----------------------------------------------------------------------------
def test_forwarded_for_first_hop_is_the_rate_limit_key(env, client, monkeypatch):
    monkeypatch.setattr(api, "limiter", api.RateLimiter(max_per_minute=1))
    headers = {"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}

    assert client.post("/verify", json={"claim": "one"}, headers=headers).status_code == 200
    assert client.post("/verify", json={"claim": "two"}, headers=headers).status_code == 429
    # A different caller has its own budget.
    other = {"X-Forwarded-For": "198.51.100.4"}
    assert client.post("/verify", json={"claim": "three"}, headers=other).status_code == 200
