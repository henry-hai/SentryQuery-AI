"""Offline HTTP tests for the question endpoint."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import api
import agent
import graph


DOC = Document(
    page_content="Net income increased 10% to $8,099 million.",
    metadata={"source": "/docs/costco-10k-2025.pdf", "page": 29, "score": 0.71},
)


@pytest.fixture
def env(monkeypatch):
    calls = []
    result = SimpleNamespace(
        answer="Net income rose 10% [1].",
        verdict="APPROVE",
        critic_reason="Every sentence is supported.",
        tool_used="docs",
        retrieved=[DOC, DOC],
        web_sources=["https://example.com/report"],
    )

    def pipeline(system, question):
        calls.append(question)
        return result

    monkeypatch.setattr(api, "run_pipeline", pipeline)
    monkeypatch.setattr(api, "get_system", lambda: object())
    monkeypatch.setattr(api, "cache", api.AnswerCache(max_entries=8))
    monkeypatch.setattr(api, "limiter", api.RateLimiter(max_per_minute=100))
    return TestClient(api.app), calls, result


def test_ask_response_has_document_and_separate_web_sources(env):
    client, calls, _ = env
    response = client.post("/ask", json={"question": "  How did income change?  "})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "question", "answer", "verdict", "critic_reason", "tool_used",
        "sources", "web_sources", "cached", "answered_at",
    }
    assert body["question"] == "How did income change?"
    assert body["answer"] == "Net income rose 10% [1]."
    assert body["verdict"] == "APPROVE"
    assert body["critic_reason"] == "Every sentence is supported."
    assert body["tool_used"] == "docs"
    assert body["sources"] == [{
        "n": 1,
        "document": "costco-10k-2025.pdf",
        "page": 30,
        "score": 0.71,
        "passage": "Net income increased 10% to $8,099 million.",
    }]
    assert body["web_sources"] == ["https://example.com/report"]
    assert body["cached"] is False
    assert body["answered_at"].endswith("Z")
    assert calls == ["How did income change?"]


def test_ask_sources_number_the_score_sorted_passages(env):
    client, _, result = env
    low = Document(
        page_content="Lower-scoring passage.",
        metadata={"source": "/docs/costco-10k-2025.pdf", "page": 1, "score": 0.3},
    )
    unscored = Document(
        page_content="Unscored passage.",
        metadata={"source": "/docs/costco-10k-2025.pdf", "page": 2},
    )
    high = Document(
        page_content="Higher-scoring passage.",
        metadata={"source": "/docs/costco-10k-2025.pdf", "page": 3, "score": 0.8},
    )
    result.retrieved = [low, unscored, high, low]

    sources = client.post("/ask", json={"question": "What changed?"}).json()["sources"]

    assert [(source["n"], source["passage"], source["score"]) for source in sources] == [
        (1, "Higher-scoring passage.", 0.8),
        (2, "Lower-scoring passage.", 0.3),
        (3, "Unscored passage.", None),
    ]


def test_ask_revise_and_web_only_result(env):
    client, _, result = env
    result.verdict = "REVISE"
    result.critic_reason = "The increase is not supported."
    result.retrieved = []
    result.tool_used = "web"
    body = client.post("/ask", json={"question": "What changed?"}).json()
    assert body["verdict"] == "REVISE"
    assert body["critic_reason"] == "The increase is not supported."
    assert body["sources"] == []
    assert body["web_sources"] == ["https://example.com/report"]


def test_ask_without_web_results_returns_empty_list(env):
    client, _, result = env
    result.web_sources = []
    body = client.post("/ask", json={"question": "What changed?"}).json()
    assert body["web_sources"] == []


@pytest.mark.parametrize("question", ["", "   ", "x" * (api.MAX_QUESTION_CHARS + 1)])
def test_invalid_question_is_400_before_pipeline(env, question):
    client, calls, _ = env
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_question"
    assert calls == []


def test_ask_uses_cache_without_another_pipeline_run(env):
    client, calls, _ = env
    first = client.post("/ask", json={"question": "What changed?"}).json()
    second = client.post("/ask", json={"question": "  WHAT changed?  "}).json()
    assert len(calls) == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answered_at"] == first["answered_at"]


def test_ask_and_verify_cache_keys_do_not_collide(env, monkeypatch):
    client, calls, _ = env
    monkeypatch.setattr(api, "critique", lambda critic, claim, chunks: SimpleNamespace(
        verdict="APPROVE", reason="Supported."
    ))
    monkeypatch.setattr(api, "get_system", lambda: SimpleNamespace(critic_llm=object()))
    assert client.post("/ask", json={"question": "Same text"}).status_code == 200
    verified = client.post("/verify", json={"claim": "Same text"})
    assert verified.status_code == 200
    assert "web_sources" not in verified.json()
    assert all("https://example.com" not in str(e) for e in verified.json()["evidence"])
    assert len(calls) == 2


def test_ask_rate_limit_is_per_caller_and_returns_429(env, monkeypatch):
    client, calls, _ = env
    monkeypatch.setattr(api, "limiter", api.RateLimiter(max_per_minute=1))
    first = {"X-Forwarded-For": "203.0.113.7"}
    other = {"X-Forwarded-For": "198.51.100.4"}
    assert client.post("/ask", json={"question": "One?"}, headers=first).status_code == 200
    blocked = client.post("/ask", json={"question": "Two?"}, headers=first)
    assert blocked.status_code == 429
    assert blocked.json()["error"] == "rate_limited"
    assert "Retry-After" in blocked.headers
    assert client.post("/ask", json={"question": "Three?"}, headers=other).status_code == 200
    assert calls == ["One?", "Three?"]


def test_root_page_has_both_tabs(env):
    client, _, _ = env
    page = client.get("/")
    assert page.status_code == 200
    assert 'role="tablist"' in page.text
    assert 'id="p-ask"' in page.text
    assert 'id="p-check"' in page.text
    assert 'id="curl"' in page.text


def test_web_urls_reach_query_and_pipeline_results(monkeypatch):
    class FakeAgent:
        def invoke(self, payload):
            handle.web_sources.extend(["https://example.com/one", "https://example.com/one"])
            return {"messages": [SimpleNamespace(content="Web answer", tool_calls=[])]}

    class FakeSynth:
        def invoke(self, prompt):
            raise RuntimeError("use the fallback")

    handle = agent.AgentHandle(FakeAgent(), object(), FakeSynth(), [], [], [])
    query_result = agent.run_query(handle, "What is new?")
    assert query_result.web_sources == ["https://example.com/one"]

    class FakeGraph:
        def invoke(self, state, config):
            return {
                "answer": query_result.answer,
                "verdict": "APPROVE",
                "web_sources": query_result.web_sources,
            }

    monkeypatch.setattr(graph, "log_run", lambda query, result: None)
    system = graph.System(FakeGraph(), handle, object())
    pipeline_result = graph.run_pipeline(system, "What is new?")
    assert pipeline_result.web_sources == ["https://example.com/one"]
