"""Offline tests for scored, company-filtered retrieval.

The index is a fake that holds a handful of chunks with preset similarity
scores per query and honors the same `source` metadata filter Pinecone does.
Nothing here embeds text or queries Pinecone, so no key is needed.

What these cover: chunks below the threshold are dropped, at most RETRIEVAL_MAX_K
survive, zero survivors becomes a no_evidence FAIL through /verify without the
Critic being called, and a claim naming Costco can never pull a Deere chunk. What
they do not cover is whether the threshold value itself is right for the live
index. That came from measured scores and is recorded in the README.
"""
from langchain_core.documents import Document

import agent
import api
from companies import COMPANY_SOURCES, companies_named, source_filter
from config import RETRIEVAL_MAX_K, RETRIEVAL_MIN_SCORE

COSTCO = "docs/costco-10k-2025.pdf"
DEERE = "docs/deere-10k-2025.pdf"
SOUTHWEST = "docs/southwest-10k-2025.pdf"


class FakeIndex:
    """Stands in for PineconeVectorStore.similarity_search_with_score.

    Every query sees the same chunks at the same scores. The filter is applied
    the way Pinecone applies {"source": {"$in": [...]}}, before top-k.
    """

    def __init__(self, rows):
        self.rows = rows  # list of (source, page, text, score)
        self.calls: list[dict] = []

    def similarity_search_with_score(self, query, k=4, filter=None):
        self.calls.append({"query": query, "k": k, "filter": filter})
        allowed = None
        if filter is not None:
            allowed = set(filter["source"]["$in"])
        hits = [
            (Document(page_content=text, metadata={"source": src, "page": page}), score)
            for src, page, text, score in self.rows
            if allowed is None or src in allowed
        ]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits[:k]


MIXED = FakeIndex(
    [
        (DEERE, 45.0, "Deere net income was $5,027 million.", 0.62),
        (COSTCO, 29.0, "Net income increased 10% to $8,099.", 0.58),
        (DEERE, 46.0, "Deere net sales and revenues were $45,684 million.", 0.57),
        (COSTCO, 44.0, "NET INCOME $ 8,099 $ 7,367 $ 6,292", 0.55),
        (SOUTHWEST, 40.0, "Operating revenues were $28 billion.", 0.53),
        (COSTCO, 0.0, "Costco Wholesale Corporation annual report cover page.", 0.44),
        (DEERE, 1.0, "Deere annual report cover page.", 0.31),
    ]
)


# -----------------------------------------------------------------------------
# The threshold
# -----------------------------------------------------------------------------
def test_chunks_below_the_threshold_are_dropped():
    index = FakeIndex([(COSTCO, 1.0, "close", 0.71), (COSTCO, 2.0, "far", RETRIEVAL_MIN_SCORE - 0.01)])
    docs = agent.scored_search(index, "anything")
    assert [d.page_content for d in docs] == ["close"]


def test_a_chunk_exactly_at_the_threshold_is_kept():
    index = FakeIndex([(COSTCO, 1.0, "edge", RETRIEVAL_MIN_SCORE)])
    assert len(agent.scored_search(index, "anything")) == 1


def test_fetches_eight_and_keeps_at_most_five():
    index = FakeIndex([(COSTCO, float(i), f"chunk {i}", 0.9 - i * 0.01) for i in range(10)])
    docs = agent.scored_search(index, "anything")
    assert index.calls[0]["k"] == 8
    assert len(docs) == RETRIEVAL_MAX_K == 5
    assert [d.page_content for d in docs] == [f"chunk {i}" for i in range(5)]


def test_each_kept_chunk_carries_its_score_closest_first():
    docs = agent.scored_search(MIXED, "net income")
    scores = [agent.similarity_score(d) for d in docs]
    assert scores == sorted(scores, reverse=True)
    assert all(s >= RETRIEVAL_MIN_SCORE for s in scores)


def test_zero_survivors_is_an_empty_list_not_the_nearest_noise():
    index = FakeIndex([(COSTCO, 1.0, "weather unrelated", 0.21), (DEERE, 2.0, "also unrelated", 0.12)])
    assert agent.scored_search(index, "What's the weather in Paris?") == []


# -----------------------------------------------------------------------------
# The company filter
# -----------------------------------------------------------------------------
def test_mapping_paths_match_the_source_metadata_format():
    # Chunks in the live index carry source paths like "docs/costco-10k-2025.pdf".
    for entry in COMPANY_SOURCES.values():
        for path in entry["sources"]:
            assert path.startswith("docs/") and path.endswith(".pdf")


def test_companies_named_matches_whole_words_case_insensitively():
    assert companies_named("Costco's annual income grew about 10%") == ["costco"]
    assert companies_named("JOHN DEERE and Southwest Airlines") == ["southwest", "deere"]
    assert companies_named("Apple's iPhone revenue grew 5%") == []
    assert companies_named("costcoish") == []


def test_source_filter_is_none_when_no_company_is_named():
    assert source_filter([]) is None
    assert source_filter(["apple"]) is None


def test_a_costco_claim_never_retrieves_a_deere_chunk():
    docs = agent.retrieve(MIXED, "Costco's annual income grew about 10%")
    assert docs, "the Costco chunks above the threshold should come back"
    assert {d.metadata["source"] for d in docs} == {COSTCO}


def test_the_claims_company_wins_over_a_rewritten_tool_query():
    # The Researcher may search a generic phrase, or even name another company.
    # The scope taken from the claim itself still pins retrieval to Costco.
    for query in ("net income growth", "Deere net income"):
        docs = agent.retrieve(MIXED, query, scope=["costco"])
        assert {d.metadata["source"] for d in docs} == {COSTCO}


def test_no_company_named_searches_the_whole_index():
    docs = agent.retrieve(MIXED, "net income growth")
    assert MIXED.calls[-1]["filter"] is None
    assert {d.metadata["source"] for d in docs} == {COSTCO, DEERE, SOUTHWEST}


def test_run_query_sets_the_scope_from_the_question(monkeypatch):
    seen: list[list[str]] = []

    class FakeAgent:
        def invoke(self, payload):
            # The tool reads handle.scope during the agent loop.
            seen.append(list(handle.scope))
            return {"messages": [type("M", (), {"content": "done", "tool_calls": None})()]}

    class FailingSynth:
        def invoke(self, prompt):
            raise RuntimeError("force the fallback path, no model needed")

    handle = agent.AgentHandle(FakeAgent(), MIXED, FailingSynth(), [], [], ["stale"])
    agent.run_query(handle, "Is it true that Costco's net income grew 10%?")
    assert seen == [["costco"]]


# -----------------------------------------------------------------------------
# Through /verify
# -----------------------------------------------------------------------------
class _StubSystem:
    critic_llm = object()


class _Result:
    def __init__(self, retrieved, tool_used):
        self.retrieved = retrieved
        self.tool_used = tool_used


def _wire(monkeypatch, index, critic_calls):
    """Stub the pipeline so it retrieves from the fake index through the real
    scored, filtered path, and stub the Critic so any call is recorded."""

    def pipeline(system, query):
        docs = agent.retrieve(index, query, companies_named(query))
        return _Result(docs, "docs" if docs else "web")

    def critic(critic_llm, claim, chunks):
        critic_calls.append(chunks)
        from schema import CriticVerdict

        return CriticVerdict(verdict="APPROVE", reason="supported")

    monkeypatch.setattr(api, "run_pipeline", pipeline)
    monkeypatch.setattr(api, "critique", critic)
    monkeypatch.setattr(api, "get_system", _StubSystem)
    monkeypatch.setattr(api, "cache", api.AnswerCache(max_entries=8))
    monkeypatch.setattr(api, "limiter", api.RateLimiter(max_per_minute=100))


def test_zero_survivors_gives_no_evidence_and_never_calls_the_critic(monkeypatch):
    critic_calls: list = []
    index = FakeIndex([(SOUTHWEST, 3.0, "Fleet of Boeing 737 aircraft.", 0.45)])
    _wire(monkeypatch, index, critic_calls)

    response = api.verify_claim("Apple's iPhone revenue grew 5% in fiscal 2025.")

    assert response["verdict"] == "FAIL"
    assert response["reason_code"] == "no_evidence"
    assert response["evidence"] == []
    assert critic_calls == []


def test_verify_evidence_carries_scores_and_only_the_named_company(monkeypatch):
    critic_calls: list = []
    _wire(monkeypatch, MIXED, critic_calls)

    response = api.verify_claim("Costco's annual income grew about 10%.")

    assert response["verdict"] == "PASS"
    assert [e["document"] for e in response["evidence"]] == ["costco-10k-2025.pdf"] * 2
    assert [e["score"] for e in response["evidence"]] == [0.58, 0.55]
    assert all(d.metadata["source"] == COSTCO for d in critic_calls[0])
