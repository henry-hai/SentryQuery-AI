"""Offline tests for scored, company-filtered hybrid retrieval.

The index is a fake that holds a handful of chunks with preset similarity
scores per query and honors the same `source` metadata filter Pinecone does.
Fake embeddings and an in-memory index keep these tests offline.

What these cover: the best cosine gates the whole fused result, low-cosine
chunks remain when the gate passes, fused rank determines order, and an empty
result becomes a no_evidence FAIL without calling the Critic. A Costco claim
cannot pull a Deere chunk. The live threshold calibration is in the README.
"""
import gzip
import json

import pytest
from langchain_core.documents import Document

import agent
import api
import chunks
from companies import COMPANY_SOURCES, companies_named, source_filter, strip_company_names
from config import RETRIEVAL_MAX_K, RETRIEVAL_MIN_SCORE
from keyword_search import KeywordIndex, load_index, tokenize

COSTCO = "docs/costco-10k-2025.pdf"
DEERE = "docs/deere-10k-2025.pdf"
SOUTHWEST = "docs/southwest-10k-2025.pdf"


@pytest.fixture(autouse=True)
def no_keyword_hits(monkeypatch):
    """Keep existing vector tests independent of the committed chunk file."""
    monkeypatch.setattr(agent, "keyword_search", lambda query, sources, k: [])
    scores = {text: score for _, _, text, score in MIXED.rows}
    scores["Fleet of Boeing 737 aircraft."] = 0.45

    class BaselineEmbeddings:
        def embed_query(self, text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[scores[text], (1 - scores[text] ** 2) ** 0.5] for text in texts]

    monkeypatch.setattr(agent, "embeddings", BaselineEmbeddings())


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
    assert docs, "the Costco candidate clears the query-level gate"
    assert {d.metadata["source"] for d in docs} == {COSTCO}


def test_the_claims_company_wins_over_a_rewritten_tool_query():
    # The Researcher may search a generic phrase, or even name another company.
    # The scope taken from the claim itself still pins retrieval to Costco.
    for query in ("net income growth", "Deere net income"):
        docs = agent.retrieve(MIXED, query, scope=["costco"])
        assert {d.metadata["source"] for d in docs} == {COSTCO}


def test_multi_company_scope_uses_scored_search_without_keyword_search(monkeypatch):
    def keyword(query, sources, k):
        raise AssertionError("multi-company retrieval must skip keyword search")

    monkeypatch.setattr(agent, "keyword_search", keyword)
    index = FakeIndex(MIXED.rows)
    scope = ["deere", "costco", "southwest"]
    query = "Costco total revenue 2025"
    docs = agent.retrieve(index, query, scope=scope)
    expected = agent.scored_search(FakeIndex(MIXED.rows), query, source_filter(scope))

    assert {doc.metadata["source"] for doc in docs} == {DEERE, COSTCO, SOUTHWEST}
    assert index.calls == [{
        "query": query,
        "k": 8,
        "filter": {"source": {"$in": [DEERE, COSTCO, SOUTHWEST]}},
    }]
    assert [(doc.page_content, doc.metadata["score"]) for doc in docs] == [
        (doc.page_content, doc.metadata["score"]) for doc in expected
    ]
    assert all("rrf" not in doc.metadata for doc in docs)


def test_one_company_scope_strips_for_auxiliary_searches(monkeypatch):
    keyword_calls = []

    def keyword(query, sources, k):
        keyword_calls.append((query, sources, k))
        return []

    monkeypatch.setattr(agent, "keyword_search", keyword)
    index = FakeIndex(MIXED.rows)
    query = "Deere total revenues 2025"
    docs = agent.retrieve(index, query, scope=["deere"])

    assert {doc.metadata["source"] for doc in docs} == {DEERE}
    assert [call["query"] for call in index.calls] == [query, "total revenues 2025"]
    assert all(call["filter"] == {"source": {"$in": [DEERE]}} for call in index.calls)
    assert all(call["k"] == 8 for call in index.calls)
    assert keyword_calls == [("total revenues 2025", [DEERE], 8)]


def test_two_company_tool_query_keeps_names_and_scope_order(monkeypatch):
    def keyword(query, sources, k):
        raise AssertionError("multi-company retrieval must skip keyword search")

    monkeypatch.setattr(agent, "keyword_search", keyword)
    index = FakeIndex(MIXED.rows)
    query = "Costco and Deere total revenue 2025"
    docs = agent.retrieve(index, query, scope=["deere", "costco"])

    assert {doc.metadata["source"] for doc in docs} == {COSTCO, DEERE}
    assert len(index.calls) == 1
    assert index.calls[-1]["query"] == query
    assert index.calls[-1]["filter"] == {"source": {"$in": [DEERE, COSTCO]}}
    assert all("rrf" not in doc.metadata for doc in docs)


def test_unchanged_single_company_query_runs_one_vector_search():
    index = FakeIndex([])
    agent.retrieve(index, "Costco's and the")

    assert [call["query"] for call in index.calls] == ["Costco's and the"]


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
    assert [e["document"] for e in response["evidence"]] == ["costco-10k-2025.pdf"] * 3
    assert [e["score"] for e in response["evidence"]] == [0.58, 0.55, 0.44]
    assert all(d.metadata["source"] == COSTCO for d in critic_calls[0])


# -----------------------------------------------------------------------------
# Local chunks and hybrid ranking
# -----------------------------------------------------------------------------
def _doc(source, page, text):
    return Document(page_content=text, metadata={"source": source, "page": page})


class FakeEmbeddings:
    def __init__(self, vectors):
        self.vectors = vectors
        self.queries = []
        self.batches = []

    def embed_query(self, text):
        self.queries.append(text)
        return self.vectors[text]

    def embed_documents(self, texts):
        self.batches.append(texts)
        return [self.vectors[text] for text in texts]


def test_chunk_export_keeps_source_page_and_text(tmp_path, monkeypatch):
    monkeypatch.setattr(chunks, "load_chunks", lambda: [_doc(COSTCO, 29, "Net income 8,099")])
    path = tmp_path / "chunks.jsonl.gz"
    assert chunks.write_chunks(path) == 1
    with gzip.open(path, "rt", encoding="utf-8") as source:
        assert json.loads(source.readline()) == {
            "source": COSTCO, "page": 29, "text": "Net income 8,099"
        }
    first_bytes = path.read_bytes()
    assert chunks.write_chunks(path) == 1
    assert path.read_bytes() == first_bytes


def test_bm25_ranks_exact_figure_first_and_filters_company():
    index = KeywordIndex([
        _doc(COSTCO, 29, "Net income increased 10% to $8,099 million."),
        _doc(COSTCO, 30, "Net income increased 4% to $7,367 million."),
        _doc(DEERE, 10, "Net income increased 10% to $8,099 million."),
    ])
    assert tokenize("$8,099 and 10%") == ["8,099", "and", "10%"]
    hits = index.search("net income 10% 8,099", [COSTCO], 3)
    assert [hit.metadata["page"] for hit in hits] == [29, 30]
    assert all(hit.metadata["source"] == COSTCO for hit in hits)


def test_missing_chunk_file_warns_once_and_returns_no_index(tmp_path, capsys):
    path = tmp_path / "missing.jsonl.gz"
    load_index.cache_clear()
    assert load_index(path) is None
    assert load_index(path) is None
    assert capsys.readouterr().err.count("Warning: keyword chunk file missing") == 1
    load_index.cache_clear()


def test_company_name_stripping_handles_possessives_multiword_names_and_fallback():
    assert strip_company_names("Costco's annual income grew 10%", ["costco"]) == "annual income grew 10%"
    assert strip_company_names("SOUTHWEST AIRLINES' revenue", ["southwest"]) == "revenue"
    assert strip_company_names("Southwest Airlines revenue", ["southwest"]) == "revenue"
    assert strip_company_names("Costco's and the", ["costco"]) == "Costco's and the"
    assert strip_company_names("costcoish revenue", ["costco"]) == "costcoish revenue"


def test_rrf_merges_shared_chunk_above_single_side_hits(monkeypatch):
    shared = "Net income increased 10% to $8,099."
    index = FakeIndex([
        (COSTCO, 1, "Vector only", 0.99),
        (COSTCO, 29, shared, 0.98),
    ])
    keyword_calls = []

    def keyword(query, sources, k):
        keyword_calls.append((query, sources, k))
        return [
            _doc(COSTCO, 29, "Net  income increased 10% to $8,099."),
            _doc(COSTCO, 2, "Keyword only"),
        ]

    monkeypatch.setattr(agent, "keyword_search", keyword)
    embeddings = FakeEmbeddings({
        "Costco income 10%": [1, 0],
        "Vector only": [0.8, 0.6],
        shared: [0.7, (1 - 0.7**2) ** 0.5],
        "Keyword only": [0.2, (1 - 0.2**2) ** 0.5],
    })
    monkeypatch.setattr(agent, "embeddings", embeddings)

    docs = agent.retrieve(index, "Costco income 10%")

    assert [doc.metadata["page"] for doc in docs] == [29, 1, 2]
    assert docs[0].metadata["rrf"] > docs[1].metadata["rrf"]
    assert [doc.metadata["score"] for doc in docs] == [0.7, 0.8, 0.2]
    assert [call["query"] for call in index.calls] == [
        "Costco income 10%", "income 10%"
    ]
    assert keyword_calls == [("income 10%", [COSTCO], 8)]
    assert embeddings.queries == ["Costco income 10%"]
    assert embeddings.batches == [[shared, "Vector only", "Keyword only"]]


def test_rrf_fuses_original_stripped_and_keyword_lists(monkeypatch):
    query = "Deere total net sales and revenues 2025"
    stripped = "total net sales and revenues 2025"
    original_only = _doc(DEERE, 48, "Income statement")
    shared = _doc(DEERE, 46, "Selected financial data")
    stripped_only = _doc(DEERE, 47, "Segment table")
    keyword_only = _doc(DEERE, 49, "Revenue note")

    class QueryIndex:
        def __init__(self):
            self.calls = []

        def similarity_search_with_score(self, text, k, filter):
            self.calls.append((text, k, filter))
            if text == query:
                return [(original_only, 0.72), (shared, 0.71)]
            assert text == stripped
            return [(stripped_only, 0.75), (shared, 0.70)]

    keyword_calls = []

    def keyword(text, sources, k):
        keyword_calls.append((text, sources, k))
        return [shared, keyword_only]

    monkeypatch.setattr(agent, "keyword_search", keyword)
    embeddings = FakeEmbeddings({
        query: [1, 0],
        original_only.page_content: [1, 0],
        shared.page_content: [1, 0],
        stripped_only.page_content: [1, 0],
        keyword_only.page_content: [1, 0],
    })
    monkeypatch.setattr(agent, "embeddings", embeddings)
    index = QueryIndex()

    docs = agent.retrieve(index, query, scope=["deere"])

    search_filter = {"source": {"$in": [DEERE]}}
    assert index.calls == [(query, 8, search_filter), (stripped, 8, search_filter)]
    assert keyword_calls == [(stripped, [DEERE], 8)]
    assert [doc.metadata["page"] for doc in docs] == [46, 48, 47, 49]
    assert docs[0].metadata["rrf"] == pytest.approx(2 / 62 + 1 / 61)
    assert len(embeddings.queries) == 1
    assert embeddings.batches == [[
        shared.page_content, original_only.page_content,
        stripped_only.page_content, keyword_only.page_content,
    ]]


def test_best_cosine_opens_gate_for_low_cosine_chunk_and_keeps_rrf(monkeypatch):
    monkeypatch.setattr(agent, "keyword_search", lambda query, sources, k: [
        _doc(COSTCO, 1, "close"), _doc(COSTCO, 2, "far")
    ])
    embeddings = FakeEmbeddings({
        "Costco income": [1, 0], "close": [0.8, 0.6], "far": [0, 1]
    })
    monkeypatch.setattr(agent, "embeddings", embeddings)

    docs = agent.retrieve(FakeIndex([]), "Costco income")

    assert [doc.page_content for doc in docs] == ["close", "far"]
    assert [doc.metadata["score"] for doc in docs] == [0.8, 0.0]
    assert [doc.metadata["rrf"] for doc in docs] == pytest.approx([1 / 61, 1 / 62])
    assert embeddings.queries == ["Costco income"]
    assert embeddings.batches == [["close", "far"]]


def test_best_cosine_below_gate_drops_all_candidates(monkeypatch):
    index = FakeIndex([(COSTCO, 1, "vector hit", 0.99)])
    monkeypatch.setattr(agent, "keyword_search", lambda query, sources, k: [
        _doc(COSTCO, 2, "keyword hit")
    ])
    embeddings = FakeEmbeddings({
        "Costco income": [1, 0],
        "vector hit": [0.49, (1 - 0.49**2) ** 0.5],
        "keyword hit": [0.2, (1 - 0.2**2) ** 0.5],
    })
    monkeypatch.setattr(agent, "embeddings", embeddings)

    assert agent.retrieve(index, "Costco income") == []
    assert embeddings.queries == ["Costco income"]
    assert embeddings.batches == [["vector hit", "keyword hit"]]


def test_keyword_only_low_cosine_still_gives_no_evidence(monkeypatch):
    monkeypatch.setattr(agent, "keyword_search", lambda query, sources, k: [
        _doc(SOUTHWEST, 3, "Fleet of Boeing 737 aircraft.")
    ])
    monkeypatch.setattr(agent, "embeddings", FakeEmbeddings({
        api.EVIDENCE_PROMPT.format(
            claim="Apple's iPhone revenue grew 5% in fiscal 2025."
        ): [1, 0],
        "Fleet of Boeing 737 aircraft.": [0, 1],
    }))
    critic_calls = []
    _wire(monkeypatch, FakeIndex([]), critic_calls)

    response = api.verify_claim("Apple's iPhone revenue grew 5% in fiscal 2025.")

    assert response["reason_code"] == "no_evidence"
    assert response["evidence"] == []
    assert critic_calls == []
