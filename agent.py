"""Layer 1, the Researcher agent and its structured-output pipeline.

Builds the create_agent Researcher with capture-wrapped tools, and run_query,
which drafts an answer and packages it into AnswerSchema (with a fallback to
free text + query-replay sources).

The agent is built via create_agent from langchain.agents (LangChain's current
agent constructor), which compiles a LangGraph graph internally. It has two
tools: a hybrid retriever over the indexed documents, and a Tavily web-search
tool for live information.
"""
import os
import math
from dataclasses import dataclass, field
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from companies import companies_named, source_filter, strip_company_names
from keyword_search import search as keyword_search
from config import (
    INDEX_NAME,
    RESEARCHER_MODEL,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_MAX_K,
    RETRIEVAL_MIN_SCORE,
    SYSTEM_PROMPT,
    embeddings,
)
from schema import AnswerSchema


# -----------------------------------------------------------------------------
# Source helpers
# -----------------------------------------------------------------------------
def document_name(doc: Document) -> str:
    """The chunk's source file name, with any directory path stripped."""
    return os.path.basename(str(doc.metadata.get("source", "unknown")))


def page_number(doc: Document):
    """The chunk's 1-based page number, or "?" when the metadata has none.

    Metadata pages are 0-indexed floats, so they are shifted to the page number
    a human reading the PDF would see.
    """
    page = doc.metadata.get("page", "?")
    try:
        return int(float(page)) + 1
    except (TypeError, ValueError):
        return page


def _source_label(doc: Document) -> str:
    """Render a chunk's origin as 'filename p.N' from its metadata.

    Split into the two helpers above because the HTTP API reports the document
    and the page as separate JSON fields and must not re-derive either.
    """
    return f"{document_name(doc)} p.{page_number(doc)}"


def _dedup(labels: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def similarity_score(doc: Document):
    """The cosine similarity the chunk was retrieved at, or None if unscored."""
    return doc.metadata.get("score")


# -----------------------------------------------------------------------------
# Scored retrieval
# -----------------------------------------------------------------------------
def scored_search(
    vectorstore,
    query: str,
    search_filter: dict | None = None,
    fetch_k: int = RETRIEVAL_FETCH_K,
    min_score: float = RETRIEVAL_MIN_SCORE,
    max_k: int = RETRIEVAL_MAX_K,
) -> list[Document]:
    """Fetch up to fetch_k chunks, keep those at or above min_score, cap at max_k.

    Each kept chunk carries its similarity in metadata["score"], so the API can
    show why it was used. An empty list is a real outcome: nothing in the index
    is close enough to the query, and the caller must treat that as no evidence
    rather than fall back to whatever was nearest.
    """
    hits = vectorstore.similarity_search_with_score(
        query, k=fetch_k, filter=search_filter
    )
    kept: list[Document] = []
    for doc, score in sorted(hits, key=lambda pair: pair[1], reverse=True):
        if score < min_score:
            continue
        doc.metadata["score"] = round(float(score), 4)
        kept.append(doc)
        if len(kept) >= max_k:
            break
    return kept


def _chunk_key(doc: Document) -> tuple[str, int, str]:
    """Match the same passage from Pinecone and the local chunk file."""
    return (
        str(doc.metadata["source"]),
        int(doc.metadata["page"]),
        " ".join(doc.page_content.split()),
    )


def _cosine(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between a query and a fused candidate."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def retrieve(vectorstore, query: str, scope: list[str] | None = None) -> list[Document]:
    """Use scored vector search for multiple companies, hybrid search otherwise.

    A nonempty scope determines the company filter exactly, regardless of which
    companies the tool query names. With no scope, use companies in the query.
    Multi-company searches use a per-chunk cosine cut and return at most
    RETRIEVAL_MAX_K vector hits, without keyword search or rank fusion.
    A single-company search also searches without that name when stripping
    changes the text. The original and stripped vector rankings then fuse with
    the stripped keyword ranking. Otherwise one vector ranking fuses with the
    keyword ranking. Multi-company searches keep the names to distinguish the
    filings. Every candidate is scored against the original query. If the best
    score clears the gate, return up to RETRIEVAL_MAX_K in fused rank order.
    """
    companies = list(scope) if scope else companies_named(query)
    if len(companies) >= 2:
        return scored_search(vectorstore, query, source_filter(companies))
    search_filter = source_filter(companies)
    search_text = (
        strip_company_names(query, companies)
        if search_filter and len(companies) == 1 else query
    )
    allowed_sources = search_filter["source"]["$in"] if search_filter else None
    vector_queries = [query]
    if search_text != query:
        vector_queries.append(search_text)
    vector_rankings = [
        sorted(
            vectorstore.similarity_search_with_score(
                text, k=RETRIEVAL_FETCH_K, filter=search_filter
            ),
            key=lambda hit: hit[1],
            reverse=True,
        )
        for text in vector_queries
    ]
    keyword_hits = keyword_search(search_text, allowed_sources, RETRIEVAL_FETCH_K)

    candidates: dict[tuple[str, int, str], Document] = {}
    ranks: dict[tuple[str, int, str], float] = {}
    for vector_hits in vector_rankings:
        for rank, (doc, _score) in enumerate(vector_hits, start=1):
            key = _chunk_key(doc)
            candidates.setdefault(key, doc)
            ranks[key] = ranks.get(key, 0.0) + 1 / (60 + rank)
    for rank, doc in enumerate(keyword_hits, start=1):
        key = _chunk_key(doc)
        candidates.setdefault(key, doc)
        ranks[key] = ranks.get(key, 0.0) + 1 / (60 + rank)

    if not candidates:
        return []
    ranked_keys = sorted(candidates, key=lambda item: ranks[item], reverse=True)
    query_vector = embeddings.embed_query(query)
    document_vectors = embeddings.embed_documents(
        [candidates[key].page_content for key in ranked_keys]
    )
    scores = [_cosine(query_vector, vector) for vector in document_vectors]
    if max(scores) < RETRIEVAL_MIN_SCORE:
        return []

    kept = []
    for key, score in zip(ranked_keys, scores):
        doc = candidates[key]
        doc.metadata["score"] = round(score, 4)
        doc.metadata["rrf"] = ranks[key]
        kept.append(doc)
        if len(kept) >= RETRIEVAL_MAX_K:
            break
    return kept


# -----------------------------------------------------------------------------
# Agent construction
# -----------------------------------------------------------------------------
@dataclass
class AgentHandle:
    """Everything a single query needs, plus per-run capture buffers.

    `retrieved` and `web_sources` are filled by the tools during agent.invoke.
    The tools close over these exact list objects, so reset() clears them in
    place between runs rather than rebinding them.
    """

    agent: object
    vectorstore: PineconeVectorStore
    synth_llm: object
    retrieved: list  # exact Document chunks the retriever returned this run
    web_sources: list  # exact web URLs the web tool returned this run
    scope: list  # indexed companies the run's question or claim names

    def reset(self) -> None:
        self.retrieved.clear()
        self.web_sources.clear()
        self.scope.clear()


def build_agent() -> AgentHandle:
    """Construct the agent and return an AgentHandle.

    The retriever and web tools are wrapped so they capture the EXACT chunks and
    URLs they return into per-run buffers. That makes the answer's sources come
    from what the agent actually used (not a re-query of the index) and gives
    the Layer 2 Critic the same chunks to verify against.
    """
    # Connect to the existing Pinecone index without re-ingesting.
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Per-run capture buffers, closed over by the tools below and reset per run.
    retrieved: list[Document] = []
    web_sources: list[str] = []
    scope: list[str] = []

    @tool("search_documents")
    def search_documents(query: str) -> str:
        """Search the indexed enterprise documents for information about the
        organizations they cover, including business, financials, strategy,
        operations, products, and policies."""
        docs = retrieve(vectorstore, query, scope)
        retrieved.extend(docs)
        if not docs:
            return "No matching documents found."
        # Prefix each chunk with its source so the model can cite inline if it
        # wants; the schema's sources come from `retrieved`, not this text.
        return "\n\n".join(f"[{_source_label(d)}]\n{d.page_content}" for d in docs)

    tavily = TavilySearch(max_results=3)

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the public web for recent or live information not contained in
        the indexed documents, including current news, events, or market data about the
        organizations the documents cover, or related industry news. Use this
        only when the indexed documents do not contain the answer or the user
        explicitly asks about recent events."""
        result = tavily.invoke({"query": query})
        results = result.get("results", []) if isinstance(result, dict) else []
        for r in results:
            url = r.get("url")
            if url:
                web_sources.append(url)
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}"
            for r in results
        )

    # GPT-4o with temperature=0 gives deterministic, factual answers.
    llm = ChatOpenAI(model=RESEARCHER_MODEL, temperature=0)

    # create_agent (langchain.agents) compiles a LangGraph graph with LLM +
    # tool-execution nodes and the standard tool-calling loop.
    agent = create_agent(
        model=llm, tools=[search_documents, web_search], system_prompt=SYSTEM_PROMPT
    )

    # CRITICAL (Layer 1 guard): structured output is applied ONLY to this
    # separate synthesis model, never to the agent itself. create_agent offers a
    # response_format= that would structure the answer inside the agent, but
    # constraining the agent's own output can interfere with its tool-calling
    # loop and would bypass our free-text fallback. So the agent stays plain and
    # structuring happens as a final, post-agent step in run_query.
    synth_llm = llm.with_structured_output(AnswerSchema)

    return AgentHandle(agent, vectorstore, synth_llm, retrieved, web_sources, scope)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def extract_search_queries(messages) -> list[str]:
    """Return every query the agent passed to search_documents.

    LangGraph appends every step (AI messages with tool_calls, ToolMessages
    with results) to the messages list. Walking the list lets the fallback path
    replay what the agent looked up so it can display matching sources.
    """
    queries: list[str] = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") == "search_documents":
                q = (tc.get("args") or {}).get("query")
                if q:
                    queries.append(q)
    return queries


def _replay_sources(
    vectorstore: PineconeVectorStore, queries: list[str], scope: list[str] | None = None
) -> list[str]:
    """Fallback source discovery: re-run each retriever query and label the hits.

    Used only when schema synthesis fails, to preserve the pre-Layer-1 behavior.
    It goes through the same scored, filtered retrieval as the tool, so a
    replay cannot cite a chunk when the query gate or company filter rejected it.
    """
    labels: list[str] = []
    for q in queries:
        for doc in retrieve(vectorstore, q, scope):
            labels.append(_source_label(doc))
    return _dedup(labels)


@dataclass
class QueryResult:
    """A packaged answer for the UI/eval, from either the schema path or fallback."""

    answer: str
    sources: list[str]
    tool_used: str  # "docs" | "web" | "none"
    confidence: Optional[float]  # None when schema synthesis failed (fallback)
    schema_ok: bool
    retriever_calls: int
    retrieved: list  # exact Document chunks (for UI expanders + Layer 2 Critic)
    web_sources: list[str] = field(default_factory=list)


_SYNTHESIS_INSTRUCTIONS = (
    "You are finalizing an assistant's answer into a structured object.\n"
    "- Copy the draft answer faithfully into 'answer': do not add, remove, or "
    "alter any factual claim; only fix obvious formatting.\n"
    "- Set 'sources' to exactly the provided source list.\n"
    "- Set 'tool_used' to the provided value.\n"
    "- Set 'confidence' to your genuine 0-1 estimate that every claim in the "
    "answer is supported by those sources."
)


def run_query(handle: AgentHandle, query: str) -> QueryResult:
    """Run one query end to end and return a packaged, schema-validated result.

    The agent runs first (its tool loop is untouched) and produces free text;
    the final answer is then constrained to AnswerSchema in a separate synthesis
    step. If synthesis raises or does not validate, fall back to the free-text
    answer with query-replay sources and no fabricated confidence.
    """
    handle.reset()
    handle.scope.extend(companies_named(query))
    response = handle.agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"]

    answer_text = messages[-1].content
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)

    queries = extract_search_queries(messages)
    web_used = any(
        tc.get("name") == "web_search"
        for msg in messages
        for tc in getattr(msg, "tool_calls", None) or []
    )
    docs_used = len(handle.retrieved) > 0
    tool_used = "docs" if docs_used else ("web" if web_used else "none")
    sources = _dedup(
        [_source_label(d) for d in handle.retrieved] + list(handle.web_sources)
    )

    try:
        prompt = (
            f"{_SYNTHESIS_INSTRUCTIONS}\n\n"
            f"Draft answer:\n{answer_text}\n\n"
            f"Provided sources: {sources}\n"
            f"tool_used: {tool_used}"
        )
        schema = handle.synth_llm.invoke(prompt)
        # Overwrite the deterministic fields so citations/routing can't drift.
        schema.sources = sources
        schema.tool_used = tool_used
        return QueryResult(
            answer=schema.answer or answer_text,
            sources=sources,
            tool_used=tool_used,
            confidence=schema.confidence,
            schema_ok=True,
            retriever_calls=len(queries),
            retrieved=list(handle.retrieved),
            web_sources=_dedup(list(handle.web_sources)),
        )
    except Exception:
        # Fallback keeps the app working if structuring fails: free-text answer,
        # query-replay sources, and confidence=None so the UI shows no number.
        return QueryResult(
            answer=answer_text,
            sources=_replay_sources(handle.vectorstore, queries, handle.scope) or sources,
            tool_used=tool_used,
            confidence=None,
            schema_ok=False,
            retriever_calls=len(queries),
            retrieved=list(handle.retrieved),
            web_sources=_dedup(list(handle.web_sources)),
        )
