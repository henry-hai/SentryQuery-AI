"""SentryQuery — agentic AI assistant over a set of indexed enterprise documents.

Two run modes share this single entry point:
  - Ingestion: `python sentry_query.py --ingest` rebuilds the Pinecone index
    from PDFs in ./docs/.
  - UI: `python -m streamlit run sentry_query.py` launches the Streamlit app.

The agent is built via create_agent from langchain.agents (LangChain's current
agent constructor), which compiles a LangGraph graph internally. It has two
tools available: a Pinecone-backed retriever over the indexed documents, and a
Tavily web-search tool for live information. Each answer is packaged into the
structured AnswerSchema (see schema.py) at a final synthesis step.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from schema import AnswerSchema

# Load OPENAI_API_KEY, PINECONE_API_KEY, and TAVILY_API_KEY from .env
# so secrets stay out of source control.
load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# A single Pinecone client instance is shared across ingestion and query modes.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# The Pinecone index must be pre-created with dimension 1536 (matches
# text-embedding-3-small) and cosine similarity.
INDEX_NAME = "sentry-index"

# The system prompt scopes the agent to the indexed documents, instructs it
# which tool to prefer for which kind of question, and tells it to refuse
# off-topic queries. This is the primary prompt-engineering surface in the app.
# It is deliberately corpus-neutral: it never names a specific company or file,
# so the same code works over whatever PDFs are ingested into ./docs/.
SYSTEM_PROMPT = """You are SentryQuery, an assistant that answers questions grounded in a set of indexed enterprise documents.

Tool routing:
- For questions whose answer could plausibly be found in the indexed documents
  (the organizations they cover, their business, financials, strategy,
  operations, products, or policies): use search_documents FIRST.
- For questions that require live or recent information not contained in the
  documents (current news, today's events, current market data): use
  web_search. Do not answer from memory — always call the tool, and summarize
  the results that come back. Do not say "I couldn't find" if the tool returned
  any content — report what it returned.
- For questions unrelated to the indexed documents and their subject matter
  (general chit-chat, the weather, unrelated coding help, and so on): politely
  refuse and explain that you only answer questions about the indexed documents.

Answering style:
- Be concise. Ground every claim in retrieved content.
- If neither tool returns useful information, say
  "I don't have that information in the indexed documents or available web
  sources." Do not speculate or rely on outside knowledge.
- When quoting specifics, prefer short verbatim phrases over paraphrases.
"""

# text-embedding-3-small produces 1536-dim vectors and is ~5x cheaper
# than ada-002 with comparable retrieval quality on MTEB benchmarks.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# -----------------------------------------------------------------------------
# Ingestion Pipeline
# -----------------------------------------------------------------------------
def run_ingest() -> None:
    """Rebuild the Pinecone index from PDFs in ./docs/.

    Pipeline: load -> split -> embed -> upsert. The index is cleared first so
    re-runs are idempotent and stale chunks from removed files do not linger.
    """
    # Step 1: clear any existing vectors so the index reflects ./docs/ exactly.
    # Promotes idempotencey and ensures removed files don't leave stale chunks behind.
    existing_index = pc.Index(INDEX_NAME)
    existing_index.delete(delete_all=True)
    print("Cleared existing vectors.")

    # Step 2: load every PDF under ./docs/ into LangChain Document objects
    # (one Document per PDF page; metadata includes source path and page index).
    loader = PyPDFDirectoryLoader("./docs")
    docs = loader.load()

    # Step 3: split into ~1000-character chunks with 200-character overlap.
    # The "recursive" splitter tries paragraph -> sentence -> word boundaries
    # in priority order so chunks land on natural breaks. Overlap preserves
    # context across chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # Step 4: embed each chunk with OpenAi's text-embedding-3-small -> update & inserte to
    # Pinecone. The embedding step happens implicitly inside from_documents.
    PineconeVectorStore.from_documents(splits, embeddings, index_name=INDEX_NAME)
    print(f"Ingestion complete: {len(splits)} chunks indexed.")


# -----------------------------------------------------------------------------
# Source helpers
# -----------------------------------------------------------------------------
def _source_label(doc: Document) -> str:
    """Render a chunk's origin as 'filename p.N' from its metadata."""
    src = os.path.basename(str(doc.metadata.get("source", "unknown")))
    page = doc.metadata.get("page", "?")
    try:
        page = int(float(page))  # metadata pages come through as floats
    except (TypeError, ValueError):
        pass
    return f"{src} p.{page}"


def _dedup(labels: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


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

    def reset(self) -> None:
        self.retrieved.clear()
        self.web_sources.clear()


def build_agent() -> AgentHandle:
    """Construct the agent and return an AgentHandle.

    The retriever and web tools are wrapped so they capture the EXACT chunks and
    URLs they return into per-run buffers. That makes the answer's sources come
    from what the agent actually used (not a re-query of the index) and gives
    the Layer 2 Critic the same chunks to verify against.
    """
    # Connect to the existing Pinecone index without re-ingesting.
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # k=5 returns the top-5 most similar chunks per query (LangChain defaults to 4).
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Per-run capture buffers, closed over by the tools below and reset per run.
    retrieved: list[Document] = []
    web_sources: list[str] = []

    @tool("search_documents")
    def search_documents(query: str) -> str:
        """Search the indexed enterprise documents for information about the
        organizations they cover — their business, financials, strategy,
        operations, products, and policies."""
        docs = retriever.invoke(query)
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
        the indexed documents — current news, events, or market data about the
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
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # create_agent (langchain.agents) compiles a LangGraph graph with LLM +
    # tool-execution nodes and the standard tool-calling loop.
    agent = create_agent(
        model=llm, tools=[search_documents, web_search], system_prompt=SYSTEM_PROMPT
    )

    # CRITICAL (Layer 1 guard): structured output is applied ONLY to this
    # separate synthesis model — never to the agent itself. create_agent offers a
    # response_format= that would structure the answer inside the agent, but
    # constraining the agent's own output can interfere with its tool-calling
    # loop and would bypass our free-text fallback. So the agent stays plain and
    # structuring happens as a final, post-agent step in run_query.
    synth_llm = llm.with_structured_output(AnswerSchema)

    return AgentHandle(agent, vectorstore, synth_llm, retrieved, web_sources)


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


def _replay_sources(vectorstore: PineconeVectorStore, queries: list[str]) -> list[str]:
    """Fallback source discovery: re-run each retriever query and label the hits.

    Used only when schema synthesis fails, to preserve the pre-Layer-1 behavior.
    """
    labels: list[str] = []
    for q in queries:
        for doc in vectorstore.similarity_search(q, k=5):
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
        )
    except Exception:
        # Fallback keeps the app working if structuring fails: free-text answer,
        # query-replay sources, and confidence=None so the UI shows no number.
        return QueryResult(
            answer=answer_text,
            sources=_replay_sources(handle.vectorstore, queries) or sources,
            tool_used=tool_used,
            confidence=None,
            schema_ok=False,
            retriever_calls=len(queries),
            retrieved=list(handle.retrieved),
        )


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def run_ui() -> None:
    """Render the chat-style UI and dispatch each query through the agent."""
    handle = build_agent()

    st.title("SentryQuery Agentic AI Assistant")
    st.caption("Powered by LangChain, LangGraph, Pinecone, GPT-4o, and Tavily")

    query = st.text_input("Ask about the indexed documents:")

    if not (st.button("Run") and query):
        return

    with st.spinner("Agent thinking..."):
        result = run_query(handle, query)

    st.write(result.answer)

    # Confidence is only rendered when the schema step produced a genuine,
    # model-derived value — we never dress up a fabricated number.
    if result.confidence is not None:
        st.caption("Model self-reported confidence")
        st.progress(result.confidence, text=f"{result.confidence:.0%}")

    # Tool-routing feedback for the cases with no document sources to show.
    if result.tool_used == "web":
        st.info("Tavily web search was used for this answer.")
    elif result.tool_used == "none":
        st.info("Agent answered without consulting any tools.")

    # Document sources: the EXACT chunks the agent retrieved, in expanders.
    if result.retrieved:
        st.divider()
        st.subheader("Sources")
        seen: set[str] = set()
        for doc in result.retrieved:
            label = _source_label(doc)
            if label in seen:
                continue
            seen.add(label)
            with st.expander(label):
                st.write(doc.page_content)

    # Web sources: any URLs the web tool returned, as links.
    web_urls = [s for s in result.sources if s.startswith("http")]
    if web_urls:
        if not result.retrieved:
            st.divider()
            st.subheader("Sources")
        st.markdown("**Web results**")
        for url in web_urls:
            st.markdown(f"- [{url}]({url})")


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        run_ingest()
    else:
        run_ui()