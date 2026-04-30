"""SentryQuery — agentic AI assistant over Ingram Micro documents.

Two run modes share this single entry point:
  - Ingestion: `python sentry_query.py --ingest` rebuilds the Pinecone index
    from PDFs in ./docs/.
  - UI: `python -m streamlit run sentry_query.py` launches the Streamlit app.

The agent is a LangGraph ReAct graph built via langgraph.prebuilt.create_react_agent,
which compiles a LangGraph StateGraph internally. It has two tools available:
a Pinecone-backed retriever over the indexed documents, and a Tavily
web-search tool for live information.
"""
import os
import sys
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent as compile_agent
from langchain_core.tools.retriever import create_retriever_tool
from langchain_tavily import TavilySearch

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

# The system prompt restricts the agent to Ingram Micro topics, instructs it
# which tool to prefer for which kind of question, and tells it to refuse
# off-topic queries. This is the primary prompt-engineering surface in the app.
SYSTEM_PROMPT = """You are SentryQuery, an enterprise document assistant for Ingram Micro.

Tool routing:
- For questions about Ingram Micro's policies, products, Xvantage platform, AI
  strategy, operations, or anything that could plausibly be in the indexed
  internal documents: use search_ingram_micro_docs FIRST.
- For questions that require live or recent information (current news, today's
  events, recent public announcements about Ingram Micro or its partners): use
  web_search. Do not answer from memory — always call the tool, and summarize
  the results that come back. Do not say "I couldn't find" if the tool returned
  any content — report what it returned.
- For questions unrelated to Ingram Micro: politely refuse and explain that you
  only answer Ingram Micro-related questions.

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
# Agent construction
# -----------------------------------------------------------------------------
def build_agent():
    """Construct the ReAct agent and return (agent, vectorstore).

    Returning the vectorstore alongside the agent lets the UI re-fetch the
    chunks the agent consulted, in order to display sources beneath the answer.
    """
    # Connect to the existing Pinecone index without re-ingesting.
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Wrap the vectorstore as a Retriever. k=5 returns the top-5 most similar
    # chunks per query, overriding LangChain's default of k=4.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # The retriever Tool exposes the vectorstore to the LLM via a name and a
    # natural-language description. The description is itself a prompt: the
    # LLM reads it to decide WHEN to call this tool.
    retriever_tool = create_retriever_tool(
        retriever,
        "search_ingram_micro_docs",
        "Search Ingram Micro documents for information about the company, "
        "Xvantage platform, AI strategy, policies, products, and operations.",
    )

    # Tavily web search: returns LLM-friendly summarized results that the LLM
    # reads as a ToolMessage before composing its final answer.
    web_search = TavilySearch(
        max_results=3,
        name="web_search",
        description=(
            "Search the public web for recent or live information "
            "about Ingram Micro, Xvantage, partners, or industry news. "
            "Use this only when the indexed documents do not contain "
            "the answer or the user explicitly asks about recent events."
        ),
    )
    tools = [retriever_tool, web_search]

    # GPT-4o with temperature=0 gives deterministic, factual answers.
    # Ideal for grounded document Q&A. 
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # create_react_agent (langgraph.prebuilt) compiles a LangGraph StateGraph with
    # two nodes (LLM, tool-execution) and the standard ReAct conditional edges.
    # prompt= accepts a plain string and prepends it as a SystemMessage each invocation.
    agent = compile_agent(llm, tools, prompt=SYSTEM_PROMPT) 
    return agent, vectorstore


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def extract_search_queries(messages) -> list[str]:
    """Return every query the agent passed to search_ingram_micro_docs.

    LangGraph appends every step (AI messages with tool_calls, ToolMessages
    with results) to the messages list. Walking the list lets the UI replay
    what the agent looked up so it can display matching sources.
    """
    queries: list[str] = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") == "search_ingram_micro_docs":
                q = (tc.get("args") or {}).get("query")
                if q:
                    queries.append(q)
    return queries


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def run_ui() -> None:
    """Render the chat-style UI and dispatch each query through the agent."""
    agent, vectorstore = build_agent()

    st.title("SentryQuery Agentic AI Assistant")
    st.caption("Powered by LangGraph, LangChain, Pinecone, GPT-4o, and Tavily")

    query = st.text_input("Ask about Ingram Micro:")

    if not (st.button("Run") and query):
        return

    with st.spinner("Agent thinking..."):
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # The final answer is the last message in the returned message list.
    st.write(response["messages"][-1].content)

    # If the agent consulted the document retriever, replay each search query
    # against the vectorstore to surface which PDF/page each answer came from.
    # This is a UX layer separate from the agent's reasoning — it does not
    # affect the answer, only what the user sees as supporting evidence.
    web_search_used = any(
        tc.get("name") == "web_search"
        for msg in response["messages"]
        for tc in getattr(msg, "tool_calls", None) or []
    )
    if web_search_used:
        st.info("Tavily web search was used for this answer.")

    search_queries = extract_search_queries(response["messages"])
    if not search_queries:
        if not web_search_used:
            st.info("Agent answered without consulting any tools.")
        return

    st.divider()
    st.subheader("Sources")
    seen: set[tuple[str, object]] = set()
    for q in search_queries:
        for doc in vectorstore.similarity_search(q, k=5):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            key = (str(src), page)
            if key in seen:
                continue
            seen.add(key)
            with st.expander(f"{os.path.basename(str(src))} — page {page}"):
                st.write(doc.page_content)


if __name__ == "__main__":
    if "--ingest" in sys.argv:
        run_ingest()
    else:
        run_ui()
