"""SentryQuery — agentic AI assistant over a set of indexed enterprise documents.

This is the entry point for both run modes:
  - Ingestion: `python sentry_query.py --ingest` rebuilds the Pinecone index
    from PDFs in ./docs/.
  - UI: `python -m streamlit run sentry_query.py` launches the Streamlit app.

The implementation is split across modules:
  - config.py — shared settings (Pinecone, index, models, system prompt).
  - schema.py — the AnswerSchema / CriticVerdict data contracts.
  - agent.py  — Layer 1: the Researcher agent and its structured-output pipeline.
  - graph.py  — Layer 2: the Researcher + Critic multi-agent graph.
"""
import sys

import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import pc, INDEX_NAME, embeddings
from agent import _source_label
from graph import build_system, run_pipeline
from observability import trace_status


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
# Streamlit UI
# -----------------------------------------------------------------------------
def run_ui() -> None:
    """Render the chat-style UI and dispatch each query through the two-agent graph."""
    system = build_system()

    st.title("SentryQuery Agentic AI Assistant")
    st.caption("Powered by LangChain, LangGraph, Pinecone, GPT-4o, and Tavily")

    # Observability status (Layer 3): shows whether LangSmith tracing is on.
    st.sidebar.subheader("Observability")
    st.sidebar.caption(trace_status())

    query = st.text_input("Ask about the indexed documents:")

    if not (st.button("Run") and query):
        return

    with st.spinner("Researcher and Critic working..."):
        result = run_pipeline(system, query)

    st.write(result.answer)

    # Critic verdict badge (Layer 2). Only shown for document-grounded answers —
    # a groundedness check against retrieved chunks is meaningless for a web
    # answer or a refusal, so we don't dress those up as "verified".
    n_sources = len({_source_label(d) for d in result.retrieved})
    if result.tool_used == "docs":
        if result.verdict == "APPROVE" and result.revisions == 0:
            st.success(
                f"✓ Verified — every claim grounded in {n_sources} retrieved source(s)."
            )
        elif result.verdict == "APPROVE" and result.revisions > 0:
            st.warning(
                f"⚠ Revised {result.revisions}× before answering, then verified. "
                f"Critic note: {result.critic_reason}"
            )
        else:  # REVISE persisted to the revision cap
            st.warning(
                f"⚠ Revised {result.revisions}× — the Critic still flags: {result.critic_reason}"
            )

    # Confidence is only rendered when the schema step produced a genuine,
    # model-derived value — and not on refusals, where "confidence that the
    # answer is grounded in sources" is meaningless (there are no sources).
    if result.confidence is not None and result.tool_used != "none":
        st.caption("Model self-reported confidence")
        st.progress(result.confidence, text=f"{result.confidence:.0%}")

    # Tool-routing feedback for the cases with no document sources to show.
    if result.tool_used == "web":
        st.info(
            "Answer from live web search (Tavily) — not verified against the indexed documents."
        )
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