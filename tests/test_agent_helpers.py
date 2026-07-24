"""Unit tests for the pure source-handling helpers in agent.py.

These do not touch the agent, Pinecone, or OpenAI, so they run offline. They
guard the citation path: page numbers shown to users must be 1-based, and source
lists must de-duplicate while preserving order.
"""
from types import SimpleNamespace

from langchain_core.documents import Document

from agent import _dedup, _source_label, extract_search_queries


def test_source_label_renders_1_based_page_from_zero_indexed_float():
    doc = Document(page_content="x", metadata={"source": "/a/b/costco-10k.pdf", "page": 4.0})
    assert _source_label(doc) == "costco-10k.pdf p.5"


def test_source_label_handles_missing_page():
    doc = Document(page_content="x", metadata={"source": "deere-10k.pdf"})
    assert _source_label(doc) == "deere-10k.pdf p.?"


def test_dedup_preserves_first_seen_order():
    assert _dedup(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_extract_search_queries_returns_only_document_searches():
    messages = [
        SimpleNamespace(
            tool_calls=[
                {"name": "search_documents", "args": {"query": "net sales"}},
                {"name": "web_search", "args": {"query": "stock price"}},
            ]
        ),
        SimpleNamespace(tool_calls=[{"name": "search_documents", "args": {"query": "revenue"}}]),
        SimpleNamespace(tool_calls=None),
    ]
    assert extract_search_queries(messages) == ["net sales", "revenue"]
