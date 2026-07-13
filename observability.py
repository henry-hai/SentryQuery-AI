"""Observability for SentryQuery: optional LangSmith tracing + always-on
structured run logging.

LangSmith tracing is fully env-driven and optional. Set these in .env to enable
it - LangChain/LangGraph pick them up automatically, so no code change is needed
to start tracing the two-agent graph (Researcher tool calls, the drafted schema,
the Critic verdict, and any revision loop all show up as nested runs):

    LANGSMITH_TRACING=true          # (or the older LANGCHAIN_TRACING_V2=true)
    LANGSMITH_API_KEY=lsv2_...      # (or the older LANGCHAIN_API_KEY=...)
    LANGSMITH_PROJECT=sentryquery   # optional, names the project in LangSmith

If those are unset the app runs identically with tracing off - nothing here ever
hard-fails on a missing tracing key. Independently of LangSmith, every pipeline
run is emitted as a structured log record (tool routing, Critic verdict,
revision count, confidence), so there is always working observability.
"""
import json
import logging
import os

logger = logging.getLogger("sentryquery")


def _configure() -> None:
    """Attach a stream handler once (idempotent across imports)."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


_configure()


def tracing_enabled() -> bool:
    """True only if a tracing flag AND an API key are set (either naming style)."""
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return str(flag).lower() == "true" and bool(key)


def trace_status() -> str:
    """A one-line, human-readable description of the current tracing state."""
    if tracing_enabled():
        project = (
            os.getenv("LANGSMITH_PROJECT")
            or os.getenv("LANGCHAIN_PROJECT")
            or "default"
        )
        return f"LangSmith tracing: ON (project={project})"
    return (
        "LangSmith tracing: OFF — set LANGSMITH_TRACING=true and LANGSMITH_API_KEY "
        "in .env to enable. Structured run logging is active regardless."
    )


def log_run(query: str, result) -> None:
    """Emit one pipeline run as a structured log record.

    This is the always-on observability fallback: it captures the Researcher's
    tool routing, the Critic's verdict, and the revision count for every run,
    whether or not LangSmith tracing is enabled.
    """
    record = {
        "query": query[:200],
        "tool_used": result.tool_used,
        "retriever_calls": result.retriever_calls,
        "n_sources": len(result.sources),
        "verdict": result.verdict,
        "revisions": result.revisions,
        "confidence": result.confidence,
        "schema_ok": result.schema_ok,
    }
    if result.verdict == "REVISE" or result.revisions > 0:
        record["critic_reason"] = result.critic_reason
    logger.info("run %s", json.dumps(record, default=str))