"""MCP server exposing SentryQuery over the Model Context Protocol.

This is an interface layer, not new agent logic. It wraps the existing
Researcher -> Critic graph (see graph.py) so any MCP client can query the
indexed corpus through one standard tool, `ask_corpus`, and receive the full
structured result: the answer, its sources, which tool produced it, the model's
confidence, and the Critic's groundedness verdict.

Run it over stdio, the transport an MCP client uses to launch a local server:

    python mcp_server.py

Keys come from the same .env the rest of the app already uses (config.py calls
load_dotenv on import). Nothing here reads, stores, or hard-codes a key.
"""
import sys
import threading
from typing import Any

from mcp.server.mcpserver import MCPServer

from graph import build_system, run_pipeline


# The compiled Researcher -> Critic graph, built once on first use. Building it
# constructs live OpenAI and Pinecone clients, so it is deliberately lazy: the
# module stays importable, and testable, without keys.
_system = None
_system_lock = threading.Lock()


def get_system():
    """Return the shared compiled graph, building it on first call."""
    global _system
    with _system_lock:
        if _system is None:
            _system = build_system()
        return _system


def _to_response(result) -> dict[str, Any]:
    """Flatten a PipelineResult into the MCP tool's JSON response.

    Carries the AnswerSchema fields (answer, sources, tool_used, confidence)
    plus the Critic's verdict and reason, so a client sees the groundedness
    result and not just answer text.
    """
    return {
        "answer": result.answer,
        "sources": list(result.sources),
        "tool_used": result.tool_used,
        "confidence": result.confidence,
        "verdict": result.verdict,
        "critic_reason": result.critic_reason,
        "revisions": result.revisions,
    }


def answer_question(question: str) -> dict[str, Any]:
    """Answer `question` through the existing pipeline and package the result."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")
    return _to_response(run_pipeline(get_system(), question))


mcp = MCPServer(
    name="sentryquery",
    version="1.0.0",
    instructions=(
        "SentryQuery answers questions grounded in a private corpus of indexed "
        "enterprise PDFs. Use ask_corpus for anything about those documents. "
        "Every answer is checked by a Critic agent against the exact retrieved "
        "chunks, and that verdict is returned alongside the answer."
    ),
)


@mcp.tool(
    name="ask_corpus",
    title="Ask the indexed corpus",
    description=(
        "Ask a question about the indexed enterprise documents. Runs the "
        "Researcher -> Critic pipeline and returns the answer, its sources, "
        "which tool produced it (docs, web, or none), the model's confidence, "
        "and the Critic's groundedness verdict (APPROVE or REVISE) with its "
        "reason."
    ),
)
def ask_corpus(question: str) -> dict[str, Any]:
    """MCP binding for answer_question, kept thin so the answering logic stays
    independent of the protocol layer."""
    return answer_question(question)


def main() -> None:
    """Serve over stdio. Status goes to stderr, since stdout is the protocol channel."""
    print("SentryQuery MCP server starting on stdio.", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
