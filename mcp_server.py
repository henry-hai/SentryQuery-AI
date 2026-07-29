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

Two cost controls live here, because every uncached query costs a paid OpenAI
embedding plus a Pinecone query and an MCP server is meant to be left running:
a bounded in-process answer cache (CACHE_MAX_ENTRIES) and a per-client rate
limit (RATE_LIMIT_PER_MINUTE). Both are in-process only, so they reset on
restart and are not shared between processes.
"""
import os
import sys
import threading
import time
from collections import OrderedDict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from mcp.server.mcpserver import Context, MCPServer

# An MCP client starts its servers itself, from whatever working directory it
# happens to have (Claude Desktop uses /). The bare load_dotenv() in config.py
# searches upward from the cwd, so it would not find this repo's .env and the
# server would die on import with a missing-key error. Anchor the lookup to this
# file's directory instead.
#
# This has to run before graph is imported, because that imports config, which
# builds the Pinecone client at import time. Hence the deliberate import-order
# exception below. load_dotenv does not override variables already present in
# the environment, so a key injected by the client still wins.
load_dotenv(Path(__file__).resolve().parent / ".env")

from graph import build_system, run_pipeline  # noqa: E402


def _env_int(name: str, default: int) -> int:
    """Read a positive integer setting from the environment, falling back to
    `default` when it is unset, blank, or not a usable number."""
    try:
        value = int(os.getenv(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# How many distinct questions the answer cache holds. At the cap the oldest
# entry by insertion is evicted. 128 is deliberately small: the cache exists to
# stop a client re-paying for a question it already asked, not to be a durable
# store. Override with SENTRYQUERY_MCP_CACHE_SIZE.
CACHE_MAX_ENTRIES = _env_int("SENTRYQUERY_MCP_CACHE_SIZE", 128)

# Maximum ask_corpus calls one client may make per rolling 60 seconds. Cache
# hits cost nothing and are not counted against it. Default 10, override with
# SENTRYQUERY_MCP_RATE_LIMIT.
RATE_LIMIT_PER_MINUTE = _env_int("SENTRYQUERY_MCP_RATE_LIMIT", 10)

# Client key used when the transport gives us no session identity. Over stdio
# one process serves exactly one client, so a single key is correct there.
DEFAULT_CLIENT_ID = "stdio"


class RateLimitExceeded(RuntimeError):
    """Raised instead of firing a paid call when a client is over its limit."""


def normalize_question(question: str) -> str:
    """Cache key for a question: trimmed, lowercased, whitespace collapsed.

    So "  What were TOTAL   net sales? " and "what were total net sales?" hit
    one cache entry instead of costing two paid runs.
    """
    return " ".join(question.split()).lower()


class AnswerCache:
    """A bounded, thread-safe cache of normalized question -> tool response.

    Insertion-ordered, so at the cap the oldest entry is evicted first. A hit
    does not refresh an entry's position: entries age out by when they were
    first stored, not by when they were last read. Values are deep-copied in and
    out, so a caller holding a returned response cannot mutate the cached copy
    (a shallow copy would still share the sources list).
    """

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES):
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            hit = self._entries.get(key)
            return deepcopy(hit) if hit is not None else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._entries[key] = deepcopy(value)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class RateLimiter:
    """Per-client sliding-window limiter over a rolling 60-second window.

    Each client key keeps the timestamps of its recent admitted calls. A call is
    admitted only if fewer than `max_per_minute` of them still fall inside the
    window, so an over-limit client is turned away before any paid call starts.
    """

    WINDOW_SECONDS = 60.0

    def __init__(self, max_per_minute: int = RATE_LIMIT_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self._calls: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, client_id: str, now: Optional[float] = None) -> None:
        """Record one call for `client_id`, or raise RateLimitExceeded.

        `now` is injectable so tests can drive the window without sleeping.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            recent = self._calls.setdefault(client_id, deque())
            while recent and now - recent[0] >= self.WINDOW_SECONDS:
                recent.popleft()
            if len(recent) >= self.max_per_minute:
                retry_in = self.WINDOW_SECONDS - (now - recent[0])
                raise RateLimitExceeded(
                    f"Rate limit reached: {self.max_per_minute} ask_corpus calls per "
                    f"minute per client. Retry in {retry_in:.0f}s, or raise "
                    f"SENTRYQUERY_MCP_RATE_LIMIT. No paid model or Pinecone call was made."
                )
            recent.append(now)

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


cache = AnswerCache()
limiter = RateLimiter()


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


def _to_response(result, cached: bool = False) -> dict[str, Any]:
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
        "cached": cached,
    }


def answer_question(question: str, client_id: str = DEFAULT_CLIENT_ID) -> dict[str, Any]:
    """Answer `question` through the existing pipeline, with both cost controls.

    The order is deliberate. The cache is consulted first, so a repeat question
    costs nothing, spends no rate-limit budget, and re-embeds nothing. The
    limiter runs next. Only after both is a paid pipeline run started.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")

    key = normalize_question(question)
    hit = cache.get(key)
    if hit is not None:
        return {**hit, "cached": True}

    limiter.check(client_id)

    response = _to_response(run_pipeline(get_system(), question), cached=False)
    cache.put(key, response)
    return response


def _client_id(ctx) -> str:
    """Rate-limit key for the caller: one key per open MCP session.

    The SDK creates a ServerSession per connected client, so its identity is the
    closest thing the protocol gives us to a client id. Falls back to a single
    shared key when there is no session, which is the correct behaviour over
    stdio (one process, one client).
    """
    if ctx is None:
        return DEFAULT_CLIENT_ID
    try:
        session = ctx.session
    except Exception:
        return DEFAULT_CLIENT_ID
    return f"session-{id(session)}" if session is not None else DEFAULT_CLIENT_ID


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
        "reason. Repeated questions are served from an in-process cache, and "
        "calls are rate limited per client."
    ),
)
def ask_corpus(question: str, ctx: Context = None) -> dict[str, Any]:
    """MCP binding for answer_question, kept thin so the answering logic stays
    independent of the protocol layer. `ctx` is injected by the SDK (it is not
    part of the tool's public schema) and is used only to identify the calling
    client for rate limiting."""
    return answer_question(question, client_id=_client_id(ctx))


def main() -> None:
    """Serve over stdio. Status goes to stderr, since stdout is the protocol channel."""
    print(
        f"SentryQuery MCP server starting on stdio (cache cap {CACHE_MAX_ENTRIES} "
        f"entries, {RATE_LIMIT_PER_MINUTE} calls/min per client).",
        file=sys.stderr,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
