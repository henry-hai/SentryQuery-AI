"""HTTP claim-verification API over the existing Researcher and Critic.

This is an interface layer, not new agent logic, in the same spirit as
mcp_server.py. It turns the pipeline into something a program can branch on:
POST a claim, get back PASS or FAIL with the exact passages the Critic ruled
against.

The graph is not rewritten and the Critic is not touched. A claim is not a
question, and the graph takes a question, so a claim is checked in two steps
that both already exist:

  1. run_pipeline asks the Researcher what the indexed documents say about the
     claim. The Researcher retrieves as it always does and the exact chunks land
     in PipelineResult.retrieved.
  2. critique, the same standalone Critic call evals/eval.py uses, rules on THE
     CLAIM ITSELF against those exact chunks.

APPROVE becomes PASS, REVISE becomes FAIL. The Critic's prompt, model and
temperature are unchanged, and its reason is passed through verbatim.

The verdict is deliberately two-valued, because a caller branching on a
guardrail has one decision to make. reason_code carries the distinction the
verdict drops, and both of its values are read off graph state rather than off
any model judgment, so they are exact:

  no_evidence      nothing relevant was retrieved, so the claim could not be
                   checked at all. Reported as FAIL, because for a guardrail
                   unverifiable must never read as approved.
  critic_rejected  passages were retrieved and the Critic ruled against the
                   claim.

It does NOT distinguish a claim the documents contradict from one they are
merely silent on. That distinction is the Critic's judgment rather than graph
state, so claiming it would be an overclaim. The reason text often carries it.

Run locally with:

    uvicorn api:app --reload

"""
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Same reason mcp_server.py does this: config.py calls a bare load_dotenv, which
# searches upward from the working directory, and a process manager may start
# this app from anywhere. Anchor the lookup to this file instead. It has to run
# before graph is imported, because that imports config, which builds the
# Pinecone client at import time. On Render there is no .env at all and the keys
# come from the environment, which load_dotenv never overrides.
load_dotenv(Path(__file__).resolve().parent / ".env")

from agent import document_name, page_number  # noqa: E402
from costcontrols import (  # noqa: E402
    AnswerCache,
    RateLimiter,
    RateLimitExceeded,
    env_int,
    normalize_key,
)
from graph import build_system, critique, run_pipeline  # noqa: E402

HERE = Path(__file__).resolve().parent
INDEX_HTML = HERE / "static" / "index.html"

# Longest claim accepted. A claim is one sentence or two. The cap is a cost
# ceiling, not a style rule: everything past it is a prompt, not a claim.
MAX_CLAIM_CHARS = 1000

# Same two ceilings the MCP server uses, same order, same module. The cache is
# consulted first, so a repeat claim costs nothing and spends no rate-limit
# budget. Only after both does a paid run start. Override with
# SENTRYQUERY_API_CACHE_SIZE and SENTRYQUERY_API_RATE_LIMIT.
CACHE_MAX_ENTRIES = env_int("SENTRYQUERY_API_CACHE_SIZE", 128)
RATE_LIMIT_PER_MINUTE = env_int("SENTRYQUERY_API_RATE_LIMIT", 10)

cache = AnswerCache(max_entries=CACHE_MAX_ENTRIES)
limiter = RateLimiter(
    max_per_minute=RATE_LIMIT_PER_MINUTE,
    call_label="/verify calls",
    override_env="SENTRYQUERY_API_RATE_LIMIT",
)

# What the Researcher is asked. It gathers evidence, it does not rule: the
# ruling is the Critic's, against the chunks this retrieves.
EVIDENCE_PROMPT = (
    "Search the indexed documents for every passage that bears on the following "
    "claim, whether it supports the claim or contradicts it, and quote what you "
    "find.\n\nClaim: {claim}"
)


# -----------------------------------------------------------------------------
# The compiled graph, built on first use
# -----------------------------------------------------------------------------
# Building it constructs live OpenAI and Pinecone clients, so it is deliberately
# lazy. The module stays importable, and testable, without keys, and the health
# check answers during a cold start without touching a paid service.
_system = None
_system_lock = threading.Lock()


def get_system():
    """Return the shared compiled graph, building it on first call."""
    global _system
    with _system_lock:
        if _system is None:
            _system = build_system()
        return _system


# -----------------------------------------------------------------------------
# Claim checking
# -----------------------------------------------------------------------------
def _evidence(chunks: list) -> list[dict[str, Any]]:
    """The exact chunks handed to the Critic, as JSON, top match first.

    Order is the retriever's, so the first entry is the closest match. Repeats
    are dropped because the Researcher may call the retriever more than once and
    get overlapping hits back. Nothing here is re-queried or re-ranked.
    """
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for doc in chunks:
        passage = doc.page_content.strip()
        key = (document_name(doc), page_number(doc), passage)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {"document": key[0], "page": key[1], "passage": passage}
        )
    return out


def verify_claim(claim: str, client_id: str = "unknown") -> dict[str, Any]:
    """Check one claim against the indexed documents and return the response body.

    Raises ValueError for an unusable claim and RateLimitExceeded when the
    caller is over its budget, both before any paid call is made.
    """
    if not claim or not claim.strip():
        raise ValueError("claim must be a non-empty string.")
    claim = claim.strip()
    if len(claim) > MAX_CLAIM_CHARS:
        raise ValueError(
            f"claim must be at most {MAX_CLAIM_CHARS} characters, got {len(claim)}."
        )

    key = normalize_key(claim)
    hit = cache.get(key)
    if hit is not None:
        # checked_at stays the time of the run that produced this, not now. A
        # cached response reports when the documents were actually consulted.
        return {**hit, "cached": True}

    limiter.check(client_id)

    system = get_system()
    result = run_pipeline(system, EVIDENCE_PROMPT.format(claim=claim))

    if not result.retrieved or result.tool_used == "none":
        # Nothing to check the claim against. Both conditions are read off graph
        # state, so this branch involves no judgment.
        response = {
            "claim": claim,
            "verdict": "FAIL",
            "reason_code": "no_evidence",
            "reason": (
                "No passage in the indexed documents bears on this claim, so it "
                "could not be verified against them."
            ),
            "evidence": [],
        }
    else:
        ruling = critique(system.critic_llm, claim, result.retrieved)
        approved = ruling.verdict == "APPROVE"
        response = {
            "claim": claim,
            "verdict": "PASS" if approved else "FAIL",
            "reason_code": None if approved else "critic_rejected",
            "reason": ruling.reason,
            "evidence": _evidence(result.retrieved),
        }

    response["cached"] = False
    response["checked_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cache.put(key, response)
    return response


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    """The request body. One field, so the contract is hard to get wrong."""

    claim: str = Field(description="The claim to check against the indexed documents.")


app = FastAPI(
    title="SentryQuery verify",
    description=(
        "Check a claim against a corpus of indexed documents. Returns PASS or "
        "FAIL with the exact passages the Critic ruled against."
    ),
    version="1.0.0",
)


def _client_id(request: Request) -> str:
    """Rate-limit key for the caller.

    Render terminates TLS at its own proxy and sets X-Forwarded-For, so the
    first hop there is the real caller. Behind a proxy that does not set it,
    every caller collapses onto one key and shares one budget, which is the
    conservative direction to fail in for a cost control.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/", include_in_schema=False)
def index():
    """The single static page. Plain HTML and CSS, no build step."""
    return FileResponse(INDEX_HTML)


@app.get("/healthz")
@app.get("/health", include_in_schema=False)
def healthz():
    """Liveness check. Makes no paid call, so it is also the warm-up ping that
    wakes a spun-down free instance before a demo."""
    return {"status": "ok"}


@app.post("/verify")
def verify(body: VerifyRequest, request: Request):
    """Check one claim. See verify_claim for the verdict rules."""
    try:
        return verify_claim(body.claim, client_id=_client_id(request))
    except ValueError as exc:
        return JSONResponse(
            status_code=400, content={"error": "invalid_claim", "detail": str(exc)}
        )
    except RateLimitExceeded as exc:
        retry_after = limiter.retry_after(_client_id(request)) or 60
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limited",
                "detail": str(exc),
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )
    except Exception as exc:
        # OpenAI or Pinecone failed. Report the class of failure without leaking
        # anything the upstream client put in its message.
        return JSONResponse(
            status_code=502,
            content={
                "error": "upstream_failure",
                "detail": f"{type(exc).__name__} while checking the claim.",
            },
        )
