"""Layer 2 — the Researcher + Critic multi-agent graph.

Wires the Researcher (see agent.py) and a Critic as two distinct nodes in an
explicit LangGraph StateGraph. The Critic verifies each drafted answer against
the EXACT chunks the Researcher retrieved (captured in graph state, never
re-queried) and returns APPROVE or REVISE; on REVISE the graph loops back to the
Researcher with the note, capped at MAX_REVISIONS passes.
"""
from dataclasses import dataclass
from typing import Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from config import CRITIC_MODEL, MAX_REVISIONS
from schema import CriticVerdict
from agent import AgentHandle, build_agent, run_query, _source_label


CRITIC_INSTRUCTIONS = (
    "You are a groundedness critic. You are given an ANSWER and the SOURCES that "
    "were retrieved to support it. Decide whether EVERY factual claim in the "
    "answer is directly supported by the SOURCES.\n"
    "- If every claim is supported, respond APPROVE.\n"
    "- If any claim is unsupported, absent from the sources, or overreaches "
    "beyond them, respond REVISE and name the specific unsupported claim.\n"
    "Judge ONLY against the provided SOURCES — do not use any outside knowledge."
)


def make_critic():
    """Build the Critic model: a cheaper LLM constrained to CriticVerdict."""
    return ChatOpenAI(model=CRITIC_MODEL, temperature=0).with_structured_output(
        CriticVerdict
    )


def critique(critic_llm, answer: str, chunks: list) -> CriticVerdict:
    """Rule on whether `answer` is grounded in `chunks`, the exact retrieved
    sources. Standalone (not just a graph node) so the eval can exercise the
    Critic directly with known-grounded and known-ungrounded answers."""
    evidence = "\n\n".join(f"[{_source_label(d)}]\n{d.page_content}" for d in chunks)
    prompt = f"{CRITIC_INSTRUCTIONS}\n\nANSWER:\n{answer}\n\nSOURCES:\n{evidence}"
    return critic_llm.invoke(prompt)


class GraphState(TypedDict, total=False):
    """State passed between the Researcher and Critic nodes."""

    query: str
    answer: str
    sources: list
    tool_used: str
    confidence: Optional[float]
    schema_ok: bool
    retriever_calls: int
    retrieved: list  # the EXACT chunks the Critic verifies against
    verdict: str
    critic_reason: str
    revisions: int


@dataclass
class PipelineResult:
    """A fully reviewed answer: the Layer 1 QueryResult fields plus the verdict."""

    answer: str
    sources: list[str]
    tool_used: str
    confidence: Optional[float]
    schema_ok: bool
    retriever_calls: int
    retrieved: list
    verdict: str  # final "APPROVE" | "REVISE"
    critic_reason: str
    revisions: int  # how many revision passes were taken


@dataclass
class System:
    """The compiled two-agent graph plus the pieces the eval reuses directly."""

    graph: object
    handle: AgentHandle
    critic_llm: object


def build_system() -> System:
    """Wire the Researcher and Critic into an explicit LangGraph StateGraph."""
    handle = build_agent()
    critic_llm = make_critic()

    def researcher_node(state: GraphState) -> dict:
        # On a revision pass, fold the Critic's note into the researcher's input.
        revisions = state.get("revisions", 0)
        if state.get("verdict") == "REVISE":
            revisions += 1
            user_content = (
                f'{state["query"]}\n\n[A reviewer flagged your previous answer as not '
                f"fully supported by the indexed documents. Feedback: "
                f'{state.get("critic_reason", "")}\n'
                f"Revise it: answer using ONLY what the sources support, and drop or "
                f"qualify any claim they do not support.]"
            )
        else:
            user_content = state["query"]

        result = run_query(handle, user_content)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "tool_used": result.tool_used,
            "confidence": result.confidence,
            "schema_ok": result.schema_ok,
            "retriever_calls": result.retriever_calls,
            "retrieved": result.retrieved,
            "revisions": revisions,
        }

    def critic_node(state: GraphState) -> dict:
        chunks = state.get("retrieved", [])
        if not chunks:
            # No document evidence to check (a refusal or a pure web answer).
            return {
                "verdict": "APPROVE",
                "critic_reason": "No indexed-document claims to verify.",
            }
        verdict = critique(critic_llm, state.get("answer", ""), chunks)
        return {"verdict": verdict.verdict, "critic_reason": verdict.reason}

    def route_after_critic(state: GraphState) -> str:
        # APPROVE ends; REVISE loops back to the Researcher until the cap is hit.
        if state.get("verdict") == "APPROVE":
            return END
        if state.get("revisions", 0) >= MAX_REVISIONS:
            return END
        return "researcher"

    graph = StateGraph(GraphState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic", route_after_critic, {"researcher": "researcher", END: END}
    )
    return System(graph.compile(), handle, critic_llm)


def run_pipeline(system: System, query: str) -> PipelineResult:
    """Run a query through the full Researcher -> Critic graph."""
    final = system.graph.invoke({"query": query, "revisions": 0, "verdict": ""})
    return PipelineResult(
        answer=final.get("answer", ""),
        sources=final.get("sources", []),
        tool_used=final.get("tool_used", "none"),
        confidence=final.get("confidence"),
        schema_ok=final.get("schema_ok", False),
        retriever_calls=final.get("retriever_calls", 0),
        retrieved=final.get("retrieved", []),
        verdict=final.get("verdict", ""),
        critic_reason=final.get("critic_reason", ""),
        revisions=final.get("revisions", 0),
    )