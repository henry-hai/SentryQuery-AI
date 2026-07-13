"""Offline eval harness for SentryQuery.

Runs each case in qa.json through the full Researcher -> Critic graph and grades:
  1) Answer relevance — the answer contains at least one expected keyword.
  2) Tool routing — the agent used (or correctly skipped) the document
     retriever, per the case spec.
  3) Schema validity — the packaged result validates against AnswerSchema.
  4) Critic verdict — where a case specifies expect_verdict, the Critic's ruling
     matches (grounded doc answers should be APPROVE).

Then runs two DIRECT Critic checks (independent of the researcher) that prove the
Critic does real work: it must REVISE a deliberately ungrounded answer and
APPROVE a grounded one, judged against the same retrieved chunks. The REVISE
check is the case that "fails without the Critic and passes with it".

Usage (from the repo root):  python evals/eval.py

The harness itself is corpus-agnostic; the cases in qa.json are written for the
corpus currently indexed — see the _README note at the top of qa.json.
"""
import json
import sys
from pathlib import Path

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from graph import build_system, run_pipeline, critique  # noqa: E402
from schema import AnswerSchema  # noqa: E402
from observability import trace_status  # noqa: E402

QA_PATH = Path(__file__).resolve().parent / "qa.json"


def load_cases() -> list[dict]:
    """Load cases from qa.json (bare list, or an object with a "cases" key)."""
    with QA_PATH.open() as f:
        data = json.load(f)
    return data["cases"] if isinstance(data, dict) else data


def validates_as_schema(result) -> bool:
    """True if the packaged result reconstructs a valid AnswerSchema.

    A fallback result has confidence=None, which fails AnswerSchema (confidence
    is a required 0-1 float) — exactly the signal we want the harness to catch.
    """
    if result.confidence is None:
        return False
    try:
        AnswerSchema(
            answer=result.answer,
            sources=result.sources,
            tool_used=result.tool_used,
            confidence=result.confidence,
        )
        return True
    except ValidationError:
        return False


def grade(case: dict, result) -> tuple[bool, list[str]]:
    """Return (passed, reasons). A case passes only when no reason is recorded."""
    reasons: list[str] = []
    answer_lower = result.answer.lower()

    expected_any = [k.lower() for k in case.get("expected_keywords_any", [])]
    if expected_any and not any(k in answer_lower for k in expected_any):
        reasons.append(f"answer missing any of {expected_any}")

    if "must_use_retriever" in case:
        used = result.retriever_calls > 0
        expected_use = bool(case["must_use_retriever"])
        if used != expected_use:
            reasons.append(
                f"retriever use mismatch: expected={expected_use}, actual={used}"
            )

    if "expect_verdict" in case and result.verdict != case["expect_verdict"]:
        reasons.append(
            f"verdict mismatch: expected={case['expect_verdict']}, actual={result.verdict}"
        )

    if not validates_as_schema(result):
        reasons.append("final output did not validate against AnswerSchema")

    return (not reasons), reasons


def critic_checks(system) -> tuple[int, int]:
    """Two direct Critic tests against real retrieved chunks.

    Proves the Critic does real work independently of the researcher: it must
    REVISE an ungrounded answer and APPROVE a grounded (verbatim) one.
    """
    chunks = system.handle.vectorstore.similarity_search(
        "total net sales for the fiscal year", k=3
    )
    passed = 0

    # 1) Fabricated, clearly ungrounded answer against real sources -> REVISE.
    #    Without a Critic this hallucination would pass straight through.
    ungrounded = (
        "The company reported total net sales of $2 trillion this year, and its "
        "CEO is a professional boxer."
    )
    v1 = critique(system.critic_llm, ungrounded, chunks)
    ok1 = v1.verdict == "REVISE"
    passed += ok1
    print(f"[critic-1] {'PASS' if ok1 else 'FAIL'} | ungrounded answer -> expect REVISE")
    print(f"    verdict: {v1.verdict} | reason: {v1.reason[:150]}")
    print()

    # 2) Verbatim source text as the answer -> trivially grounded -> APPROVE.
    grounded = chunks[0].page_content[:300]
    v2 = critique(system.critic_llm, grounded, chunks)
    ok2 = v2.verdict == "APPROVE"
    passed += ok2
    print(f"[critic-2] {'PASS' if ok2 else 'FAIL'} | grounded (verbatim) answer -> expect APPROVE")
    print(f"    verdict: {v2.verdict} | reason: {v2.reason[:150]}")
    print()

    return passed, 2


def main() -> int:
    cases = load_cases()
    system = build_system()
    passed = 0
    total = len(cases)
    print(trace_status())
    print(f"Running {len(cases)} eval cases through the Researcher -> Critic graph...\n")

    for i, case in enumerate(cases, 1):
        label = case.get("id", case["question"])
        try:
            result = run_pipeline(system, case["question"])
        except Exception as exc:  # keep going so one bad case doesn't hide the rest
            print(f"[{i}] ERROR | {label}: {exc}\n")
            continue

        ok, reasons = grade(case, result)
        if ok:
            passed += 1

        conf = "n/a" if result.confidence is None else f"{result.confidence:.2f}"
        print(f"[{i}] {'PASS' if ok else 'FAIL'} | {label}")
        print(f"    Q: {case['question']}")
        print(f"    A: {result.answer[:200].replace(chr(10), ' ')}")
        print(
            f"    retriever calls: {result.retriever_calls} | tool_used: {result.tool_used} | "
            f"schema_ok: {result.schema_ok} | confidence: {conf}"
        )
        print(
            f"    verdict: {result.verdict} | revisions: {result.revisions}"
        )
        for r in reasons:
            print(f"    !! {r}")
        print()

    print("--- direct Critic checks ---\n")
    cp, ct = critic_checks(system)
    passed += cp
    total += ct

    print(f"=== {passed}/{total} passed ===")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())