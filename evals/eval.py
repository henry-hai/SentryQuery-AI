"""Offline eval harness for SentryQuery.

Runs each case in qa.json through the agent and grades:
  1) Answer relevance — the answer contains at least one expected keyword.
  2) Tool routing — the agent used (or correctly skipped) the document
     retriever, per the case spec.
  3) Schema validity — the packaged result validates against AnswerSchema
     (the Layer 1 structured-output contract).

Usage (from the repo root):  python evals/eval.py

The harness itself is corpus-agnostic; the cases in qa.json are written for the
corpus currently indexed. Question wording and any figures there are
placeholders to be finalized against the real PDFs at test time — see the
_README note at the top of qa.json.
"""
import json
import sys
from pathlib import Path

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sentry_query import build_agent, run_query  # noqa: E402
from schema import AnswerSchema  # noqa: E402

QA_PATH = Path(__file__).resolve().parent / "qa.json"


def load_cases() -> list[dict]:
    """Load cases from qa.json.

    Supports either a bare list of cases or an object with a "cases" key (so the
    file can carry a human-readable _README note alongside the data).
    """
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

    if not validates_as_schema(result):
        reasons.append("final output did not validate against AnswerSchema")

    return (not reasons), reasons


def main() -> int:
    cases = load_cases()
    handle = build_agent()
    passed = 0
    print(f"Running {len(cases)} eval cases...\n")

    for i, case in enumerate(cases, 1):
        label = case.get("id", case["question"])
        try:
            result = run_query(handle, case["question"])
        except Exception as exc:  # keep going so one bad case doesn't hide the rest
            print(f"[{i}] ERROR | {label}: {exc}\n")
            continue

        ok, reasons = grade(case, result)
        if ok:
            passed += 1

        conf = "n/a" if result.confidence is None else f"{result.confidence:.2f}"
        print(f"[{i}] {'PASS' if ok else 'FAIL'} | {label}")
        print(f"    Q: {case['question']}")
        print(f"    A: {result.answer[:220].replace(chr(10), ' ')}")
        print(
            f"    retriever calls: {result.retriever_calls} | "
            f"tool_used: {result.tool_used} | schema_ok: {result.schema_ok} | "
            f"confidence: {conf}"
        )
        for r in reasons:
            print(f"    !! {r}")
        print()

    print(f"=== {passed}/{len(cases)} passed ===")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())