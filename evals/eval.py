"""Offline eval harness for SentryQuery.

Runs each case in qa.json through the agent and grades two things:
  1) Answer relevance — the answer contains at least one expected keyword.
  2) Tool routing — the agent used (or correctly skipped) the document
     retriever, per the case spec.

Usage (from the repo root):  python evals/eval.py

The harness itself is corpus-agnostic; the cases in qa.json are written for the
corpus currently indexed. Question wording and any figures there are
placeholders to be finalized against the real PDFs at test time — see the
_README note at the top of qa.json.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sentry_query import build_agent, extract_search_queries  # noqa: E402

QA_PATH = Path(__file__).resolve().parent / "qa.json"


def load_cases() -> list[dict]:
    """Load cases from qa.json.

    Supports either a bare list of cases or an object with a "cases" key (so the
    file can carry a human-readable _README note alongside the data).
    """
    with QA_PATH.open() as f:
        data = json.load(f)
    return data["cases"] if isinstance(data, dict) else data


def grade(case: dict, answer: str, search_queries: list[str]) -> tuple[bool, list[str]]:
    """Return (passed, reasons). A case passes only when no reason is recorded."""
    reasons: list[str] = []
    answer_lower = answer.lower()

    expected_any = [k.lower() for k in case.get("expected_keywords_any", [])]
    if expected_any and not any(k in answer_lower for k in expected_any):
        reasons.append(f"answer missing any of {expected_any}")

    if "must_use_retriever" in case:
        used = bool(search_queries)
        expected_use = bool(case["must_use_retriever"])
        if used != expected_use:
            reasons.append(
                f"retriever use mismatch: expected={expected_use}, actual={used}"
            )

    return (not reasons), reasons


def run_case(agent, case: dict) -> tuple[str, list[str]]:
    """Invoke the agent on one case; return (final_answer_text, retriever_queries)."""
    response = agent.invoke({"messages": [{"role": "user", "content": case["question"]}]})
    answer = response["messages"][-1].content
    if not isinstance(answer, str):
        answer = str(answer)
    return answer, extract_search_queries(response["messages"])


def main() -> int:
    cases = load_cases()
    agent, _ = build_agent()
    passed = 0
    print(f"Running {len(cases)} eval cases...\n")

    for i, case in enumerate(cases, 1):
        label = case.get("id", case["question"])
        try:
            answer, search_queries = run_case(agent, case)
        except Exception as exc:  # keep going so one bad case doesn't hide the rest
            print(f"[{i}] ERROR | {label}: {exc}\n")
            continue

        ok, reasons = grade(case, answer, search_queries)
        if ok:
            passed += 1

        print(f"[{i}] {'PASS' if ok else 'FAIL'} | {label}")
        print(f"    Q: {case['question']}")
        print(f"    A: {answer[:220].replace(chr(10), ' ')}")
        print(f"    retriever calls: {len(search_queries)}")
        for r in reasons:
            print(f"    !! {r}")
        print()

    print(f"=== {passed}/{len(cases)} passed ===")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())