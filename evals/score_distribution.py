"""Measure retrieval similarity scores, to set or re-check RETRIEVAL_MIN_SCORE.

Read-only. It embeds each query and asks the live Pinecone index for its top
RETRIEVAL_FETCH_K matches with no threshold and no filter, then prints the
scores grouped by whether the query should find evidence in the corpus. It
never writes to the index. Cost is one embedding call per query.

Usage (from the repo root):  python evals/score_distribution.py

Re-run it whenever the corpus changes, and move the threshold only if the
groups stop separating where they do now.
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from langchain_pinecone import PineconeVectorStore  # noqa: E402

from config import INDEX_NAME, RETRIEVAL_FETCH_K, RETRIEVAL_MIN_SCORE, embeddings  # noqa: E402

QA_PATH = Path(__file__).resolve().parent / "qa.json"

# Questions and claims the corpus can answer, each naming an indexed company.
IN_CORPUS = [
    "Costco's annual income grew about 10%",
    "Costco membership fee income increased",
    "Costco reported net sales of $269,912 million for fiscal year 2025.",
    "How many warehouses does Costco operate?",
    "Costco net income fiscal 2025",
    "Southwest Airlines fuel expense in 2025",
    "How many employees does Southwest Airlines have?",
    "Southwest Airlines assigned seating change",
    "Deere reported net sales and revenues of $45.7 billion in fiscal 2025, an increase over the prior year.",
    "Deere net income attributable to Deere & Company 2025",
    "Deere production and precision agriculture segment operating profit",
]

OFF_TOPIC = [
    "The moon is made of cheese.",
    "Give me a recipe for banana bread.",
    "Who won the 2022 FIFA World Cup?",
]

# Companies that are not indexed, in industries the corpus does not cover.
NOT_INDEXED_OTHER_INDUSTRY = [
    "Apple's iPhone revenue grew 5% in fiscal 2025.",
    "Tesla delivered 1.8 million vehicles in 2025.",
    "Nvidia data center revenue more than doubled.",
    "Microsoft Azure revenue grew 30%.",
]

# Companies that are not indexed but compete with one that is. Their filings
# would read like the indexed ones, so a score alone is not expected to reject
# them. They are measured to show that limit, not to set the cut.
NOT_INDEXED_SAME_INDUSTRY = [
    "Walmart's net sales grew about 5% in fiscal 2025.",
    "Delta Air Lines reported record operating revenue in 2025.",
    "Caterpillar's sales and revenues declined in 2025.",
]


def eval_groups() -> dict[str, list[str]]:
    """The eval questions, split by whether they should retrieve from the docs."""
    with QA_PATH.open() as f:
        cases = json.load(f)["cases"]
    retrieve, skip = [], []
    for case in cases:
        text = case.get("question") or case.get("claim")
        if case.get("must_use_retriever") is False:
            skip.append(text)
        elif "claim" in case and case.get("expect_reason_code") == "no_evidence":
            skip.append(text)
        else:
            retrieve.append(text)
    return {"eval, should retrieve": retrieve, "eval, should not retrieve": skip}


def main() -> int:
    vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    groups = eval_groups()
    groups["in corpus, extra probes"] = IN_CORPUS
    groups["off topic"] = OFF_TOPIC
    groups["not indexed, other industry"] = NOT_INDEXED_OTHER_INDUSTRY
    groups["not indexed, same industry"] = NOT_INDEXED_SAME_INDUSTRY

    print(f"threshold now: {RETRIEVAL_MIN_SCORE}, top {RETRIEVAL_FETCH_K} per query\n")
    for name, queries in groups.items():
        tops, alls = [], []
        print(f"== {name}")
        for q in queries:
            hits = vs.similarity_search_with_score(q, k=RETRIEVAL_FETCH_K)
            scores = [round(s, 3) for _, s in hits]
            files = sorted({os.path.basename(str(d.metadata.get("source"))) for d, _ in hits})
            tops.append(scores[0])
            alls.extend(scores)
            print(f"  top {scores[0]:.3f}  low {scores[-1]:.3f}  {', '.join(files)}  | {q[:70]}")
        print(
            f"  group: top-1 {min(tops):.3f} to {max(tops):.3f}, "
            f"all hits {min(alls):.3f} to {max(alls):.3f}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
