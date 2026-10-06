"""Measure whether the retriever finds verified filing pages.

Live mode queries the index and records both the chunks returned by
agent.retrieve and the unthresholded top eight. Offline mode grades the saved
snapshot without loading clients or credentials.

Run from the repo root with ``python evals/retrieval_eval.py`` or add
``--offline`` to grade the saved snapshot.
"""

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
QA_PATH = EVAL_DIR / "retrieval_qa.json"
SNAPSHOT_PATH = EVAL_DIR / "retrieval_snapshot.json"


def first_hit_rank(case: dict, ranking: list[dict], limit: int) -> int | None:
    """Return the first matching rank within limit, or None for a miss."""
    for rank, hit in enumerate(ranking[:limit], start=1):
        if hit["source"] == case["source"] and hit["page"] in case["pages"]:
            return rank
    return None


def summarize(
    cases: list[dict], snapshot_cases: list[dict], ranking_key: str, limit: int
) -> dict:
    """Return per-case ranks and aggregate hit rate and reciprocal rank."""
    expected = {case["id"]: case for case in cases}
    rows = []
    for saved in snapshot_cases:
        case = expected[saved["id"]]
        ranking = saved[ranking_key]
        rank = first_hit_rank(case, ranking, limit)
        rows.append({
            "id": saved["id"],
            "rank": rank,
            "top_score": ranking[0]["score"] if ranking else None,
        })
    count = len(rows)
    return {
        "rows": rows,
        "hit_rate": sum(row["rank"] is not None for row in rows) / count if count else None,
        "mrr": sum(1 / row["rank"] for row in rows if row["rank"] is not None) / count
        if count else None,
    }


def load_json(path: Path) -> dict:
    """Load a JSON object from disk."""
    with path.open() as file:
        return json.load(file)


def metadata_row(metadata: dict, score: float) -> dict:
    """Keep only the source, PDF page index, and similarity score."""
    page = metadata.get("page")
    return {
        "source": str(metadata.get("source", "")),
        "page": int(float(page)) if page is not None else None,
        "score": float(score),
    }


def capture_live(cases: list[dict]) -> list[dict]:
    """Query the existing index without writing to it."""
    repo_root = EVAL_DIR.parent
    sys.path.insert(0, str(repo_root))

    from langchain_pinecone import PineconeVectorStore

    from agent import retrieve
    from companies import companies_named, source_filter
    from config import INDEX_NAME, RETRIEVAL_FETCH_K, embeddings

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    saved = []
    for case in cases:
        question = case["question"]
        scope = companies_named(question)
        filtered = retrieve(vectorstore, question, scope=scope)
        raw = vectorstore.similarity_search_with_score(
            question, k=RETRIEVAL_FETCH_K, filter=source_filter(scope)
        )
        saved.append({
            "id": case["id"],
            "thresholded": [
                metadata_row(doc.metadata, doc.metadata["score"]) for doc in filtered
            ],
            "raw": [
                metadata_row(doc.metadata, score)
                for doc, score in sorted(raw, key=lambda hit: hit[1], reverse=True)
            ],
        })
    return saved


def print_report(cases: list[dict], snapshot_cases: list[dict]) -> None:
    """Print the two aggregate scores and a rank and score for each case."""
    thresholded = summarize(cases, snapshot_cases, "thresholded", 5)
    raw = summarize(cases, snapshot_cases, "raw", 8)

    def rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    print(f"Cases: {len(snapshot_cases)}/{len(cases)}")
    print(f"Thresholded hit@5: {rate(thresholded['hit_rate'])}")
    print(f"Thresholded MRR@5: {rate(thresholded['mrr'])}")
    print(f"Raw hit@8: {rate(raw['hit_rate'])}")
    print(f"Raw MRR@8: {rate(raw['mrr'])}")
    print(f"{'id':<31} {'hit@5':>6} {'top@5':>7} {'hit@8':>6} {'top@8':>7}")
    for filtered_row, raw_row in zip(thresholded["rows"], raw["rows"]):
        def rank_text(row: dict) -> str:
            return str(row["rank"]) if row["rank"] is not None else "miss"

        def score_text(row: dict) -> str:
            score = row["top_score"]
            return f"{score:.4f}" if score is not None else "n/a"

        print(
            f"{filtered_row['id']:<31} {rank_text(filtered_row):>6} "
            f"{score_text(filtered_row):>7} {rank_text(raw_row):>6} "
            f"{score_text(raw_row):>7}"
        )


def main() -> int:
    """Capture a live snapshot or grade the existing one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="grade the saved snapshot")
    args = parser.parse_args()
    cases = load_json(QA_PATH)["cases"]
    if args.offline:
        snapshot_cases = load_json(SNAPSHOT_PATH)["cases"]
    else:
        snapshot_cases = capture_live(cases)
        SNAPSHOT_PATH.write_text(json.dumps({"cases": snapshot_cases}, indent=2) + "\n")
    print_report(cases, snapshot_cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
