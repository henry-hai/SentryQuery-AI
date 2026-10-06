"""Offline checks for retrieval ranking metrics and the saved snapshot."""

from pathlib import Path

import retrieval_eval


SOURCE = "docs/costco-10k-2025.pdf"
OTHER_SOURCE = "docs/deere-10k-2025.pdf"
CASE = {"id": "sample", "source": SOURCE, "pages": [3, 5]}


def hit(source: str, page: int, score: float = 0.8) -> dict:
    """Make a ranked hit without a document or index client."""
    return {"source": source, "page": page, "score": score}


def test_first_hit_at_rank_one():
    ranking = [hit(SOURCE, 5), hit(OTHER_SOURCE, 3)]
    assert retrieval_eval.first_hit_rank(CASE, ranking, 5) == 1


def test_first_hit_at_rank_three():
    ranking = [hit(SOURCE, 2), hit(OTHER_SOURCE, 3), hit(SOURCE, 3)]
    assert retrieval_eval.first_hit_rank(CASE, ranking, 5) == 3
    assert retrieval_eval.first_hit_rank(CASE, ranking, 2) is None


def test_miss_and_wrong_source_with_right_page():
    assert retrieval_eval.first_hit_rank(CASE, [hit(SOURCE, 2)], 5) is None
    assert retrieval_eval.first_hit_rank(CASE, [hit(OTHER_SOURCE, 3)], 5) is None


def test_hit_rate_and_mrr():
    cases = [{**CASE, "id": case_id} for case_id in ("one", "three", "miss", "wrong")]
    saved = [
        {"id": "one", "raw": [hit(SOURCE, 3, 0.9)]},
        {"id": "three", "raw": [hit(SOURCE, 2), hit(OTHER_SOURCE, 5), hit(SOURCE, 5)]},
        {"id": "miss", "raw": [hit(SOURCE, 2)]},
        {"id": "wrong", "raw": [hit(OTHER_SOURCE, 3)]},
    ]
    result = retrieval_eval.summarize(cases, saved, "raw", 5)
    assert result["hit_rate"] == 0.5
    assert result["mrr"] == (1 + 1 / 3) / 4
    assert [row["rank"] for row in result["rows"]] == [1, 3, None, None]


def test_committed_snapshot_ids_are_known_cases():
    eval_dir = Path(retrieval_eval.__file__).resolve().parent
    cases = retrieval_eval.load_json(eval_dir / "retrieval_qa.json")["cases"]
    snapshot = retrieval_eval.load_json(eval_dir / "retrieval_snapshot.json")
    known_ids = {case["id"] for case in cases}
    assert isinstance(snapshot["cases"], list)
    assert all(saved["id"] in known_ids for saved in snapshot["cases"])
