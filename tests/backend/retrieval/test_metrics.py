import math

from backend.retrieval.metrics import (
    evaluate_ranking,
    hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_basic():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"c", "e"}
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert precision_at_k(retrieved, relevant, 5) == 2 / 5


def test_precision_at_k_empty_inputs():
    assert precision_at_k([], {"a"}, 3) == 0.0
    assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0


def test_recall_at_k_basic():
    retrieved = ["a", "b", "c"]
    relevant = {"c", "z"}  # z never retrieved
    assert recall_at_k(retrieved, relevant, 3) == 1 / 2


def test_recall_at_k_no_relevant_items_is_zero_not_error():
    assert recall_at_k(["a", "b"], set(), 3) == 0.0


def test_hit_rate():
    retrieved = ["a", "b", "c"]
    assert hit_rate(retrieved, {"b"}, 2) == 1.0
    assert hit_rate(retrieved, {"c"}, 2) == 0.0
    assert hit_rate(retrieved, set(), 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 1 / 2
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0
    assert reciprocal_rank(["a", "b", "c"], set()) == 0.0


def test_mean_reciprocal_rank_averages_across_queries():
    per_query_retrieved = [["a", "b"], ["x", "c"]]
    per_query_relevant = [{"a"}, {"c"}]
    # rr1 = 1/1 = 1.0, rr2 = 1/2 = 0.5 -> mean = 0.75
    assert mean_reciprocal_rank(per_query_retrieved, per_query_relevant) == 0.75


def test_mean_reciprocal_rank_empty_batch():
    assert mean_reciprocal_rank([], []) == 0.0


def test_ndcg_perfect_ranking_scores_one():
    # All relevant items ranked first, in a perfect order -> nDCG == 1.0
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    score = ndcg_at_k(retrieved, relevant, 3)
    assert math.isclose(score, 1.0, rel_tol=1e-9)


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    perfect = ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3)
    worse = ndcg_at_k(["c", "a", "b"], {"a", "b"}, 3)  # relevant items pushed later
    assert worse < perfect


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k(["a", "b"], set(), 3) == 0.0


def test_evaluate_ranking_returns_all_expected_keys():
    result = evaluate_ranking(["a", "b", "c"], {"b"}, k=3)
    assert set(result.keys()) == {"precision_at_k", "recall_at_k", "hit_rate", "reciprocal_rank", "ndcg_at_k"}
    assert all(0.0 <= v <= 1.0 for v in result.values())
