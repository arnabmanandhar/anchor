import math

from anchor.retrieval_metrics import recall_at_k, mrr, ndcg_at_k, map_score


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_respects_cutoff():
    # relevant doc is at rank 4, k=3 should not find it
    assert recall_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0


def test_mrr_first_hit_at_rank_2():
    assert mrr(["x", "a", "y"], {"a"}) == 0.5


def test_mrr_no_hit():
    assert mrr(["x", "y", "z"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_worse_ranking_is_less_than_one():
    perfect = ndcg_at_k(["a", "b"], {"a", "b"}, k=2)
    worse = ndcg_at_k(["b", "a"], {"a"}, k=2)  # relevant doc pushed to rank 2
    assert worse < perfect


def test_map_score_single_relevant_at_rank_1():
    assert map_score(["a", "x", "y"], {"a"}) == 1.0


def test_map_score_averages_precision_at_each_hit():
    # relevant = {a, b}; a at rank 1 (p=1/1), b at rank 3 (p=2/3)
    expected = (1 / 1 + 2 / 3) / 2
    assert math.isclose(map_score(["a", "x", "b"], {"a", "b"}), expected)


def test_empty_relevant_set_returns_zero_not_error():
    assert recall_at_k(["a"], set(), k=3) == 0.0
    assert mrr(["a"], set()) == 0.0
    assert ndcg_at_k(["a"], set(), k=3) == 0.0
    assert map_score(["a"], set()) == 0.0


def test_ndcg_hand_computed_value():
    # relevant {a, b}; ranking x, a, b at k=3
    # DCG  = 0 + 1/log2(3) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert math.isclose(ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3), dcg / idcg)


def test_ndcg_ideal_is_capped_at_k_when_more_relevant_than_k():
    # 3 relevant docs, k=2, both slots hit -> perfect at this cutoff
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=2) == 1.0


def test_recall_is_order_insensitive_within_k_but_mrr_is_not():
    assert recall_at_k(["x", "a"], {"a"}, k=2) == recall_at_k(["a", "x"], {"a"}, k=2)
    assert mrr(["x", "a"], {"a"}) < mrr(["a", "x"], {"a"})


def test_map_penalises_missing_relevant_docs():
    # only one of two relevant docs ever retrieved -> AP divided by 2
    assert map_score(["a", "x"], {"a", "b"}) == 0.5
