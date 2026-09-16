import numpy as np

from lkhit.data.tasks import TaskSpec
from lkhit.metrics import (
    aggregate_seeds,
    decisions_from_probs,
    document_errors,
    jaccard_per_document,
    label_tiers,
    macro_f1,
    micro_f1,
    paired_bootstrap,
)


def _spec(multi_label: bool = True) -> TaskSpec:
    return TaskSpec(
        name="toy",
        multi_label=multi_label,
        language="en",
        label_names=["a", "b", "c"],
        segmentation="paragraph",
        max_segments=4,
        max_segment_tokens=8,
    )


def test_micro_macro_f1_perfect_and_empty() -> None:
    gold = np.array([[1, 0, 1], [0, 1, 0]])
    assert micro_f1(gold, gold) == 1.0
    assert macro_f1(gold, gold) == 1.0
    pred = np.zeros_like(gold)
    assert micro_f1(pred, gold) == 0.0


def test_jaccard_empty_sets_are_one() -> None:
    pred = np.zeros((2, 3), dtype=int)
    gold = np.zeros((2, 3), dtype=int)
    assert np.allclose(jaccard_per_document(pred, gold), 1.0)


def test_decisions_threshold_and_argmax() -> None:
    multi = _spec(True)
    single = _spec(False)
    probs = np.array([[0.7, 0.2, 0.6], [0.1, 0.9, 0.4]])
    assert np.array_equal(decisions_from_probs(probs, multi), np.array([[1, 0, 1], [0, 1, 0]]))
    assert decisions_from_probs(probs, single).argmax(axis=1).tolist() == [0, 1]


def test_document_errors_single_label() -> None:
    spec = _spec(False)
    pred = np.array([[1, 0, 0], [0, 1, 0]])
    gold = np.array([[1, 0, 0], [0, 0, 1]])
    assert document_errors(pred, gold, spec).tolist() == [0.0, 1.0]


def test_frequency_tiers_rank_and_count() -> None:
    counts = np.array([100, 50, 10, 80])
    rank = label_tiers(counts, {"scheme": "rank", "boundaries": [1, 3]})
    assert rank["frequent"] == [0]
    assert set(rank["medium"]) == {1, 3}
    assert rank["rare"] == [2]
    count = label_tiers(counts, {"scheme": "count", "boundaries": [20, 90]})
    assert count["frequent"] == [0]
    assert set(count["medium"]) == {1, 3}
    assert count["rare"] == [2]


def test_aggregate_seeds() -> None:
    records = [{"macro_f1": 60.0}, {"macro_f1": 70.0}]
    out = aggregate_seeds(records)
    assert out["macro_f1"]["mean"] == 65.0
    assert out["macro_f1"]["n"] == 2


def test_paired_bootstrap_detects_a_gain() -> None:
    rng = np.random.RandomState(0)
    gold = rng.randint(0, 2, size=(80, 5))
    pred_b = gold.copy()
    pred_b[:20] = 0
    pred_a = gold.copy()
    result = paired_bootstrap(pred_a, pred_b, gold, n_resamples=400, seed=0, chunk=100)
    assert result["delta_macro_f1"] > 0
    assert result["p_value"] < 0.05
