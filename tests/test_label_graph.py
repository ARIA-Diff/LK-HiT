import numpy as np

from lkhit.data.label_graph import (
    _article_number,
    cooccurrence_pmi,
    criminal_law_groups,
    echr_groups,
    normalise_adjacency,
    structural_adjacency,
)
from lkhit.data.tasks import ECTHR_ARTICLES


def test_positive_pmi_edges_only() -> None:
    label_sets = [[0, 1], [0, 1], [0, 1], [2], [2], [0, 2]]
    weights = cooccurrence_pmi(label_sets, 3)
    assert weights.shape == (3, 3)
    assert np.allclose(np.diag(weights), 0.0)
    assert weights[0, 1] > 0
    assert weights[1, 0] == weights[0, 1]


def test_structural_adjacency_connects_same_group() -> None:
    names = ["2", "3", "14"]
    groups = {"2": "life", "3": "life", "14": "discrimination"}
    adj = structural_adjacency(names, groups)
    assert adj[0, 1] == 1.0
    assert adj[1, 0] == 1.0
    assert adj[0, 2] == 0.0


def test_normalised_adjacency_is_symmetric_and_has_self_loops() -> None:
    raw = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    normed = normalise_adjacency(raw)
    assert np.allclose(normed, normed.T)
    assert normed[0, 0] > 0


def test_echr_groups_match_shipped_sections(resources_dir) -> None:
    groups = echr_groups(ECTHR_ARTICLES + ["none"], resources_dir)
    assert groups["2"] == groups["3"]
    assert groups["8"] == groups["11"]
    assert groups["14"] != groups["P1-1"]
    assert "none" not in groups


def test_criminal_law_chapter_ranges(resources_dir) -> None:
    groups = criminal_law_groups(["264", "133-1", "234", "第382条"], resources_dir)
    assert groups["264"] == "ch5_property"
    assert groups["133-1"] == "ch2_public_security"
    assert groups["234"] == "ch4_personal_rights"
    assert groups["第382条"] == "ch8_embezzlement_bribery"


def test_article_number_parsing() -> None:
    assert _article_number("264") == 264
    assert _article_number("133-1") == 133
    assert _article_number("第234条") == 234
    assert _article_number("none") is None
