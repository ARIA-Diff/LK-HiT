from lkhit.data.descriptions import ECTHR_NONE_DESCRIPTIONS, echr_descriptions
from lkhit.data.tasks import TaskSpec


def _ecthr_spec(name: str) -> TaskSpec:
    return TaskSpec(
        name=name,
        multi_label=True,
        language="en",
        label_names=["2", "3", "5", "6", "8", "9", "10", "11", "14", "P1-1", "none"],
        none_label="none",
        segmentation="paragraph",
        max_segments=64,
        max_segment_tokens=128,
    )


def test_echr_descriptions_cover_every_label(resources_dir) -> None:
    spec = _ecthr_spec("ecthr_a")
    mapping = echr_descriptions(spec, resources_dir)
    assert set(mapping) == set(spec.label_names)
    assert mapping["none"] == ECTHR_NONE_DESCRIPTIONS["ecthr_a"]
    assert "Right to life" in mapping["2"]
    assert "Protocol" in mapping["P1-1"] or "possessions" in mapping["P1-1"]


def test_ecthr_b_none_sentence_differs(resources_dir) -> None:
    a = echr_descriptions(_ecthr_spec("ecthr_a"), resources_dir)
    b = echr_descriptions(_ecthr_spec("ecthr_b"), resources_dir)
    assert a["none"] != b["none"]
    assert a["6"] == b["6"]
