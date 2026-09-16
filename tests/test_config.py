from pathlib import Path

import pytest

from lkhit.config import load_config, resolve_encoder_name, run_directory


EXPERIMENTS = sorted((Path(__file__).resolve().parents[1] / "configs" / "experiments").glob("*/*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_experiment_config_composes(path: Path) -> None:
    cfg = load_config(path)
    assert cfg["data"]["task"] == path.parent.name
    assert cfg["model"]["arch"]
    assert cfg["run"]["name"]
    if cfg["model"]["arch"] in ("zero_shot_llm", "tfidf_svm"):
        return
    assert "train" in cfg
    assert cfg["train"]["epochs"] >= 1
    assert cfg["train"]["grad_accumulation"] * cfg["train"]["batch_size"] >= 1


def test_dotted_override_and_run_directory() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "experiments" / "ecthr_a" / "lkhit.yaml", ["train.batch_size=2", "run.name=smoke"])
    assert cfg["train"]["batch_size"] == 2
    assert resolve_encoder_name(cfg) == "nlpaueb/legal-bert-base-uncased"
    assert run_directory(cfg, 3) == Path("runs") / "ecthr_a" / "smoke" / "seed3"


def test_ablation_overrides_label_knowledge() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "experiments"
    full = load_config(root / "ecthr_a" / "lkhit.yaml")
    ablated = load_config(root / "ecthr_a" / "lkhit_no_label_knowledge.yaml")
    assert full["model"]["label_knowledge"] == "statute"
    assert ablated["model"]["label_knowledge"] == "random"
    assert ablated["run"]["name"] == "lkhit_no_label_knowledge"


def test_cail_encoder_role() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "experiments" / "cail2018" / "lkhit.yaml")
    assert resolve_encoder_name(cfg) == "hfl/chinese-roberta-wwm-ext"
    assert cfg["data"]["multi_label"] is False
    assert cfg["loss"]["single_label"] == "class_balanced_ce"
