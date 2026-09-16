from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RESOURCES = ROOT / "resources"


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def configs_root() -> Path:
    return CONFIGS


@pytest.fixture
def resources_dir() -> Path:
    return RESOURCES
