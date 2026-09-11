import pytest

from miniharness.config import Config, load_config
from miniharness.models import ProviderConfig


def test_config_relative_paths(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[provider]\nmodel="test"\n[runtime]\nworkspace="work"', encoding="utf-8")
    assert load_config(path).workspace == tmp_path / "work"


@pytest.mark.parametrize(
    "values", [{"context_window": 100}, {"max_output_tokens": 0}, {"model": ""}]
)
def test_invalid_budget(values):
    config = Config(provider=ProviderConfig(**({"model": "test"} | values)))
    with pytest.raises(ValueError):
        config.validate()


def test_missing_config(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_config(tmp_path / "missing.toml")
