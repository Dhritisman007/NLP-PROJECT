from pathlib import Path

import pytest

from mml.config import apply_override, deep_merge, load_config


def test_default_config_loads_and_resolves_pair_and_profile():
    cfg = load_config()
    assert cfg.pair_key == "en-hi"
    assert cfg.pair["tgt_lang"] == "hin_Deva"
    assert cfg.profile["dtype"] == "float32"
    assert cfg.seed == 42


def test_second_pair_through_config_only():
    cfg = load_config(pair="en-de")
    assert cfg.pair["tgt_lang"] == "deu_Latn"
    assert cfg.pair["mt_backend"] == "nllb"


def test_profile_and_set_overrides():
    cfg = load_config(profile="colab_t4", overrides=["mt.num_beams=3", "llm.anthropic.model=foo"])
    assert cfg.profile["device"] == "cuda"
    assert cfg.get("mt.num_beams") == 3
    assert cfg.get("llm.anthropic.model") == "foo"


def test_unknown_pair_and_profile_fail_fast():
    with pytest.raises(KeyError):
        load_config(pair="en-xx")
    with pytest.raises(KeyError):
        load_config(profile="nope")


def test_bad_override_rejected():
    with pytest.raises(ValueError):
        apply_override({}, "no_equals_sign")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}} and base == {"a": {"b": 1, "c": 2}}


def test_paths_resolved_against_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert Path(cfg.raw["paths"]["data_dir"]).is_absolute()
    assert (Path(cfg.raw["paths"]["data_dir"]).parent / "configs" / "default.yaml").exists()


def test_model_ids_in_config_are_pinned_where_remote_code_is_trusted():
    cfg = load_config()
    spec = cfg.raw["mt"]["backends"]["indictrans2"]
    assert spec["trust_remote_code"] is True and len(spec["revision"]) == 40
