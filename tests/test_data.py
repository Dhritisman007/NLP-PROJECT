from mml.config import load_config
from mml.data import download_all, download_flores, download_halomi, download_mqm


def test_hindi_has_no_halomi_or_mqm_and_says_so(tmp_path):
    cfg = load_config(pair="en-hi")
    assert download_halomi(cfg, tmp_path).status == "skipped"
    assert download_mqm(cfg, tmp_path).status == "skipped"


def test_flores_without_token_is_skipped_with_instructions(monkeypatch, tmp_path):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("mml.data.get_hf_token", lambda: None)
    res = download_flores(load_config(overrides=["data.flores_source=flores_plus"]), tmp_path)
    assert res.status == "skipped" and "HF_TOKEN" in res.detail


def test_dry_run_downloads_nothing(tmp_path):
    cfg = load_config(pair="en-hi", overrides=[f"paths.data_dir={tmp_path}"])
    results = download_all(cfg, ["flores"], dry_run=True)
    assert results[0].status == "dry-run" and results[0].source == "flores200"
    assert not any(tmp_path.rglob("*.jsonl"))
