import json

import pytest

from stylist.cli import main

FIXTURE = "tests/fixtures/sample_500.jsonl.gz"


@pytest.fixture(scope="module")
def built(tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("cli")
    catalog = root / "catalog.parquet"
    index_dir = root / "index"
    monkeypatch_module.setenv("EMBEDDER", "hash")
    assert main(["ingest", "--raw", FIXTURE, "--out", str(catalog)]) == 0
    assert (
        main(
            [
                "build-index",
                "--catalog",
                str(catalog),
                "--index-dir",
                str(index_dir),
                "--limit",
                "200",
                "--sampling",
                "popular",
            ]
        )
        == 0
    )
    return catalog, index_dir


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_ingest_and_build_index_create_artifacts(built):
    catalog, index_dir = built
    assert catalog.exists()
    assert (index_dir / "meta.json").exists()
    assert json.loads((index_dir / "meta.json").read_text())["n_rows"] == 200


def test_recommend_json_output(built, capsys, monkeypatch_module):
    _, index_dir = built
    monkeypatch_module.setenv("INDEX_DIR", str(index_dir))
    monkeypatch_module.setenv("LLM_PROVIDER", "none")
    rc = main(["recommend", "snow boots", "--k", "2", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["slots"][0]["items"] and len(out["slots"][0]["items"]) <= 2
    assert out["llm_info"]["planner_used"] == "heuristic"


def test_recommend_pretty_output_mentions_titles(built, capsys, monkeypatch_module):
    _, index_dir = built
    monkeypatch_module.setenv("INDEX_DIR", str(index_dir))
    monkeypatch_module.setenv("LLM_PROVIDER", "none")
    assert main(["recommend", "snow boots", "--k", "1", "--max-price", "500"]) == 0
    text = capsys.readouterr().out
    assert "snow boots" in text.lower() and "1." in text


def test_recommend_fails_cleanly_without_index(capsys, monkeypatch_module, tmp_path):
    monkeypatch_module.setenv("INDEX_DIR", str(tmp_path / "missing"))
    monkeypatch_module.setenv("LLM_PROVIDER", "none")
    assert main(["recommend", "anything"]) == 2
    assert "index" in capsys.readouterr().err.lower()


def test_no_command_prints_help_and_returns_nonzero(capsys):
    assert main([]) == 1


def test_cli_config_error_is_a_clean_exit(capsys, monkeypatch_module):
    monkeypatch_module.setenv("LLM_PROVIDER", "gemini")
    assert main(["recommend", "x"]) == 2
    assert "config error" in capsys.readouterr().err


def test_cli_invalid_request_is_a_clean_exit(built, capsys, monkeypatch_module):
    _, index_dir = built
    monkeypatch_module.setenv("INDEX_DIR", str(index_dir))
    monkeypatch_module.setenv("LLM_PROVIDER", "none")
    assert main(["recommend", "x", "--min-price", "50", "--max-price", "10"]) == 2
    assert "min_price" in capsys.readouterr().err


def test_cli_unpriced_flag_is_tri_state(built, capsys, monkeypatch_module):
    _, index_dir = built
    monkeypatch_module.setenv("INDEX_DIR", str(index_dir))
    monkeypatch_module.setenv("LLM_PROVIDER", "none")
    assert main(["recommend", "swimsuit under $5", "--json"]) == 0
    auto = json.loads(capsys.readouterr().out)
    assert any(not i["price_known"] for i in auto["slots"][0]["items"])  # auto: inferred budget
    assert main(["recommend", "swimsuit under $5", "--json", "--no-include-unpriced"]) == 0
    strict = json.loads(capsys.readouterr().out)
    assert all(i["price_known"] for i in strict["slots"][0]["items"])
