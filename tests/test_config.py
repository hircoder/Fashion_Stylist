from pathlib import Path

import pytest

from stylist.config import ConfigError, Settings


def test_defaults_without_env_use_data_dir_and_no_llm():
    s = Settings.from_env({})
    assert s.llm_provider == "none"
    assert s.llm_model is None
    assert s.index_dir == Path("data/index")
    assert s.raw_path == Path("data/raw/meta_Amazon_Fashion.jsonl.gz")
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.request_deadline_s == 25.0


def test_provider_precedence_anthropic_wins_when_both_keys_present():
    s = Settings.from_env({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "b"})
    assert s.llm_provider == "anthropic"
    assert s.llm_model == "claude-opus-5"


def test_openai_is_picked_when_only_openai_key_present():
    s = Settings.from_env({"OPENAI_API_KEY": "b"})
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-5-mini"


def test_explicit_model_overrides_default():
    s = Settings.from_env({"OPENAI_API_KEY": "b", "LLM_MODEL": "gpt-5-nano"})
    assert s.llm_model == "gpt-5-nano"


def test_explicit_provider_without_key_is_a_config_error():
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_PROVIDER": "anthropic"})


def test_unknown_provider_is_a_config_error():
    with pytest.raises(ConfigError):
        Settings.from_env({"LLM_PROVIDER": "gemini", "OPENAI_API_KEY": "x"})


def test_provider_none_ignores_keys():
    s = Settings.from_env({"LLM_PROVIDER": "none", "OPENAI_API_KEY": "x"})
    assert s.llm_provider == "none"


def test_numeric_knobs_parse_from_strings():
    s = Settings.from_env({"REQUEST_DEADLINE_S": "7.5", "TOP_N_PER_CHANNEL": "40"})
    assert s.request_deadline_s == 7.5
    assert s.top_n_per_channel == 40


def test_bad_number_is_a_config_error():
    with pytest.raises(ConfigError):
        Settings.from_env({"TOP_N_PER_CHANNEL": "lots"})


def test_data_dir_moves_all_paths():
    s = Settings.from_env({"DATA_DIR": "/tmp/x"})
    assert s.index_dir == Path("/tmp/x/index")
    assert s.processed_path == Path("/tmp/x/processed/catalog.parquet")


def test_embedding_name_follows_embedder_choice():
    assert Settings.from_env({}).embedding_name == "BAAI/bge-small-en-v1.5"
    assert Settings.from_env({"EMBEDDER": "hash"}).embedding_name == "hash"
