import pytest
from pydantic import ValidationError

from ragu.common.env import Env


_ALL_ENV_KEYS = [
    "LLM_MODEL_NAME",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "EMBEDDER_BASE_URL",
    "EMBEDDER_API_KEY",
    "EMBEDDER_MODEL_NAME",
    "RERANKER_BASE_URL",
    "RERANKER_API_KEY",
    "RERANKER_MODEL_NAME",
]


@pytest.fixture(autouse=True)
def clean_ragu_env(monkeypatch):
    for key in _ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_env_loads_required_and_optional_fields_from_os_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("EMBEDDER_BASE_URL", "https://emb.example.com/v1")
    monkeypatch.setenv("EMBEDDER_API_KEY", "emb-key")
    monkeypatch.setenv("EMBEDDER_MODEL_NAME", "text-embedding-3-large")
    monkeypatch.setenv("RERANKER_BASE_URL", "https://rerank.example.com/v1")
    monkeypatch.setenv("RERANKER_API_KEY", "rerank-key")
    monkeypatch.setenv("RERANKER_MODEL_NAME", "bge-reranker-v2-m3")

    env = Env.from_env()

    assert env.llm_model_name == "gpt-4o-mini"
    assert env.llm_base_url == "https://llm.example.com/v1"
    assert env.llm_api_key == "llm-key"
    assert env.embedder_base_url == "https://emb.example.com/v1"
    assert env.embedder_api_key == "emb-key"
    assert env.embedder_model_name == "text-embedding-3-large"
    assert env.reranker_base_url == "https://rerank.example.com/v1"
    assert env.reranker_api_key == "rerank-key"
    assert env.reranker_model_name == "bge-reranker-v2-m3"


def test_env_optional_fields_default_to_none(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")

    env = Env.from_env()

    assert env.embedder_base_url is None
    assert env.embedder_api_key is None
    assert env.embedder_model_name is None
    assert env.reranker_base_url is None
    assert env.reranker_api_key is None
    assert env.reranker_model_name is None


def test_env_from_explicit_dotenv_path(tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_MODEL_NAME=gpt-4.1-mini",
                "LLM_BASE_URL=https://llm.file/v1",
                "LLM_API_KEY=file-llm-key",
                "EMBEDDER_MODEL_NAME=embed-file-model",
                "RERANKER_MODEL_NAME=rerank-file-model",
            ]
        ),
        encoding="utf-8",
    )

    env = Env.from_env(str(env_file))

    assert env.llm_model_name == "gpt-4.1-mini"
    assert env.llm_base_url == "https://llm.file/v1"
    assert env.llm_api_key == "file-llm-key"
    assert env.embedder_model_name == "embed-file-model"
    assert env.reranker_model_name == "rerank-file-model"
    assert env.embedder_base_url is None
    assert env.reranker_base_url is None


def test_env_requires_llm_fields():
    with pytest.raises(ValidationError):
        Env.from_env()

