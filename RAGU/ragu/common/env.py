from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    """
    Load runtime configuration from environment variables.

    :param llm_model_name: Model name for the LLM backend.
    :param llm_base_url: Base URL for the LLM API.
    :param llm_api_key: API key for the LLM API.
    :param embedder_base_url: Optional base URL for the embedder API.
    :param embedder_api_key: Optional API key for the embedder API.
    :param embedder_model_name: Optional embedder model name.
    :param reranker_base_url: Optional base URL for the reranker API.
    :param reranker_api_key: Optional API key for the reranker API.
    :param reranker_model_name: Optional reranker model name.
    """
    llm_model_name: str
    llm_base_url: str
    llm_api_key: str

    embedder_base_url: str | None = None
    embedder_api_key: str | None = None
    embedder_model_name: str | None = None

    reranker_base_url: str | None = None
    reranker_api_key: str | None = None
    reranker_model_name: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @classmethod
    def from_env(cls, env_path: str | None = None) -> "Env":
        """
        Create configuration from environment variables and optional `.env` path.

        :param env_path: Optional path to a `.env` file. Uses default `.env` when omitted.
        :return: Loaded configuration object.
        """
        return cls(_env_file=env_path) if env_path else cls()
