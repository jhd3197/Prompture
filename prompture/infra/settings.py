from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Provider selection
    ai_provider: str = "ollama"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"

    # Claude
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-3-haiku-20240307"

    # HuggingFace
    hf_endpoint: Optional[str] = None
    hf_token: Optional[str] = None

    # Ollama
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama2"

    # Azure (default / OpenAI backend)
    azure_api_key: Optional[str] = None
    azure_api_endpoint: Optional[str] = None
    azure_deployment_id: Optional[str] = None

    # Azure - Claude backend (optional)
    azure_claude_api_key: Optional[str] = None
    azure_claude_endpoint: Optional[str] = None
    azure_claude_api_version: Optional[str] = None

    # Azure - Mistral backend (optional)
    azure_mistral_api_key: Optional[str] = None
    azure_mistral_endpoint: Optional[str] = None
    azure_mistral_api_version: Optional[str] = None

    # LM Studio
    lmstudio_endpoint: str = "http://127.0.0.1:1234/v1/chat/completions"
    lmstudio_model: str = "deepseek/deepseek-r1-0528-qwen3-8b"
    lmstudio_api_key: Optional[str] = None

    # Google
    google_api_key: Optional[str] = None
    google_model: str = "gemini-1.5-pro"

    # Groq
    groq_api_key: Optional[str] = None
    groq_model: str = "llama2-70b-4096"

    # OpenRouter
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "openai/gpt-4o-mini"

    # Grok
    grok_api_key: Optional[str] = None
    grok_model: str = "grok-4-fast-reasoning"

    # Moonshot AI (Kimi)
    moonshot_api_key: Optional[str] = None
    moonshot_model: str = "kimi-k2-0905-preview"
    moonshot_endpoint: str = "https://api.moonshot.ai/v1"

    # Z.ai (Zhipu AI)
    zhipu_api_key: Optional[str] = None
    zhipu_model: str = "glm-4.7"
    zhipu_endpoint: str = "https://api.z.ai/api/paas/v4"

    # ModelScope (Alibaba Cloud)
    modelscope_api_key: Optional[str] = None
    modelscope_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    modelscope_endpoint: str = "https://api-inference.modelscope.cn/v1"

    # AirLLM
    airllm_model: str = "meta-llama/Llama-2-7b-hf"
    airllm_compression: Optional[str] = None  # "4bit" or "8bit"

    # Model rates cache
    model_rates_ttl_days: int = 7  # How often to refresh models.dev cache

    # Response cache
    cache_enabled: bool = False
    cache_backend: str = "memory"
    cache_ttl_seconds: int = 3600
    cache_memory_maxsize: int = 256
    cache_sqlite_path: Optional[str] = None
    cache_redis_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="",
        protected_namespaces=(),  # Allow model_* field names (e.g., model_rates_ttl_days)
    )


settings = Settings()
