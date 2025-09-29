from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

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

    # Azure
    azure_api_key: Optional[str] = None
    azure_api_endpoint: Optional[str] = None
    azure_deployment_id: Optional[str] = None

    # LM Studio
    lmstudio_endpoint: str = "http://127.0.0.1:1234/v1/chat/completions"
    lmstudio_model: str = "deepseek/deepseek-r1-0528-qwen3-8b"

    # Google
    google_api_key: Optional[str] = None
    google_model: str = "gemini-1.5-pro"

    # Groq
    groq_api_key: Optional[str] = None
    groq_model: str = "llama2-70b-4096"

    # OpenRouter
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "openai/gpt-3.5-turbo"

    # Grok
    grok_api_key: Optional[str] = None
    grok_model: str = "grok-4-fast-reasoning"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="",
    )


settings = Settings()
