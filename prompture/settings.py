from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    ai_provider: str = "ollama"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    claude_api_key: Optional[str] = None
    claude_model: str = "claude-3-haiku-20240307"
    hf_endpoint: Optional[str] = None
    hf_token: Optional[str] = None
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama2"
    azure_api_key: Optional[str] = None
    azure_api_endpoint: Optional[str] = None
    azure_deployment_id: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="",
    )

settings = Settings()