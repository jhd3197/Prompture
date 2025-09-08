from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # Fields automatically map to UPPER_CASE environment variables
    default_provider: str = "mock"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    hf_endpoint: Optional[str] = None
    hf_token: Optional[str] = None
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama2"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()