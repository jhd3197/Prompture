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

    # Google Vertex AI (Gemini + Claude via Model Garden)
    google_vertex_api_key: Optional[str] = None
    google_vertex_project_id: Optional[str] = None
    google_vertex_location: str = "us-central1"
    google_vertex_model: str = "gemini-2.5-flash"
    google_vertex_access_token: Optional[str] = None

    # Groq
    groq_api_key: Optional[str] = None
    groq_model: str = "llama2-70b-4096"

    # OpenRouter
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "openai/gpt-4o-mini"

    # Grok
    grok_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    grok_model: str = "grok-4-fast-reasoning"
    grok_video_model: str = "grok-imagine-video"

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

    # CachiBot.ai (proxy)
    cachibot_api_key: Optional[str] = None
    cachibot_endpoint: str = "https://cachibot.ai/api/v1"

    # Stability AI (image generation)
    stability_api_key: Optional[str] = None
    stability_endpoint: Optional[str] = None

    # Runway (image generation via /v1/text_to_image)
    runway_api_key: Optional[str] = None
    runway_endpoint: Optional[str] = None

    # Kling AI (image + video generation)
    kling_access_key: Optional[str] = None
    kling_secret_key: Optional[str] = None
    kling_endpoint: Optional[str] = None
    kling_image_model: str = "kling-v2-1"
    kling_video_model: str = "kling-v2-1"

    # MiniMax / Hailuo (LLM + video generation)
    minimax_api_key: Optional[str] = None
    hailuo_api_key: Optional[str] = None
    minimax_endpoint: str = "https://api.minimax.io/v1"
    minimax_model: str = "MiniMax-Text-01"
    minimax_video_model: str = "MiniMax-Hailuo-2.3"

    # Fal.ai (image + video generation aggregator)
    fal_api_key: Optional[str] = None
    fal_endpoint: Optional[str] = None
    fal_image_model: str = "fal-ai/flux/dev"
    fal_video_model: str = "fal-ai/kling-video/v2.6/pro/image-to-video"

    # Luma AI (Dream Machine — video generation)
    luma_api_key: Optional[str] = None  # nosec B105
    luma_video_model: str = "ray-2"

    # Pika Labs (video generation)
    pika_api_key: Optional[str] = None  # nosec B105
    pika_video_model: str = "pika-2.2"

    # Phase 7: Ideogram (image generation)
    ideogram_api_key: Optional[str] = None  # nosec B105
    ideogram_image_model: str = "ideogram-v3"

    # Phase 7: Black Forest Labs (BFL) — direct Flux access
    bfl_api_key: Optional[str] = None  # nosec B105
    bfl_image_model: str = "flux-pro-1.1"

    # Mistral AI
    mistral_api_key: Optional[str] = None
    mistral_model: str = "mistral-large-latest"

    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"

    # OpenAI-compatible aggregators (used via the openai_compatible driver
    # with a `profile=` arg). Each setting is consumed lazily — only the
    # profile you actually use needs its key set.
    fireworks_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    cerebras_api_key: Optional[str] = None
    sambanova_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    deepinfra_api_key: Optional[str] = None
    siliconflow_api_key: Optional[str] = None

    # Phase 2: Rerank providers
    cohere_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None

    # Phase 3: Cohere LLM + Cohere/Voyage/Jina embedding model defaults
    cohere_model: str = "command-r-plus"
    cohere_embedding_model: str = "embed-v4.0"
    voyage_embedding_model: str = "voyage-3.5"
    jina_embedding_model: str = "jina-embeddings-v3"

    # Phase 4: AWS Bedrock
    aws_access_key_id: Optional[str] = None  # nosec B105
    aws_secret_access_key: Optional[str] = None  # nosec B105
    aws_region: str = "us-east-1"
    bedrock_model: str = "anthropic.claude-3-5-haiku-20241022-v1:0"

    # ElevenLabs (audio)
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_tts_model: str = "eleven_multilingual_v2"
    elevenlabs_endpoint: str = "https://api.elevenlabs.io/v1"

    # Phase 5: Audio providers (Cartesia, Deepgram, AssemblyAI)
    cartesia_api_key: Optional[str] = None  # nosec B105
    cartesia_tts_model: str = "sonic-2"

    deepgram_api_key: Optional[str] = None  # nosec B105
    deepgram_stt_model: str = "nova-3"
    deepgram_tts_model: str = "aura-2-thalia-en"

    assemblyai_api_key: Optional[str] = None  # nosec B105
    assemblyai_stt_model: str = "universal"

    # Model rates cache
    model_rates_ttl_days: int = 7  # How often to refresh models.dev cache

    # Pricing source resolution. Controls which built-in pricing sources are
    # registered at startup. Custom sources can always be added via
    # ``prompture.infra.pricing.register_pricing_source(...)`` regardless of
    # this setting.
    #   "local_first"      — local KB first, then models.dev fallback (default)
    #   "local_only"       — only the local KB; missing models report no cost
    #   "models_dev_only"  — only models.dev (legacy behaviour)
    pricing_source: str = "local_first"

    # Usage tracking
    usage_tracking_enabled: bool = False
    usage_db_path: Optional[str] = None  # default ~/.prompture/usage/usage.db
    usage_flush_threshold: int = 10  # batch writes before flushing

    # Response cache
    cache_enabled: bool = False
    cache_backend: str = "memory"
    cache_ttl_seconds: int = 3600
    cache_memory_maxsize: int = 256
    cache_sqlite_path: Optional[str] = None
    cache_redis_url: Optional[str] = None

    # Document ingestion
    ingest_max_file_size: int = 52428800  # 50 MB
    ingest_pdf_backend: str = "pdfplumber"  # "pdfplumber" | "pypdf" | "pymupdf"
    ingest_chunk_max_chars: int = 50000
    ingest_chunk_overlap: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="",
        protected_namespaces=(),  # Allow model_* field names (e.g., model_rates_ttl_days)
    )


settings = Settings()
