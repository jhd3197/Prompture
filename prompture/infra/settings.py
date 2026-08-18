from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Provider selection
    ai_provider: str = "ollama"

    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-3.5-turbo"

    # Claude
    claude_api_key: str | None = None
    claude_model: str = "claude-3-haiku-20240307"

    # HuggingFace
    hf_endpoint: str | None = None
    hf_token: str | None = None

    # Ollama
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama2"

    # Azure (default / OpenAI backend)
    azure_api_key: str | None = None
    azure_api_endpoint: str | None = None
    azure_deployment_id: str | None = None

    # Azure - Claude backend (optional)
    azure_claude_api_key: str | None = None
    azure_claude_endpoint: str | None = None
    azure_claude_api_version: str | None = None

    # Azure - Mistral backend (optional)
    azure_mistral_api_key: str | None = None
    azure_mistral_endpoint: str | None = None
    azure_mistral_api_version: str | None = None

    # LM Studio
    lmstudio_endpoint: str = "http://127.0.0.1:1234/v1/chat/completions"
    lmstudio_model: str = "deepseek/deepseek-r1-0528-qwen3-8b"
    lmstudio_api_key: str | None = None

    # Google
    google_api_key: str | None = None
    google_model: str = "gemini-1.5-pro"

    # Google Vertex AI (Gemini + Claude via Model Garden)
    google_vertex_api_key: str | None = None
    google_vertex_project_id: str | None = None
    google_vertex_location: str = "us-central1"
    google_vertex_model: str = "gemini-2.5-flash"
    google_vertex_access_token: str | None = None

    # Groq
    groq_api_key: str | None = None
    groq_model: str = "llama2-70b-4096"

    # OpenRouter
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-4o-mini"

    # Grok
    grok_api_key: str | None = None
    xai_api_key: str | None = None
    grok_model: str = "grok-4-fast-reasoning"
    grok_video_model: str = "grok-imagine-video"

    # Moonshot AI (Kimi)
    moonshot_api_key: str | None = None
    moonshot_model: str = "kimi-k2-0905-preview"
    moonshot_endpoint: str = "https://api.moonshot.ai/v1"

    # Z.ai (Zhipu AI)
    zhipu_api_key: str | None = None
    zhipu_model: str = "glm-4.7"
    zhipu_endpoint: str = "https://api.z.ai/api/paas/v4"

    # ModelScope (Alibaba Cloud)
    modelscope_api_key: str | None = None
    modelscope_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    modelscope_endpoint: str = "https://api-inference.modelscope.cn/v1"

    # AirLLM
    airllm_model: str = "meta-llama/Llama-2-7b-hf"
    airllm_compression: str | None = None  # "4bit" or "8bit"

    # CachiBot.ai (proxy)
    cachibot_api_key: str | None = None
    cachibot_endpoint: str = "https://cachibot.ai/api/v1"

    # Stability AI (image generation)
    stability_api_key: str | None = None
    stability_endpoint: str | None = None

    # Runway (image generation via /v1/text_to_image)
    runway_api_key: str | None = None
    runway_endpoint: str | None = None

    # Kling AI (image + video generation)
    kling_access_key: str | None = None
    kling_secret_key: str | None = None
    kling_endpoint: str | None = None
    kling_image_model: str = "kling-v2-1"
    kling_video_model: str = "kling-v2-1"

    # MiniMax / Hailuo (LLM + video generation)
    minimax_api_key: str | None = None
    hailuo_api_key: str | None = None
    minimax_endpoint: str = "https://api.minimax.io/v1"
    minimax_model: str = "MiniMax-Text-01"
    minimax_video_model: str = "MiniMax-Hailuo-2.3"

    # Fal.ai (image + video generation aggregator)
    fal_api_key: str | None = None
    fal_endpoint: str | None = None
    fal_image_model: str = "fal-ai/flux/dev"
    fal_video_model: str = "fal-ai/kling-video/v2.6/pro/image-to-video"

    # Muapi.ai (multi-modal generation aggregator: image / video / edit / lipsync / audio)
    muapi_api_key: str | None = None  # nosec B105
    muapi_endpoint: str = "https://api.muapi.ai"
    muapi_image_model: str = "nano-banana"
    muapi_video_model: str = "kling-video-v2-1"
    muapi_lipsync_model: str = "infinite-talk"
    muapi_music_model: str = "suno-create-music"

    # Luma AI (Dream Machine — video generation)
    luma_api_key: str | None = None  # nosec B105
    luma_video_model: str = "ray-2"

    # Pika Labs (video generation)
    pika_api_key: str | None = None  # nosec B105
    pika_video_model: str = "pika-2.2"

    # Phase 7: Ideogram (image generation)
    ideogram_api_key: str | None = None  # nosec B105
    ideogram_image_model: str = "ideogram-v3"

    # Phase 7: Black Forest Labs (BFL) — direct Flux access
    bfl_api_key: str | None = None  # nosec B105
    bfl_image_model: str = "flux-pro-1.1"

    # Mistral AI
    mistral_api_key: str | None = None
    mistral_model: str = "mistral-large-latest"

    # DeepSeek
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"

    # OpenAI-compatible aggregators (used via the openai_compatible driver
    # with a `profile=` arg). Each setting is consumed lazily — only the
    # profile you actually use needs its key set.
    fireworks_api_key: str | None = None
    together_api_key: str | None = None
    cerebras_api_key: str | None = None
    sambanova_api_key: str | None = None
    perplexity_api_key: str | None = None
    nvidia_api_key: str | None = None
    deepinfra_api_key: str | None = None
    siliconflow_api_key: str | None = None

    # Phase 2: Rerank providers
    cohere_api_key: str | None = None
    voyage_api_key: str | None = None
    jina_api_key: str | None = None

    # Phase 3: Cohere LLM + Cohere/Voyage/Jina embedding model defaults
    cohere_model: str = "command-r-plus"
    cohere_embedding_model: str = "embed-v4.0"
    voyage_embedding_model: str = "voyage-3.5"
    jina_embedding_model: str = "jina-embeddings-v3"

    # Phase 4: AWS Bedrock
    aws_access_key_id: str | None = None  # nosec B105
    aws_secret_access_key: str | None = None  # nosec B105
    aws_region: str = "us-east-1"
    bedrock_model: str = "anthropic.claude-3-5-haiku-20241022-v1:0"

    # ElevenLabs (audio)
    elevenlabs_api_key: str | None = None
    elevenlabs_tts_model: str = "eleven_multilingual_v2"
    elevenlabs_endpoint: str = "https://api.elevenlabs.io/v1"

    # Phase 5: Audio providers (Cartesia, Deepgram, AssemblyAI)
    cartesia_api_key: str | None = None  # nosec B105
    cartesia_tts_model: str = "sonic-2"

    deepgram_api_key: str | None = None  # nosec B105
    deepgram_stt_model: str = "nova-3"
    deepgram_tts_model: str = "aura-2-thalia-en"

    assemblyai_api_key: str | None = None  # nosec B105
    assemblyai_stt_model: str = "universal"

    # Phase 8: Nomic, Mixedbread, GitHub Models
    nomic_api_key: str | None = None  # nosec B105
    nomic_embedding_model: str = "nomic-embed-text-v1.5"

    mixedbread_api_key: str | None = None  # nosec B105
    mxbai_embedding_model: str = "mxbai-embed-large-v1"
    mxbai_rerank_model: str = "mxbai-rerank-large-v1"

    # GitHub Models — used through the openai_compatible driver
    # (profile=github_models). A GitHub Personal Access Token.
    github_token: str | None = None  # nosec B105

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
    usage_db_path: str | None = None  # default ~/.prompture/usage/usage.db
    usage_flush_threshold: int = 10  # batch writes before flushing

    # Response cache
    cache_enabled: bool = False
    cache_backend: str = "memory"
    cache_ttl_seconds: int = 3600
    cache_memory_maxsize: int = 256
    cache_sqlite_path: str | None = None
    cache_redis_url: str | None = None

    # Web search providers (used by prompture.tools.WebSearchTool)
    tavily_api_key: str | None = None
    serper_api_key: str | None = None
    brave_search_api_key: str | None = None
    searxng_endpoint: str | None = None

    # Coding-agent CLI binary overrides (env var: CODING_AGENT_BIN_<UPPER>)
    coding_agent_bin_claude: str | None = None
    coding_agent_bin_codex: str | None = None
    coding_agent_bin_gemini: str | None = None
    coding_agent_bin_qwen: str | None = None
    coding_agent_bin_aider: str | None = None
    coding_agent_bin_opencode: str | None = None
    coding_agent_bin_cursor_agent: str | None = None
    coding_agent_bin_crush: str | None = None

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
