"""prompture - API package to convert LLM outputs into JSON + test harness."""

from dotenv import load_dotenv

from .agents import *

try:
    from .infra.tukuy_backend import TukuyLLMBackend, create_tukuy_backend
except ImportError:  # tukuy not installed
    TukuyLLMBackend = None  # type: ignore[assignment,misc]
    create_tukuy_backend = None  # type: ignore[assignment]
from . import eval as eval
from . import (
    jobs,
    mcp,
    plugins,
    rag,
    workflow,
)
from .checkpoints import (
    Checkpoint,
    CheckpointManager,
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    RunStatus,
    SQLiteCheckpointStore,
    restore_conversation,
    snapshot_conversation,
)
from .citations import (
    CITATION_INSTRUCTION,
    Citation,
    CitationTracker,
    CitedAnswer,
    Source,
    extract_citations,
)
from .cli import *
from .dataset import (
    ChatTurn,
    InstructionPair,
    QAPair,
    agenerate_qa_dataset,
    generate_qa_dataset,
    to_alpaca,
    to_jsonl,
    to_sharegpt,
)
from .drivers import *
from .drivers.media_capabilities import (
    MediaModelInfo,
    get_model_schema,
    get_models_by_modality,
    get_models_by_op,
)
from .eval import (
    EvalError,
    FaithfulnessEvaluator,
    FaithfulnessResult,
    JudgeResult,
    LLMJudge,
    PairwiseJudge,
    PairwiseResult,
    SelfConsistencyEvaluator,
    SelfConsistencyResult,
)
from .exceptions import (
    BudgetExceededError,
    ConfigurationError,
    DriverError,
    ExtractionError,
    PromptureError,
    ValidationError,
)
from .extraction import *
from .groups import *
from .infra import *
from .infra.media_pricing import estimate_cost, get_media_rate, register_media_rate
from .ingestion import *
from .integrations import *
from .jobs import JobHandle, JobResult, JobStatus, MediaAsset
from .kg import (
    Entity,
    EntityStore,
    InMemoryEntityStore,
    KnowledgeGraph,
    Mention,
    Relation,
    SQLiteEntityStore,
    extract_entities,
    extract_relations,
)
from .media import *
from .media.agent_tools import media_tool_definitions, register_media_tools
from .persistence import *
from .pipeline import *
from .refusal import (
    RefusalCategory,
    RefusalDetector,
    RefusalEvaluator,
    RefusalReport,
    RefusalResult,
    is_refusal,
)
from .security import (
    InjectionCategory,
    InjectionResult,
    PIICategory,
    PIIMatch,
    PIIRedactor,
    PromptInjectionDetector,
    RedactionResult,
    is_prompt_injection,
    redact_pii,
)
from .session_memory import (
    InMemorySessionStore,
    MemoryFact,
    MemoryKind,
    SessionMemory,
    SessionMemoryStore,
    SQLiteSessionStore,
    Summarizer,
)
from .swarm import (
    AllScheduler,
    CallableScheduler,
    Environment,
    Event,
    EventKind,
    InMemoryStore,
    Memory,
    MemoryStore,
    PriorityScheduler,
    RoundRobinScheduler,
    SampleScheduler,
    Scheduler,
    Swarm,
    SwarmAgent,
    SwarmCallbacks,
    SwarmResult,
    SwarmStep,
)
from .tools import (
    PythonSandboxTool,
    SearchResult,
    WebSearchTool,
    python_execute_tool,
    web_search_tool,
)
from .workflow import (
    ArchitectResult,
    CompiledWorkflow,
    Graph,
    GraphRunner,
    Node,
    architect,
    build_graph,
    compile_graph,
    run_graph,
)

# Tukuy type re-exports (aliased to avoid collision with Prompture names)
try:
    from tukuy import (
        AvailabilityReason as TukuyAvailabilityReason,
    )
    from tukuy import (
        Branch as TukuyBranch,
    )
    from tukuy import (
        Chain as TukuyChain,
    )
    from tukuy import (
        ConfigParam as TukuyConfigParam,
    )
    from tukuy import (
        ConfigScope as TukuyConfigScope,
    )
    from tukuy import (
        Parallel as TukuyParallel,
    )
    from tukuy import (
        PluginDiscoveryResult as TukuyPluginDiscoveryResult,
    )
    from tukuy import (
        PluginManifest as TukuyPluginManifest,
    )
    from tukuy import (
        PluginRequirements as TukuyPluginRequirements,
    )
    from tukuy import (
        RiskLevel as TukuyRiskLevel,
    )
    from tukuy import (
        SafetyPolicy as TukuySafetyPolicy,
    )
    from tukuy import (
        Skill as TukuySkill,
    )
    from tukuy import (
        SkillAvailability as TukuySkillAvailability,
    )
    from tukuy import (
        SkillContext as TukuySkillContext,
    )
    from tukuy import (
        SkillResult as TukuySkillResult,
    )
    from tukuy import (
        branch as tukuy_branch,
    )
    from tukuy import (
        discover_plugins as tukuy_discover_plugins,
    )
    from tukuy import (
        get_available_skills as tukuy_get_available_skills,
    )
    from tukuy import (
        parallel as tukuy_parallel,
    )
    from tukuy import (
        skill as tukuy_skill,
    )
    from tukuy.safety import (
        SecurityContext as TukuySecurityContext,
    )
except ImportError:
    pass

# Load environment variables from .env file
load_dotenv()

# Auto-configure cache from settings if enabled
from .infra.settings import settings as _settings

if _settings.cache_enabled:
    configure_cache(
        backend=_settings.cache_backend,
        enabled=True,
        ttl=_settings.cache_ttl_seconds,
        maxsize=_settings.cache_memory_maxsize,
        db_path=_settings.cache_sqlite_path,
        redis_url=_settings.cache_redis_url,
    )

# Auto-configure usage tracker from settings if enabled
if _settings.usage_tracking_enabled:
    configure_tracker(
        enabled=True,
        db_path=_settings.usage_db_path,
        flush_threshold=_settings.usage_flush_threshold,
    )

# runtime package version (from installed metadata)
try:
    # Python 3.8+
    from importlib.metadata import version as _get_version
except Exception:
    # older python using importlib-metadata backport (if you include it)
    from importlib_metadata import version as _get_version  # type: ignore[no-redef]

try:
    __version__ = _get_version("prompture")
except Exception:
    # fallback during local editable development
    __version__ = "0.0.0"
