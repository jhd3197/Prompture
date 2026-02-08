"""prompture - API package to convert LLM outputs into JSON + test harness."""

from dotenv import load_dotenv

from .agents import *
from .analysis import *
from .cli import *
from .drivers import *
from .extraction import *
from .groups import *
from .infra import *
from .integrations import *
from .media import *
from .persistence import *
from .pipeline import *
from .sandbox import *

# Tukuy type re-exports (aliased to avoid collision with Prompture names)
try:
    from tukuy import (
        Branch as TukuyBranch,
    )
    from tukuy import (
        Chain as TukuyChain,
    )
    from tukuy import (
        Parallel as TukuyParallel,
    )
    from tukuy import (
        SafetyPolicy as TukuySafetyPolicy,
    )
    from tukuy import (
        Skill as TukuySkill,
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
        parallel as tukuy_parallel,
    )
    from tukuy import (
        skill as tukuy_skill,
    )
    from tukuy.safety import (
        SecurityContext as TukuySecurityContext,
    )
    from tukuy import (
        RiskLevel as TukuyRiskLevel,
    )
    from tukuy import (
        ConfigScope as TukuyConfigScope,
    )
    from tukuy import (
        ConfigParam as TukuyConfigParam,
    )
    from tukuy import (
        PluginManifest as TukuyPluginManifest,
    )
    from tukuy import (
        PluginRequirements as TukuyPluginRequirements,
    )
    from tukuy import (
        AvailabilityReason as TukuyAvailabilityReason,
    )
    from tukuy import (
        SkillAvailability as TukuySkillAvailability,
    )
    from tukuy import (
        PluginDiscoveryResult as TukuyPluginDiscoveryResult,
    )
    from tukuy import (
        get_available_skills as tukuy_get_available_skills,
    )
    from tukuy import (
        discover_plugins as tukuy_discover_plugins,
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

# runtime package version (from installed metadata)
try:
    # Python 3.8+
    from importlib.metadata import version as _get_version
except Exception:
    # older python using importlib-metadata backport (if you include it)
    from importlib_metadata import version as _get_version

try:
    __version__ = _get_version("prompture")
except Exception:
    # fallback during local editable development
    __version__ = "0.0.0"
