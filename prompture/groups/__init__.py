"""Multi-agent groups, orchestration, and consensus."""

from .async_groups import (
    AsyncLoopGroup,
    AsyncRouterAgent,
    AsyncSequentialGroup,
    ParallelGroup,
)
from .consensus import (
    ConsensusResult,
    ConsensusStrategy,
    ModelVote,
    extract_with_consensus,
    extract_with_consensus_async,
)
from .debate import (
    AsyncDebateGroup,
    DebateConfig,
    DebateEntry,
    DebateGroup,
    DebateResult,
)
from .fusion import AsyncFusionDriver, FusionDriver, FusionSpec, get_fusion, list_fusions, register_fusion
from .groups import (
    GroupAsAgent,
    LoopGroup,
    RouterAgent,
    RoutingStrategy,
    SequentialGroup,
)
from .types import (
    AgentError,
    ErrorPolicy,
    GroupCallbacks,
    GroupResult,
    GroupStep,
)

__all__ = [
    "AgentError",
    "AsyncDebateGroup",
    "AsyncFusionDriver",
    "AsyncLoopGroup",
    "AsyncRouterAgent",
    "AsyncSequentialGroup",
    "ConsensusResult",
    "ConsensusStrategy",
    "DebateConfig",
    "DebateEntry",
    "DebateGroup",
    "DebateResult",
    "ErrorPolicy",
    "FusionDriver",
    "FusionSpec",
    "GroupAsAgent",
    "GroupCallbacks",
    "GroupResult",
    "GroupStep",
    "LoopGroup",
    "ModelVote",
    "ParallelGroup",
    "RouterAgent",
    "RoutingStrategy",
    "SequentialGroup",
    "extract_with_consensus",
    "extract_with_consensus_async",
    "get_fusion",
    "list_fusions",
    "register_fusion",
]
