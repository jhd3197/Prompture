"""High-level checkpointing API.

The intent is non-invasive: do not modify :class:`Agent` or
:class:`Conversation`; instead provide free functions and a manager
that callers compose with the existing hook points
(:attr:`Conversation.before_turn`, :attr:`AgentCallbacks.on_step`,
``run_id`` argument from the caller).
"""

from __future__ import annotations

import contextlib
import copy
import uuid
from typing import TYPE_CHECKING, Any

from .stores import CheckpointStore, InMemoryCheckpointStore
from .types import Checkpoint, RunStatus

if TYPE_CHECKING:
    from ..agents.agent import Agent
    from ..agents.conversation import Conversation


# ----------------------------------------------------------------------
# Free functions — snapshot / restore
# ----------------------------------------------------------------------


def snapshot_conversation(
    conv: Conversation,
    *,
    run_id: str | None = None,
    iteration: int = 0,
    prompt: str = "",
    status: str = RunStatus.running.value,
    agent_name: str | None = None,
    partial_output: str = "",
    metadata: dict[str, Any] | None = None,
) -> Checkpoint:
    """Capture a :class:`Conversation` as a :class:`Checkpoint`.

    The system prompt, message history, and per-conversation usage
    counters are recorded.  Free-form *metadata* is forwarded verbatim
    so callers can stash extra state (todos, sub-agent specs,
    in-flight tool results).

    Args:
        conv: The conversation to snapshot.
        run_id: Override the run identifier.  Defaults to a fresh
            UUID4 hex string.
        iteration: Current iteration/turn index.
        prompt: The user prompt that launched the run (or the most
            recent prompt being processed).
        status: Lifecycle state — see :class:`RunStatus`.
        agent_name: Optional name of the owning agent.
        partial_output: Any partial generation text worth persisting.
        metadata: Free-form payload merged into the snapshot.
    """
    model = getattr(conv, "model_name", None) or getattr(getattr(conv, "_driver", None), "model", None)
    system_prompt = getattr(conv, "system_prompt", None)
    return Checkpoint(
        run_id=run_id or uuid.uuid4().hex,
        agent_name=agent_name,
        model=model,
        system_prompt=system_prompt,
        messages=list(conv.messages),
        usage=dict(getattr(conv, "usage", {}) or {}),
        iteration=iteration,
        status=status,
        prompt=prompt,
        partial_output=partial_output,
        metadata=copy.deepcopy(metadata) if metadata else {},
    )


def restore_conversation(
    checkpoint: Checkpoint,
    *,
    model: str | None = None,
    driver: Any | None = None,
    **conversation_kwargs: Any,
) -> Conversation:
    """Reconstruct a :class:`Conversation` from *checkpoint*.

    The message history, system prompt, and previously-recorded usage
    are restored so the next call to :meth:`Conversation.ask` continues
    where the snapshot left off.

    Args:
        checkpoint: The snapshot to resume.
        model: Override the model string.  Defaults to the value
            recorded in the checkpoint.
        driver: Pre-built driver to inject (preferred over *model*
            when both are supplied).
        **conversation_kwargs: Forwarded to
            :class:`~prompture.agents.Conversation`.  Callers can
            inject tools, callbacks, options, etc.

    Raises:
        ValueError: When neither *model*, *driver*, nor the
            checkpoint's recorded model can supply a target.
    """
    from ..agents.conversation import Conversation

    effective_model = model or checkpoint.model
    if driver is None and not effective_model:
        raise ValueError("Cannot restore conversation: no model in checkpoint and no override provided")

    kwargs: dict[str, Any] = dict(conversation_kwargs)
    if "system_prompt" not in kwargs and checkpoint.system_prompt:
        kwargs["system_prompt"] = checkpoint.system_prompt
    if driver is not None:
        kwargs["driver"] = driver
    else:
        kwargs["model_name"] = effective_model

    conv = Conversation(**kwargs)
    # Restore message history.  ``messages`` is a read-only property,
    # so write the backing list directly.
    conv._messages = list(checkpoint.messages)
    if checkpoint.usage:
        # Best-effort restore — Conversation.usage may be read-only on
        # some builds; silently skip when the attribute can't be updated.
        with contextlib.suppress(Exception):
            conv.usage.update(checkpoint.usage)
    return conv


# ----------------------------------------------------------------------
# CheckpointManager
# ----------------------------------------------------------------------


class CheckpointManager:
    """Lifecycle helper binding a ``run_id`` to a :class:`CheckpointStore`.

    The manager owns a single run identifier and the most recent
    snapshot for it.  Callers wire it into existing hook points:

    - Pass :meth:`save_conversation` as a ``before_turn`` callback on
      :class:`~prompture.agents.Conversation`.
    - Call :meth:`save_agent` from
      :attr:`AgentCallbacks.on_step` /
      :attr:`AgentCallbacks.on_iteration`.
    - On startup, call :meth:`load` to detect an interrupted run and
      :meth:`resume_conversation` to keep going.

    Args:
        store: Backing :class:`CheckpointStore`.  Defaults to
            :class:`InMemoryCheckpointStore`.
        run_id: Stable run identifier.  Auto-generated when omitted.
        agent_name: Optional default for snapshots produced by this
            manager.

    Example::

        store = SQLiteCheckpointStore()
        mgr = CheckpointManager(store, run_id="trip-research")
        conv = Conversation(
            model_name="openai/gpt-4o",
            before_turn=lambda c, turn: mgr.save_conversation(c, iteration=turn),
        )
        conv.ask("Plan a 5-day Paris trip.")
        # ... crash ...
        cp = mgr.load()
        resumed = mgr.resume_conversation()
        resumed.ask("Continue from where we left off.")
    """

    def __init__(
        self,
        store: CheckpointStore | None = None,
        *,
        run_id: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        self._store: CheckpointStore = store if store is not None else InMemoryCheckpointStore()
        self._run_id = run_id or uuid.uuid4().hex
        self._agent_name = agent_name

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def store(self) -> CheckpointStore:
        return self._store

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save_conversation(
        self,
        conv: Conversation,
        *,
        iteration: int = 0,
        prompt: str = "",
        status: str = RunStatus.running.value,
        partial_output: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Snapshot *conv* and persist under this manager's run id."""
        cp = snapshot_conversation(
            conv,
            run_id=self._run_id,
            iteration=iteration,
            prompt=prompt,
            status=status,
            agent_name=self._agent_name,
            partial_output=partial_output,
            metadata=metadata,
        )
        self._store.save(cp)
        return cp

    def save_agent(
        self,
        agent: Agent,
        *,
        iteration: int = 0,
        prompt: str = "",
        status: str = RunStatus.running.value,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint | None:
        """Snapshot the persistent :class:`Conversation` an Agent holds.

        Only works when the underlying agent has
        ``persistent_conversation=True`` and has already executed at
        least once.  Returns ``None`` if no conversation is available
        yet.
        """
        conv = getattr(agent, "conversation", None) or getattr(agent, "_conversation", None)
        if conv is None:
            return None
        return self.save_conversation(
            conv,
            iteration=iteration,
            prompt=prompt,
            status=status,
            metadata=metadata,
        )

    def mark(self, status: str, *, metadata: dict[str, Any] | None = None) -> Checkpoint | None:
        """Update the latest checkpoint's status (e.g. mark as completed).

        Returns ``None`` when no checkpoint exists for this run.
        """
        cp = self._store.load(self._run_id)
        if cp is None:
            return None
        cp.status = status
        if metadata:
            cp.metadata = {**cp.metadata, **metadata}
        self._store.save(cp)
        return cp

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def load(self) -> Checkpoint | None:
        """Return the latest checkpoint for this run id, or ``None``."""
        return self._store.load(self._run_id)

    def exists(self) -> bool:
        """``True`` when a checkpoint exists for this run id."""
        return self._store.load(self._run_id) is not None

    def resume_conversation(
        self,
        *,
        model: str | None = None,
        driver: Any | None = None,
        **conversation_kwargs: Any,
    ) -> Conversation:
        """Load the latest checkpoint and rebuild the conversation.

        Raises :class:`LookupError` when no checkpoint exists.  All
        keyword arguments are forwarded to
        :func:`restore_conversation`.
        """
        cp = self.load()
        if cp is None:
            raise LookupError(f"No checkpoint found for run_id={self._run_id!r}")
        return restore_conversation(cp, model=model, driver=driver, **conversation_kwargs)

    def delete(self) -> bool:
        """Drop this manager's checkpoint.  Returns whether it existed."""
        return self._store.delete(self._run_id)
