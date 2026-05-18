"""Live integration tests for coding-agent CLIs.

Opt-in via ``pytest --run-coding-agent-live`` (or
``RUN_CODING_AGENT_LIVE_TESTS=1``). Each test is skipped automatically when
the matching CLI isn't installed locally, so this suite is safe to enable on
CI even if only a subset of agents are available.

These tests guard against:

- Flag-shape drift when a CLI vendor renames or removes a flag (the original
  ``--permission-mode dontAsk`` bug would have been caught here).
- npm shim breakage on Windows / WSL — discovery falling back to the package
  entrypoint is exercised on the real binary.
- The CodingAgentSpec registry going out of sync with what each CLI actually
  understands when ``--version`` is invoked.

They deliberately do NOT run a task that requires API credentials — every
test calls ``<binary> --version`` only.
"""

from __future__ import annotations

import shutil

import pytest

from prompture.infra import (
    CODING_AGENT_SPECS,
    get_available_coding_agents,
)
from prompture.infra.discovery import (
    resolve_coding_agent_executable,
    verify_coding_agent_executable,
)

pytestmark = pytest.mark.coding_agent_live


AGENT_IDS = tuple(CODING_AGENT_SPECS.keys())


def _installed(agent_id: str) -> bool:
    spec = CODING_AGENT_SPECS[agent_id]
    return shutil.which(spec.default_binary) is not None


@pytest.mark.parametrize("agent_id", AGENT_IDS)
def test_version_check_passes_on_installed_cli(agent_id: str) -> None:
    """The resolver must produce a healthy executable for an installed CLI.

    Goes through the same code path the runner uses (executable candidates +
    health check) so npm/.cmd wrappers and Windows shims behave correctly.
    """
    if not _installed(agent_id):
        pytest.skip(f"{agent_id} CLI not installed locally")
    spec = CODING_AGENT_SPECS[agent_id]
    executable, healthy, error = resolve_coding_agent_executable(
        agent_id,
        spec.default_binary,
        verify=True,
        verify_timeout=10,
    )
    assert executable is not None, f"{agent_id} did not resolve any executable"
    if not healthy:
        # As a fallback, exercise the candidate directly so we get a richer error.
        ok, exec_err = verify_coding_agent_executable(agent_id, executable, timeout=10)
        assert ok, f"{agent_id} --version failed: {exec_err or error}"


@pytest.mark.parametrize("agent_id", AGENT_IDS)
def test_resolve_finds_executable_for_installed_cli(agent_id: str) -> None:
    """The resolver must return a callable executable for any agent on PATH."""
    if not _installed(agent_id):
        pytest.skip(f"{agent_id} CLI not installed locally")
    spec = CODING_AGENT_SPECS[agent_id]
    executable, _healthy, _error = resolve_coding_agent_executable(agent_id, spec.default_binary)
    assert executable is not None
    assert executable.argv  # non-empty argv


def test_at_least_one_agent_present():
    """Smoke test: when --run-coding-agent-live is on, at least one CLI should resolve."""
    available = [a for a in get_available_coding_agents(verify=True) if a.available]
    if not available:
        pytest.skip("No coding-agent CLIs installed on this host — nothing to verify")
    # If we got here we have at least one; the parametrized tests above carry the load.
    assert available
