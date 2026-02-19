"""Agent Skills module for Prompture.

Provides integration with the Agent Skills ecosystem (skills.sh / Claude Code format).
Parses SKILL.md files with YAML frontmatter, converts to Persona instances, and
maintains a thread-safe skill registry.

Features:
- Frozen ``SkillInfo`` dataclass with Persona conversion
- YAML frontmatter parsing from SKILL.md files
- Thread-safe global skill registry with dict-like proxy
- Auto-discovery from standard skill locations
- Resource path resolution for skill assets
- Async variants for web framework compatibility
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import logging
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .persona import Persona

logger = logging.getLogger("prompture.skills")

_SERIALIZATION_VERSION = 1

# Type alias for skill source locations
SkillSource = Literal["project", "user", "node_modules", "additional"]


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class SkillParseError(Exception):
    """Raised when a SKILL.md file cannot be parsed.

    Attributes:
        message: Human-readable error description.
        path: Path to the file that failed to parse.
        line: Line number where the error occurred (if applicable).
        suggestion: Suggested fix for the error.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path | str | None = None,
        line: int | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.message = message
        self.path = Path(path) if path else None
        self.line = line
        self.suggestion = suggestion

        # Build detailed error message
        parts = [message]
        if path:
            parts.append(f"File: {path}")
        if line:
            parts.append(f"Line: {line}")
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")

        super().__init__("\n".join(parts))


# Regex pattern for YAML frontmatter (--- delimited block at start of file)
_FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    re.DOTALL,
)


# ------------------------------------------------------------------
# SkillInfo dataclass
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SkillInfo:
    """Metadata and instructions for an Agent Skill.

    Instances are immutable (frozen). Use :meth:`as_persona` to convert
    to a Prompture Persona for use in conversations.

    Args:
        name: Skill identifier (from YAML frontmatter ``name`` field).
        description: Human-readable description of what the skill does.
        instructions: The markdown body containing agent instructions.
        user_invocable: Whether the skill can be invoked by users directly.
        metadata: Additional metadata from the YAML frontmatter.
        source_path: Path to the SKILL.md file (if loaded from file).
        skill_dir: Directory containing the skill (for resource resolution).
        source: Where the skill was discovered from ("project", "user",
            "node_modules", or "additional"). None if loaded directly.
    """

    name: str
    description: str
    instructions: str
    user_invocable: bool = True
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    source_path: Path | None = None
    skill_dir: Path | None = None
    source: SkillSource | None = None

    # Backwards-compatible alias for source_path
    @property
    def path(self) -> Path | None:
        """Alias for source_path (for backwards compatibility)."""
        return self.source_path

    def as_persona(self) -> Persona:
        """Convert this skill to a Prompture Persona.

        The skill's instructions become the system prompt. The skill name
        is converted to a valid Python identifier (hyphens to underscores).

        Returns:
            A Persona instance configured with this skill's data.
        """
        from .persona import Persona

        # Build settings dict with skill metadata if present
        settings: dict[str, Any] = {}
        if self.metadata:
            settings["skill_metadata"] = dict(self.metadata)

        return Persona(
            name=self.name.replace("-", "_"),
            system_prompt=self.instructions,
            description=self.description,
            settings=settings if settings else {},
        )

    def get_resource_path(self, relative_path: str) -> Path | None:
        """Get path to a resource file within the skill directory.

        Args:
            relative_path: Path relative to the skill directory.

        Returns:
            Absolute Path to the resource, or None if skill_dir is not set.
        """
        if self.skill_dir is None:
            return None
        return self.skill_dir / relative_path

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        data: dict[str, Any] = {
            "version": _SERIALIZATION_VERSION,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "user_invocable": self.user_invocable,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        if self.source_path is not None:
            data["source_path"] = str(self.source_path)
        if self.skill_dir is not None:
            data["skill_dir"] = str(self.skill_dir)
        if self.source is not None:
            data["source"] = self.source
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillInfo:
        """Deserialize from a dictionary."""
        source_path = data.get("source_path")
        skill_dir = data.get("skill_dir")
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            instructions=data.get("instructions", ""),
            user_invocable=data.get("user_invocable", True),
            metadata=dict(data.get("metadata", {})),
            source_path=Path(source_path) if source_path else None,
            skill_dir=Path(skill_dir) if skill_dir else None,
            source=data.get("source"),
        )


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------


def _parse_frontmatter(
    content: str,
    *,
    filepath: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: The full file content with optional YAML frontmatter.
        filepath: Optional path for error reporting.

    Returns:
        Tuple of (frontmatter_dict, body_text).
        If no frontmatter is found, returns ({}, original_content).

    Raises:
        SkillParseError: If YAML frontmatter is malformed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required for skill parsing. Install with: pip install pyyaml") from None

    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = match.group(2)

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
        if frontmatter is None:
            frontmatter = {}
    except yaml.YAMLError as e:
        # Extract line number from YAML error if available
        line = None
        if hasattr(e, "problem_mark") and e.problem_mark is not None:
            line = e.problem_mark.line + 1  # YAML uses 0-indexed lines

        raise SkillParseError(
            f"Invalid YAML frontmatter: {e}",
            path=filepath,
            line=line,
            suggestion="Ensure YAML frontmatter is enclosed in --- delimiters and uses valid YAML syntax",
        ) from e

    return frontmatter, body


def load_skill(
    path: str | Path,
    *,
    source: SkillSource | None = None,
) -> SkillInfo:
    """Load a skill from a SKILL.md file.

    Args:
        path: Path to the SKILL.md file.
        source: Where the skill was discovered from (for discovery tracking).

    Returns:
        SkillInfo instance with parsed data.

    Raises:
        FileNotFoundError: If the file does not exist.
        SkillParseError: If the file cannot be parsed.
        ImportError: If pyyaml is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")

    content = path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content, filepath=path)

    # Extract required fields
    name = frontmatter.get("name")
    if not name:
        # Fall back to directory name or file stem
        name = path.parent.name if path.name == "SKILL.md" else path.stem
        logger.debug(f"No 'name' in frontmatter, using: {name}")

    description = frontmatter.get("description", "")

    # Check for user-invocable flag (supports both hyphenated and underscored)
    user_invocable = frontmatter.get("user-invocable", frontmatter.get("user_invocable", True))

    # Extract metadata - either from a nested 'metadata' key or remaining top-level keys
    known_keys = {"name", "description", "user-invocable", "user_invocable", "metadata"}
    if "metadata" in frontmatter and isinstance(frontmatter["metadata"], dict):
        # Use explicit metadata block
        metadata = dict(frontmatter["metadata"])
    else:
        # Collect remaining top-level keys as metadata
        metadata = {k: v for k, v in frontmatter.items() if k not in known_keys}

    return SkillInfo(
        name=name,
        description=description,
        instructions=body.strip(),
        user_invocable=bool(user_invocable),
        metadata=metadata,
        source_path=path.resolve(),
        skill_dir=path.parent.resolve(),
        source=source,
    )


async def load_skill_async(
    path: str | Path,
    *,
    source: SkillSource | None = None,
) -> SkillInfo:
    """Async variant of :func:`load_skill`.

    Runs file I/O in a thread pool to avoid blocking the event loop.

    Args:
        path: Path to the SKILL.md file.
        source: Where the skill was discovered from (for discovery tracking).

    Returns:
        SkillInfo instance with parsed data.

    Raises:
        FileNotFoundError: If the file does not exist.
        SkillParseError: If the file cannot be parsed.
        ImportError: If pyyaml is not installed.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: load_skill(path, source=source))


def load_skill_from_directory(
    directory: str | Path,
    *,
    source: SkillSource | None = None,
) -> SkillInfo:
    """Load a skill from a directory containing SKILL.md.

    Args:
        directory: Path to the skill directory.
        source: Where the skill was discovered from (for discovery tracking).

    Returns:
        SkillInfo instance.

    Raises:
        FileNotFoundError: If SKILL.md is not found in the directory.
    """
    directory = Path(directory)
    skill_file = directory / "SKILL.md"
    if not skill_file.exists():
        raise FileNotFoundError(f"No SKILL.md found in directory: {directory}")
    return load_skill(skill_file, source=source)


# ------------------------------------------------------------------
# Global skill registry
# ------------------------------------------------------------------

_skill_registry_lock = threading.Lock()
_skill_global_registry: dict[str, SkillInfo] = {}


def register_skill(skill: SkillInfo) -> None:
    """Register a skill in the global registry.

    Args:
        skill: The SkillInfo instance to register.
    """
    with _skill_registry_lock:
        _skill_global_registry[skill.name] = skill
        logger.debug(f"Registered skill: {skill.name}")


def get_skill(name: str) -> SkillInfo | None:
    """Retrieve a skill by name.

    Args:
        name: The skill name.

    Returns:
        SkillInfo instance, or None if not found.
    """
    with _skill_registry_lock:
        return _skill_global_registry.get(name)


def get_skill_names() -> list[str]:
    """Return all registered skill names."""
    with _skill_registry_lock:
        return list(_skill_global_registry.keys())


def unregister_skill(name: str) -> bool:
    """Remove a skill from the registry.

    Args:
        name: The skill name to unregister.

    Returns:
        True if the skill was found and removed, False otherwise.
    """
    with _skill_registry_lock:
        if name in _skill_global_registry:
            del _skill_global_registry[name]
            logger.debug(f"Unregistered skill: {name}")
            return True
        return False


def clear_skill_registry() -> None:
    """Remove all skills from the global registry."""
    with _skill_registry_lock:
        _skill_global_registry.clear()
        logger.debug("Cleared skill registry")


def get_skill_registry_snapshot() -> dict[str, SkillInfo]:
    """Return a shallow copy of the current skill registry."""
    with _skill_registry_lock:
        return dict(_skill_global_registry)


# ------------------------------------------------------------------
# Skill registry proxy (dict-like access)
# ------------------------------------------------------------------


class _SkillRegistryProxy(dict[str, SkillInfo], collections.abc.MutableMapping[str, SkillInfo]):
    """Dict-like proxy for the global skill registry.

    Allows ``SKILLS["add-persona"]`` style access.
    """

    def __getitem__(self, key: str) -> SkillInfo:
        skill = get_skill(key)
        if skill is None:
            raise KeyError(f"Skill '{key}' not found in registry. Available: {', '.join(get_skill_names())}")
        return skill

    def __setitem__(self, key: str, value: SkillInfo) -> None:
        if not isinstance(value, SkillInfo):
            raise TypeError(f"Expected SkillInfo instance, got {type(value).__name__}")
        with _skill_registry_lock:
            _skill_global_registry[key] = value

    def __delitem__(self, key: str) -> None:
        with _skill_registry_lock:
            if key in _skill_global_registry:
                del _skill_global_registry[key]
            else:
                raise KeyError(f"Skill '{key}' not found in registry")

    def __contains__(self, key: object) -> bool:
        return key in get_skill_names()

    def __iter__(self) -> Iterator[str]:
        return iter(get_skill_names())

    def keys(self) -> list[str]:  # type: ignore[override]
        return get_skill_names()

    def values(self) -> list[SkillInfo]:  # type: ignore[override]
        with _skill_registry_lock:
            return list(_skill_global_registry.values())

    def items(self) -> list[tuple[str, SkillInfo]]:  # type: ignore[override]
        with _skill_registry_lock:
            return list(_skill_global_registry.items())

    def __len__(self) -> int:
        with _skill_registry_lock:
            return len(_skill_global_registry)

    def get(self, key: str, default: Any = None) -> Any:
        skill = get_skill(key)
        return skill if skill is not None else default

    def __repr__(self) -> str:
        return f"SKILLS({get_skill_names()})"


# Public proxy instance
SKILLS = _SkillRegistryProxy()


# ------------------------------------------------------------------
# Skill discovery
# ------------------------------------------------------------------


def _get_standard_skill_paths() -> list[Path]:
    """Get the standard skill directory paths to scan.

    Returns:
        List of paths: [cwd/.claude/skills, ~/.claude/skills]
    """
    paths = []

    # Project-local skills
    cwd_skills = Path.cwd() / ".claude" / "skills"
    if cwd_skills.exists():
        paths.append(cwd_skills)

    # User-global skills
    home = Path.home()
    home_skills = home / ".claude" / "skills"
    if home_skills.exists() and home_skills not in paths:
        paths.append(home_skills)

    return paths


def load_skills_from_directory(
    path: str | Path,
    *,
    register: bool = True,
    source: SkillSource | None = None,
) -> list[SkillInfo]:
    """Bulk-load skills from a directory.

    Scans for subdirectories containing SKILL.md files and loads each one.

    Args:
        path: Directory to scan for skill subdirectories.
        register: If True, register each loaded skill in the global registry.
        source: Where the skills were discovered from (for discovery tracking).

    Returns:
        List of loaded SkillInfo instances.
    """
    directory = Path(path)
    if not directory.exists():
        logger.debug(f"Skill directory does not exist: {directory}")
        return []

    loaded: list[SkillInfo] = []

    for entry in sorted(directory.iterdir()):
        if not entry.is_dir():
            continue

        skill_file = entry / "SKILL.md"
        if not skill_file.exists():
            continue

        try:
            skill = load_skill(skill_file, source=source)
            if register:
                register_skill(skill)
            loaded.append(skill)
            logger.debug(f"Loaded skill from {entry}: {skill.name}")
        except Exception as e:
            logger.warning(f"Failed to load skill from {entry}: {e}")

    return loaded


def discover_skills(
    *,
    additional_paths: list[str | Path] | None = None,
    exclude_paths: list[str | Path] | None = None,
    scan_node_modules: bool = False,
    register: bool = True,
) -> list[SkillInfo]:
    """Discover and load skills from standard locations.

    Scans the following locations:
    - ``./.claude/skills/*/SKILL.md`` (project-local, source="project")
    - ``~/.claude/skills/*/SKILL.md`` (user-global, source="user")
    - Any additional paths provided (source="additional")

    Each discovered skill has its ``source`` attribute set to indicate where
    it was found: "project", "user", "node_modules", or "additional".

    Args:
        additional_paths: Extra directories to scan for skills.
        exclude_paths: Directories to exclude from scanning (e.g., to skip
            user-global skills when only project skills are desired).
        scan_node_modules: If True, also scan ``./node_modules/.claude/skills/``.
            Disabled by default for performance.
        register: If True, register discovered skills in the global registry.

    Returns:
        List of all discovered SkillInfo instances.

    Example:
        >>> # Discover only project-local skills
        >>> from pathlib import Path
        >>> skills = discover_skills(
        ...     exclude_paths=[Path.home() / ".claude" / "skills"]
        ... )
        >>> for skill in skills:
        ...     print(f"{skill.name}: {skill.source}")
    """
    all_skills: list[SkillInfo] = []
    scanned_dirs: set[Path] = set()

    # Build exclusion set
    excluded: set[Path] = set()
    if exclude_paths:
        for p in exclude_paths:
            excluded.add(Path(p).resolve())

    # Project-local skills
    cwd_skills = Path.cwd() / ".claude" / "skills"
    if cwd_skills.exists() and cwd_skills.resolve() not in excluded:
        scanned_dirs.add(cwd_skills)
        all_skills.extend(load_skills_from_directory(cwd_skills, register=register, source="project"))

    # User-global skills
    home_skills = Path.home() / ".claude" / "skills"
    if home_skills.exists() and home_skills.resolve() not in excluded and home_skills not in scanned_dirs:
        scanned_dirs.add(home_skills)
        all_skills.extend(load_skills_from_directory(home_skills, register=register, source="user"))

    # Additional paths
    if additional_paths:
        for path in additional_paths:
            path = Path(path)
            if path.resolve() in excluded:
                continue
            if path in scanned_dirs:
                continue
            scanned_dirs.add(path)
            all_skills.extend(load_skills_from_directory(path, register=register, source="additional"))

    # Optional: node_modules
    if scan_node_modules:
        node_skills = Path.cwd() / "node_modules" / ".claude" / "skills"
        if node_skills.exists() and node_skills.resolve() not in excluded and node_skills not in scanned_dirs:
            scanned_dirs.add(node_skills)
            all_skills.extend(load_skills_from_directory(node_skills, register=register, source="node_modules"))

    logger.info(f"Discovered {len(all_skills)} skills from {len(scanned_dirs)} directories")
    return all_skills


async def discover_skills_async(
    *,
    additional_paths: list[str | Path] | None = None,
    exclude_paths: list[str | Path] | None = None,
    scan_node_modules: bool = False,
    register: bool = True,
) -> list[SkillInfo]:
    """Async variant of :func:`discover_skills`.

    Runs file I/O in a thread pool to avoid blocking the event loop.
    Useful for web applications using async frameworks like FastAPI or aiohttp.

    Args:
        additional_paths: Extra directories to scan for skills.
        exclude_paths: Directories to exclude from scanning.
        scan_node_modules: If True, also scan ``./node_modules/.claude/skills/``.
        register: If True, register discovered skills in the global registry.

    Returns:
        List of all discovered SkillInfo instances.

    Example:
        >>> # In an async context (e.g., FastAPI startup)
        >>> skills = await discover_skills_async()
        >>> for skill in skills:
        ...     print(f"{skill.name} from {skill.source}")
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: discover_skills(
            additional_paths=additional_paths,
            exclude_paths=exclude_paths,
            scan_node_modules=scan_node_modules,
            register=register,
        ),
    )
