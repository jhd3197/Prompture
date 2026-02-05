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
"""

from __future__ import annotations

import collections.abc
import dataclasses
import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .persona import Persona

logger = logging.getLogger("prompture.skills")

_SERIALIZATION_VERSION = 1

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
    """

    name: str
    description: str
    instructions: str
    user_invocable: bool = True
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    source_path: Path | None = None
    skill_dir: Path | None = None

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
        )


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: The full file content with optional YAML frontmatter.

    Returns:
        Tuple of (frontmatter_dict, body_text).
        If no frontmatter is found, returns ({}, original_content).
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
        logger.warning(f"Failed to parse YAML frontmatter: {e}")
        return {}, content

    return frontmatter, body


def load_skill(path: str | Path) -> SkillInfo:
    """Load a skill from a SKILL.md file.

    Args:
        path: Path to the SKILL.md file.

    Returns:
        SkillInfo instance with parsed data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing from frontmatter.
        ImportError: If pyyaml is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")

    content = path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(content)

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
    )


def load_skill_from_directory(directory: str | Path) -> SkillInfo:
    """Load a skill from a directory containing SKILL.md.

    Args:
        directory: Path to the skill directory.

    Returns:
        SkillInfo instance.

    Raises:
        FileNotFoundError: If SKILL.md is not found in the directory.
    """
    directory = Path(directory)
    skill_file = directory / "SKILL.md"
    if not skill_file.exists():
        raise FileNotFoundError(f"No SKILL.md found in directory: {directory}")
    return load_skill(skill_file)


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


class _SkillRegistryProxy(dict, collections.abc.MutableMapping):
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

    def __iter__(self):
        return iter(get_skill_names())

    def keys(self):
        return get_skill_names()

    def values(self):
        with _skill_registry_lock:
            return list(_skill_global_registry.values())

    def items(self):
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
) -> list[SkillInfo]:
    """Bulk-load skills from a directory.

    Scans for subdirectories containing SKILL.md files and loads each one.

    Args:
        path: Directory to scan for skill subdirectories.
        register: If True, register each loaded skill in the global registry.

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
            skill = load_skill(skill_file)
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
    scan_node_modules: bool = False,
    register: bool = True,
) -> list[SkillInfo]:
    """Discover and load skills from standard locations.

    Scans the following locations:
    - ``./.claude/skills/*/SKILL.md`` (project-local)
    - ``~/.claude/skills/*/SKILL.md`` (user-global)
    - Any additional paths provided

    Args:
        additional_paths: Extra directories to scan for skills.
        scan_node_modules: If True, also scan ``./node_modules/.claude/skills/``.
            Disabled by default for performance.
        register: If True, register discovered skills in the global registry.

    Returns:
        List of all discovered SkillInfo instances.
    """
    all_skills: list[SkillInfo] = []
    scanned_dirs: set[Path] = set()

    # Standard paths
    for skill_dir in _get_standard_skill_paths():
        if skill_dir in scanned_dirs:
            continue
        scanned_dirs.add(skill_dir)
        all_skills.extend(load_skills_from_directory(skill_dir, register=register))

    # Additional paths
    if additional_paths:
        for path in additional_paths:
            path = Path(path)
            if path in scanned_dirs:
                continue
            scanned_dirs.add(path)
            all_skills.extend(load_skills_from_directory(path, register=register))

    # Optional: node_modules
    if scan_node_modules:
        node_skills = Path.cwd() / "node_modules" / ".claude" / "skills"
        if node_skills.exists() and node_skills not in scanned_dirs:
            all_skills.extend(load_skills_from_directory(node_skills, register=register))

    logger.info(f"Discovered {len(all_skills)} skills from {len(scanned_dirs)} directories")
    return all_skills
