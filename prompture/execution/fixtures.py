"""Versioned evaluation fixtures for the three roadmap workloads.

A :class:`FixtureSet` is an immutable, versioned, content-addressed collection of
:class:`FixtureCase`.  Each case belongs to exactly one :class:`Split`:

``dev``
    Free to look at, prompt against, and optimise on.
``heldout``
    Reserved for final reporting.  :meth:`FixtureSet.dev` is the only accessor
    the optimisation code in :mod:`prompture.execution.improve` is allowed to
    call; the held-out split is reachable only through the explicit
    :meth:`FixtureSet.heldout` accessor, and
    :func:`assert_optimization_safe` refuses a set that mixes the two.

The bundled sets are small and synthetic.  They exist to make the harness,
scoring, and gates runnable and reproducible without network access or licensed
data — they are **not** a benchmark whose absolute numbers mean anything.  Point
:func:`load_fixture_set` at your own JSON file for real measurement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .outcomes import TaskCategory

__all__ = [
    "BUNDLED_FIXTURE_SETS",
    "FixtureCase",
    "FixtureSet",
    "Split",
    "assert_optimization_safe",
    "bundled_fixture_path",
    "load_bundled_fixture_set",
    "load_fixture_set",
    "save_fixture_set",
]

#: Directory holding the bundled JSON fixture files (package data).
DATA_DIR = Path(__file__).parent / "data"

#: ``name -> filename`` for the fixture sets shipped with Prompture.
BUNDLED_FIXTURE_SETS: dict[str, str] = {
    "extraction_contacts": "extraction_contacts.json",
    "document_qa_policies": "document_qa_policies.json",
    "tool_task_inventory": "tool_task_inventory.json",
}


class Split(str, Enum):
    """Which half of a fixture set a case belongs to."""

    DEV = "dev"
    HELDOUT = "heldout"


@dataclass(frozen=True)
class FixtureCase:
    """One evaluation task.

    Attributes:
        id: Stable identifier, unique within its :class:`FixtureSet`.  Becomes
            the ``task_id`` on every :class:`~prompture.execution.outcomes.OutcomeRecord`
            produced for this case, so repeat runs are groupable.
        category: Which workload this is.
        split: ``dev`` or ``heldout``.
        inputs: Everything the strategy needs to execute the task.  Shape is
            per-category and documented on the bundled sets:

            * ``extraction`` — ``{"text": str}``
            * ``document_qa`` — ``{"question": str, "passages": [{"id", "text"}]}``
            * ``tool_task`` — ``{"instruction": str, "initial_state": {...}}``
        expected: Everything the scorer needs to grade the result.  Also
            per-category:

            * ``extraction`` — ``{"fields": {name: value}}``
            * ``document_qa`` — ``{"answerable": bool, "answer_contains": [str],
              "supporting_source_ids": [str]}``
            * ``tool_task`` — ``{"final_state": {...}}``
        tags: Free-form labels (``"hard"``, ``"unanswerable"``, …) used to slice
            reports.
        notes: Provenance / authoring note.  Never fed to a model.
    """

    id: str
    category: TaskCategory
    split: Split
    inputs: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        data["split"] = self.split.value
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixtureCase:
        return cls(
            id=data["id"],
            category=TaskCategory(data["category"]),
            split=Split(data.get("split", Split.DEV.value)),
            inputs=dict(data.get("inputs") or {}),
            expected=dict(data.get("expected") or {}),
            tags=tuple(data.get("tags") or ()),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class FixtureSet:
    """A named, versioned collection of :class:`FixtureCase`.

    Attributes:
        name: Stable set name (``"extraction_contacts"``).
        version: Caller-managed version string.  Bump it whenever a case is
            added, removed, or edited — reports quote it so two runs are only
            comparable when the versions match.
        category: The workload every case in the set belongs to.
        cases: The cases, in file order.
        description: What the set is for.
        source: Where the data came from (``"synthetic"``, a URL, a dataset
            name).  Recorded in reports as data provenance.
        license: Licence of the underlying data, when it is not synthetic.
    """

    name: str
    version: str
    category: TaskCategory
    cases: tuple[FixtureCase, ...] = ()
    description: str = ""
    source: str = "synthetic"
    license: str = ""

    def __post_init__(self) -> None:
        ids = [case.id for case in self.cases]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"FixtureSet {self.name!r} has duplicate case ids: {duplicates}")
        mismatched = sorted({c.id for c in self.cases if c.category is not self.category})
        if mismatched:
            raise ValueError(
                f"FixtureSet {self.name!r} is declared as {self.category.value!r} but cases {mismatched} disagree"
            )

    # ---- access -------------------------------------------------------

    def __iter__(self) -> Iterator[FixtureCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def dev(self) -> tuple[FixtureCase, ...]:
        """The development split — safe to optimise against."""
        return tuple(c for c in self.cases if c.split is Split.DEV)

    def heldout(self) -> tuple[FixtureCase, ...]:
        """The held-out split — for final reporting only.

        Calling this from optimisation code defeats the split.  The candidate
        search in :mod:`prompture.execution.improve` calls :meth:`dev` and takes
        the held-out set as a separate, explicitly-passed argument so the
        boundary is visible at the call site.
        """
        return tuple(c for c in self.cases if c.split is Split.HELDOUT)

    def case(self, case_id: str) -> FixtureCase:
        for c in self.cases:
            if c.id == case_id:
                return c
        raise KeyError(case_id)

    def filter_tags(self, *tags: str) -> tuple[FixtureCase, ...]:
        """Cases carrying **every** tag in *tags*."""
        wanted = set(tags)
        return tuple(c for c in self.cases if wanted.issubset(set(c.tags)))

    # ---- provenance ---------------------------------------------------

    def checksum(self) -> str:
        """Stable SHA-256 over the set's content.

        Two runs quoting the same ``name``, ``version`` and ``checksum`` really
        did evaluate the same cases — a version string alone can be forgotten
        after an edit.
        """
        payload = json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "category": self.category.value,
                "cases": [c.to_dict() for c in self.cases],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def provenance(self) -> dict[str, Any]:
        """Compact provenance block for embedding in a report."""
        return {
            "name": self.name,
            "version": self.version,
            "category": self.category.value,
            "source": self.source,
            "license": self.license,
            "checksum": self.checksum(),
            "n_cases": len(self.cases),
            "n_dev": len(self.dev()),
            "n_heldout": len(self.heldout()),
        }

    # ---- serialisation ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "category": self.category.value,
            "description": self.description,
            "source": self.source,
            "license": self.license,
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixtureSet:
        return cls(
            name=data["name"],
            version=str(data.get("version", "0")),
            category=TaskCategory(data["category"]),
            cases=tuple(FixtureCase.from_dict(c) for c in data.get("cases") or ()),
            description=data.get("description", ""),
            source=data.get("source", "synthetic"),
            license=data.get("license", ""),
        )


def load_fixture_set(path: str | Path) -> FixtureSet:
    """Load a :class:`FixtureSet` from a JSON file."""
    raw = Path(path).read_text(encoding="utf-8")
    return FixtureSet.from_dict(json.loads(raw))


def save_fixture_set(fixture_set: FixtureSet, path: str | Path) -> Path:
    """Write *fixture_set* to *path* as pretty JSON.  Returns the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(fixture_set.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def bundled_fixture_path(name: str) -> Path:
    """Filesystem path of a bundled fixture set."""
    try:
        filename = BUNDLED_FIXTURE_SETS[name]
    except KeyError:
        raise KeyError(f"Unknown bundled fixture set {name!r}. Known: {sorted(BUNDLED_FIXTURE_SETS)}") from None
    return DATA_DIR / filename


def load_bundled_fixture_set(name: str) -> FixtureSet:
    """Load one of the small synthetic sets shipped with Prompture.

    ``"extraction_contacts"``, ``"document_qa_policies"``, or
    ``"tool_task_inventory"``.
    """
    return load_fixture_set(bundled_fixture_path(name))


def assert_optimization_safe(cases: tuple[FixtureCase, ...] | list[FixtureCase]) -> None:
    """Raise when *cases* contain anything from the held-out split.

    Called by the offline optimiser before it scores a candidate, so an
    accidental ``fixture_set.cases`` (rather than ``fixture_set.dev()``) fails
    loudly instead of quietly contaminating the final report.
    """
    leaked = sorted(c.id for c in cases if c.split is Split.HELDOUT)
    if leaked:
        raise ValueError(
            "Held-out cases must not be used for optimisation or candidate "
            f"selection; got {leaked}. Use FixtureSet.dev() for tuning and pass "
            "the held-out split separately for final reporting."
        )
