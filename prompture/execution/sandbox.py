"""A deterministic, in-process tool world for evaluating tool-using tasks.

Real side-effecting tools cannot be used for benchmarking: a shadow run or a
fault-injection test would duplicate the external action.  :class:`InventoryWorld`
is a small, fully local stand-in with the properties an evaluation harness needs:

* **Deterministic.**  No clocks, no randomness, no network.  The same call
  sequence always produces the same state.
* **Inspectable.**  :meth:`InventoryWorld.snapshot` returns the whole world, so a
  task is graded on *state*, not on what the model claimed it did.
* **Idempotent by operation id.**  Every mutating call accepts an
  ``operation_id``; replaying a call with an id the world has already applied
  returns the original result instead of applying it twice.  That is what makes
  Phase F's crash/replay tests meaningful.
* **Injectable failure.**  :class:`FaultPlan` makes a chosen call raise, or raise
  *after* the state change has been applied — the crash window that
  at-least-once systems actually have to survive.

The world exposes its operations as a
:class:`~prompture.agents.tools_schema.ToolRegistry` so an agent, a strategy, or a
plain test can drive it through the same surface.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FaultPlan",
    "InjectedFailure",
    "InventoryWorld",
    "ToolCallRecord",
    "normalize_inventory_state",
]


class InjectedFailure(RuntimeError):
    """Raised by :class:`InventoryWorld` when a :class:`FaultPlan` fires."""


class InventoryError(ValueError):
    """A legitimate refusal by the world (insufficient stock, unknown order).

    Distinct from :class:`InjectedFailure`: this is the world working correctly
    and saying no, which a strategy is expected to surface as an abstention
    rather than retry blindly.
    """


@dataclass
class ToolCallRecord:
    """One attempted tool call, successful or not."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    result: Any = None
    error: str | None = None
    operation_id: str | None = None
    replayed: bool = False
    applied: bool = False
    """``True`` when the state change landed, even if the call then raised.

    A fault injected *after* the write leaves ``applied=True`` with
    ``ok=False`` — exactly the ambiguous case a caller has to reconcile.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "operation_id": self.operation_id,
            "replayed": self.replayed,
            "applied": self.applied,
        }


@dataclass
class FaultPlan:
    """When and how :class:`InventoryWorld` should fail.

    Attributes:
        fail_on: ``{tool_name: nth_call}`` — 1-based.  The nth call to that
            tool raises :class:`InjectedFailure`.
        after_apply: Tool names whose injected failure fires *after* the state
            mutation is committed.  Names not listed fail before applying.
        message: Text of the raised error.
    """

    fail_on: dict[str, int] = field(default_factory=dict)
    after_apply: set[str] = field(default_factory=set)
    message: str = "injected transport failure"

    def should_fail(self, tool: str, call_index: int) -> bool:
        """``call_index`` is 1-based for this tool."""
        return self.fail_on.get(tool) == call_index

    def fails_after_apply(self, tool: str) -> bool:
        return tool in self.after_apply


def normalize_inventory_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """Canonical form used for comparing a world state to an expectation.

    Zero quantities and emptied containers are dropped so ``{"A": 0}`` and a
    missing ``"A"`` compare equal — fixtures should not have to spell out every
    warehouse that ended up empty.  Prices are coerced to float so ``2`` and
    ``2.0`` match.
    """
    state = state or {}
    stock: dict[str, dict[str, int]] = {}
    for sku, warehouses in (state.get("stock") or {}).items():
        cleaned = {w: int(q) for w, q in (warehouses or {}).items() if int(q) != 0}
        if cleaned:
            stock[sku] = dict(sorted(cleaned.items()))

    reservations: dict[str, dict[str, int]] = {}
    for order, lines in (state.get("reservations") or {}).items():
        cleaned = {sku: int(q) for sku, q in (lines or {}).items() if int(q) != 0}
        if cleaned:
            reservations[order] = dict(sorted(cleaned.items()))

    prices = {sku: float(p) for sku, p in (state.get("prices") or {}).items()}

    return {
        "stock": dict(sorted(stock.items())),
        "reservations": dict(sorted(reservations.items())),
        "prices": dict(sorted(prices.items())),
    }


class InventoryWorld:
    """A tiny warehouse world driven by six tools.

    Args:
        state: Initial state in the fixture shape
            ``{"stock": {sku: {warehouse: qty}}, "reservations": {order: {sku: qty}},
            "prices": {sku: float}}``.  Copied, never aliased.
        faults: Optional :class:`FaultPlan`.

    Example::

        world = InventoryWorld({"stock": {"SKU-1": {"A": 5}}})
        world.move_stock("SKU-1", "A", "B", 2)
        assert world.snapshot()["stock"] == {"SKU-1": {"A": 3, "B": 2}}
    """

    #: Tools that change state.  Read-only tools are excluded from the
    #: operation ledger and from ``writes``.
    MUTATING_TOOLS = frozenset({"move_stock", "reserve_stock", "cancel_order", "restock", "set_price"})

    def __init__(self, state: dict[str, Any] | None = None, *, faults: FaultPlan | None = None) -> None:
        base = normalize_inventory_state(state)
        self._stock: dict[str, dict[str, int]] = copy.deepcopy(base["stock"])
        self._reservations: dict[str, dict[str, int]] = copy.deepcopy(base["reservations"])
        self._prices: dict[str, float] = dict(base["prices"])
        self._faults = faults or FaultPlan()
        self._calls: list[ToolCallRecord] = []
        self._call_counts: dict[str, int] = {}
        self._applied_ops: dict[str, Any] = {}
        self._lock = threading.RLock()

    # ---- inspection ---------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A normalized deep copy of the current state."""
        with self._lock:
            return normalize_inventory_state(
                {
                    "stock": copy.deepcopy(self._stock),
                    "reservations": copy.deepcopy(self._reservations),
                    "prices": dict(self._prices),
                }
            )

    @property
    def calls(self) -> list[ToolCallRecord]:
        """Every attempted call, in order."""
        with self._lock:
            return list(self._calls)

    @property
    def writes(self) -> list[ToolCallRecord]:
        """Attempted calls to mutating tools."""
        return [c for c in self.calls if c.name in self.MUTATING_TOOLS]

    @property
    def applied_operation_ids(self) -> set[str]:
        """Operation ids whose effect is already committed."""
        with self._lock:
            return set(self._applied_ops)

    def matches(self, expected_state: dict[str, Any]) -> bool:
        """Whether the current state equals *expected_state* after normalisation."""
        return self.snapshot() == normalize_inventory_state(expected_state)

    # ---- internals ----------------------------------------------------

    def _record(self, record: ToolCallRecord) -> None:
        self._calls.append(record)

    def _begin(self, tool: str, arguments: dict[str, Any], operation_id: str | None) -> tuple[bool, Any]:
        """Bump the call counter and short-circuit an already-applied op.

        Returns ``(replayed, cached_result)``.
        """
        self._call_counts[tool] = self._call_counts.get(tool, 0) + 1
        if operation_id is not None and operation_id in self._applied_ops:
            cached = self._applied_ops[operation_id]
            self._record(
                ToolCallRecord(
                    name=tool,
                    arguments=dict(arguments),
                    ok=True,
                    result=cached,
                    operation_id=operation_id,
                    replayed=True,
                    applied=True,
                )
            )
            return True, cached
        return False, None

    def _maybe_fail_before(self, tool: str, arguments: dict[str, Any], operation_id: str | None) -> None:
        index = self._call_counts.get(tool, 0)
        if self._faults.should_fail(tool, index) and not self._faults.fails_after_apply(tool):
            self._record(
                ToolCallRecord(
                    name=tool,
                    arguments=dict(arguments),
                    ok=False,
                    error=self._faults.message,
                    operation_id=operation_id,
                    applied=False,
                )
            )
            raise InjectedFailure(self._faults.message)

    def _finish(
        self,
        tool: str,
        arguments: dict[str, Any],
        result: Any,
        operation_id: str | None,
    ) -> Any:
        """Commit an applied mutation, then honour an after-apply fault."""
        if operation_id is not None and tool in self.MUTATING_TOOLS:
            self._applied_ops[operation_id] = result
        index = self._call_counts.get(tool, 0)
        if self._faults.should_fail(tool, index) and self._faults.fails_after_apply(tool):
            self._record(
                ToolCallRecord(
                    name=tool,
                    arguments=dict(arguments),
                    ok=False,
                    result=result,
                    error=self._faults.message,
                    operation_id=operation_id,
                    applied=True,
                )
            )
            raise InjectedFailure(self._faults.message)
        self._record(
            ToolCallRecord(
                name=tool,
                arguments=dict(arguments),
                ok=True,
                result=result,
                operation_id=operation_id,
                applied=tool in self.MUTATING_TOOLS,
            )
        )
        return result

    def _fail(self, tool: str, arguments: dict[str, Any], message: str, operation_id: str | None) -> None:
        self._record(
            ToolCallRecord(
                name=tool,
                arguments=dict(arguments),
                ok=False,
                error=message,
                operation_id=operation_id,
                applied=False,
            )
        )
        raise InventoryError(message)

    def _available(self, sku: str, warehouse: str) -> int:
        return int(self._stock.get(sku, {}).get(warehouse, 0))

    def _add(self, sku: str, warehouse: str, delta: int) -> None:
        bucket = self._stock.setdefault(sku, {})
        bucket[warehouse] = bucket.get(warehouse, 0) + delta
        if bucket[warehouse] == 0:
            bucket.pop(warehouse)
        if not bucket:
            self._stock.pop(sku, None)

    # ---- tools --------------------------------------------------------

    def get_stock(self, sku: str, warehouse: str) -> int:
        """Return the units of *sku* currently held in *warehouse*.

        Args:
            sku: Stock keeping unit, e.g. ``"SKU-100"``.
            warehouse: Warehouse code, e.g. ``"A"``.

        Returns:
            The available unit count (0 when unknown).
        """
        with self._lock:
            args = {"sku": sku, "warehouse": warehouse}
            self._begin("get_stock", args, None)
            self._maybe_fail_before("get_stock", args, None)
            return self._finish("get_stock", args, self._available(sku, warehouse), None)

    def list_inventory(self) -> dict[str, Any]:
        """Return the full stock table as ``{sku: {warehouse: units}}``."""
        with self._lock:
            args: dict[str, Any] = {}
            self._begin("list_inventory", args, None)
            self._maybe_fail_before("list_inventory", args, None)
            return self._finish("list_inventory", args, copy.deepcopy(self._stock), None)

    def move_stock(
        self,
        sku: str,
        source: str,
        destination: str,
        quantity: int,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Move *quantity* units of *sku* from *source* to *destination*.

        Args:
            sku: Stock keeping unit.
            source: Warehouse to take units from.
            destination: Warehouse to put units into.
            quantity: Positive number of units to move.
            operation_id: Optional stable id making the move idempotent.

        Returns:
            ``{"moved": int, "source_remaining": int, "destination_total": int}``.
        """
        with self._lock:
            args = {
                "sku": sku,
                "source": source,
                "destination": destination,
                "quantity": quantity,
                "operation_id": operation_id,
            }
            replayed, cached = self._begin("move_stock", args, operation_id)
            if replayed:
                return dict(cached)
            self._maybe_fail_before("move_stock", args, operation_id)
            quantity = int(quantity)
            if quantity <= 0:
                self._fail("move_stock", args, "quantity must be positive", operation_id)
            available = self._available(sku, source)
            if available < quantity:
                self._fail(
                    "move_stock",
                    args,
                    f"insufficient stock: {sku} has {available} unit(s) in {source}, need {quantity}",
                    operation_id,
                )
            self._add(sku, source, -quantity)
            self._add(sku, destination, quantity)
            result = {
                "moved": quantity,
                "source_remaining": self._available(sku, source),
                "destination_total": self._available(sku, destination),
            }
            return dict(self._finish("move_stock", args, result, operation_id))

    def reserve_stock(
        self,
        sku: str,
        warehouse: str,
        quantity: int,
        order_id: str,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Reserve *quantity* units of *sku* in *warehouse* for *order_id*.

        Reserved units leave available stock.

        Returns:
            ``{"reserved": int, "order_id": str, "warehouse_remaining": int}``.
        """
        with self._lock:
            args = {
                "sku": sku,
                "warehouse": warehouse,
                "quantity": quantity,
                "order_id": order_id,
                "operation_id": operation_id,
            }
            replayed, cached = self._begin("reserve_stock", args, operation_id)
            if replayed:
                return dict(cached)
            self._maybe_fail_before("reserve_stock", args, operation_id)
            quantity = int(quantity)
            if quantity <= 0:
                self._fail("reserve_stock", args, "quantity must be positive", operation_id)
            available = self._available(sku, warehouse)
            if available < quantity:
                self._fail(
                    "reserve_stock",
                    args,
                    f"insufficient stock: {sku} has {available} unit(s) in {warehouse}, need {quantity}",
                    operation_id,
                )
            self._add(sku, warehouse, -quantity)
            lines = self._reservations.setdefault(order_id, {})
            lines[sku] = lines.get(sku, 0) + quantity
            result = {
                "reserved": quantity,
                "order_id": order_id,
                "warehouse_remaining": self._available(sku, warehouse),
            }
            return dict(self._finish("reserve_stock", args, result, operation_id))

    def cancel_order(self, order_id: str, warehouse: str, operation_id: str | None = None) -> dict[str, Any]:
        """Cancel *order_id*, returning every reserved unit to *warehouse*.

        Returns:
            ``{"released": {sku: units}}``.
        """
        with self._lock:
            args = {"order_id": order_id, "warehouse": warehouse, "operation_id": operation_id}
            replayed, cached = self._begin("cancel_order", args, operation_id)
            if replayed:
                return dict(cached)
            self._maybe_fail_before("cancel_order", args, operation_id)
            lines = self._reservations.get(order_id)
            if not lines:
                self._fail("cancel_order", args, f"unknown or empty order {order_id!r}", operation_id)
            released = dict(lines)
            for sku, qty in released.items():
                self._add(sku, warehouse, int(qty))
            self._reservations.pop(order_id, None)
            result = {"released": released}
            return dict(self._finish("cancel_order", args, result, operation_id))

    def restock(
        self,
        sku: str,
        warehouse: str,
        quantity: int,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Add *quantity* units of *sku* to *warehouse*, creating it if new.

        Returns:
            ``{"sku": str, "warehouse": str, "total": int}``.
        """
        with self._lock:
            args = {"sku": sku, "warehouse": warehouse, "quantity": quantity, "operation_id": operation_id}
            replayed, cached = self._begin("restock", args, operation_id)
            if replayed:
                return dict(cached)
            self._maybe_fail_before("restock", args, operation_id)
            quantity = int(quantity)
            if quantity <= 0:
                self._fail("restock", args, "quantity must be positive", operation_id)
            self._add(sku, warehouse, quantity)
            result = {"sku": sku, "warehouse": warehouse, "total": self._available(sku, warehouse)}
            return dict(self._finish("restock", args, result, operation_id))

    def set_price(self, sku: str, price: float, operation_id: str | None = None) -> dict[str, Any]:
        """Set the unit price of *sku*.

        Returns:
            ``{"sku": str, "price": float}``.
        """
        with self._lock:
            args = {"sku": sku, "price": price, "operation_id": operation_id}
            replayed, cached = self._begin("set_price", args, operation_id)
            if replayed:
                return dict(cached)
            self._maybe_fail_before("set_price", args, operation_id)
            value = float(price)
            if value < 0:
                self._fail("set_price", args, "price must not be negative", operation_id)
            self._prices[sku] = value
            result = {"sku": sku, "price": value}
            return dict(self._finish("set_price", args, result, operation_id))

    # ---- registry -----------------------------------------------------

    def as_tool_registry(self, *, include: set[str] | None = None) -> Any:
        """Expose the world's operations as a :class:`ToolRegistry`.

        Args:
            include: Optional subset of tool names.  Anything omitted is simply
                not registered, which is how a caller's allowed-tool list is
                enforced *before* the model ever sees a catalogue.
        """
        from ..agents.tools_schema import ToolRegistry

        registry = ToolRegistry()
        candidates = {
            "get_stock": self.get_stock,
            "list_inventory": self.list_inventory,
            "move_stock": self.move_stock,
            "reserve_stock": self.reserve_stock,
            "cancel_order": self.cancel_order,
            "restock": self.restock,
            "set_price": self.set_price,
        }
        for name, fn in candidates.items():
            if include is not None and name not in include:
                continue
            registry.register(
                fn,
                name=name,
                metadata={"is_write": name in self.MUTATING_TOOLS, "sandbox": True},
            )
        return registry
