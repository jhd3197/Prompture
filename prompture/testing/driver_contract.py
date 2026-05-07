"""Driver capability contract validation.

For every Prompture driver class (sync or async), each ``supports_*``
class attribute must pair with an overridden corresponding method.  The
functions in this module check that contract for any driver class —
built-in or third-party plugin.
"""

from __future__ import annotations

from ..drivers.async_base import AsyncDriver
from ..drivers.base import Driver

# Capability flag → method that must be overridden when the flag is True.
# Limited to flags whose support is determined by overriding a single
# method.  ``json_mode`` / ``json_schema`` / ``vision`` are negotiated
# through option flags inside the ``generate``-family methods rather than
# by overriding a separate method, so they are excluded from this contract.
FLAG_TO_METHOD: dict[str, str] = {
    "supports_streaming": "generate_messages_stream",
    "supports_tool_use": "generate_messages_with_tools",
    "supports_messages": "generate_messages",
}


def _base_for(cls: type) -> type:
    if issubclass(cls, AsyncDriver):
        return AsyncDriver
    if issubclass(cls, Driver):
        return Driver
    raise TypeError(f"{cls.__name__} is not a Driver or AsyncDriver subclass")


def validate_driver_capabilities(cls: type) -> list[str]:
    """Return a list of contract violations for *cls* (empty == passes).

    The check is bidirectional: ``supports_X = True`` iff the
    corresponding method is overridden from the base class.  Each
    violation message names the flag, the method, and which direction
    is wrong.
    """
    base = _base_for(cls)
    violations: list[str] = []
    for flag, method_name in FLAG_TO_METHOD.items():
        declared = bool(getattr(cls, flag, False))
        cls_method = getattr(cls, method_name, None)
        base_method = getattr(base, method_name, None)
        implemented = cls_method is not None and cls_method is not base_method

        if declared and not implemented:
            violations.append(
                f"{flag}=True but {method_name} not overridden — "
                f"auto-strategy will recommend an unsupported feature"
            )
        elif implemented and not declared:
            violations.append(
                f"{method_name} overridden but {flag}=False — "
                f"capability is hidden from auto-strategy selection"
            )
    return violations


def assert_driver_capabilities(cls: type) -> None:
    """Assert that *cls* satisfies the driver capability contract.

    Raises :class:`AssertionError` listing every violation when the
    contract is broken.  Use this in your test suite to lock in driver
    correctness — Prompture's own contract test does the same.
    """
    violations = validate_driver_capabilities(cls)
    if violations:
        raise AssertionError(
            "{} fails driver capability contract:\n  - {}".format(
                cls.__name__, "\n  - ".join(violations)
            )
        )
