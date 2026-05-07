"""Testing utilities for Prompture drivers and third-party plugin drivers.

Use this module to verify that a driver class honors the Prompture
capability contract: every ``supports_*`` flag must correspond to an
overridden method.

Example for a third-party plugin author::

    from prompture.testing import assert_driver_capabilities

    def test_my_driver_contract():
        assert_driver_capabilities(MyDriver)
"""

from .driver_contract import (
    FLAG_TO_METHOD,
    assert_driver_capabilities,
    validate_driver_capabilities,
)

__all__ = [
    "FLAG_TO_METHOD",
    "assert_driver_capabilities",
    "validate_driver_capabilities",
]
