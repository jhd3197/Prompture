"""Base class for Prompture provider plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..drivers.provider_descriptors import ProviderDescriptor


class ProviderPlugin(ABC):
    """Base class for Prompture provider plugins.

    Third-party packages export instances (or classes) of ``ProviderPlugin``
    subclasses via the ``prompture.providers`` entry-point group. Prompture
    discovers and registers them at import time.

    Example (third-party plugin)::

        # my_package/plugin.py
        from prompture.plugins import ProviderPlugin
        from prompture.drivers.provider_descriptors import (
            ProviderDescriptor,
            DriverSpec,
        )

        class FooPlugin(ProviderPlugin):
            name = "foo"
            version = "1.0.0"

            def descriptors(self):
                return [
                    ProviderDescriptor(
                        name="foo",
                        llm_sync=DriverSpec(
                            "my_package.driver.FooDriver",
                            {"api_key": "foo_api_key"},
                            "foo-default",
                        ),
                    ),
                ]

        # pyproject.toml of `prompture-foo`:
        #   [project.entry-points."prompture.providers"]
        #   foo = "my_package.plugin:FooPlugin"
    """

    name: str = ""
    version: str = ""

    @abstractmethod
    def descriptors(self) -> list[ProviderDescriptor]:
        """Return one or more :class:`ProviderDescriptor` instances this plugin contributes."""
