"""Tests for the plugin/custom driver registration system."""

import pytest

from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver
from prompture.drivers import (
    DRIVER_REGISTRY,
    get_driver_for_model,
)
from prompture.drivers.async_registry import (
    ASYNC_DRIVER_REGISTRY,
    get_async_driver_for_model,
)
from prompture.drivers.registry import (
    get_async_driver_factory,
    get_driver_factory,
    is_async_driver_registered,
    is_driver_registered,
    list_registered_async_drivers,
    list_registered_drivers,
    register_async_driver,
    register_driver,
    unregister_async_driver,
    unregister_driver,
)


class MockDriver(Driver):
    """A mock sync driver for testing."""

    def __init__(self, model=None, api_key=None):
        self.model = model or "default-mock-model"
        self.api_key = api_key

    def generate(self, prompt, options):
        return {
            "text": f"Mock response for {self.model}",
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.0,
                "raw_response": {},
            },
        }


class MockAsyncDriver(AsyncDriver):
    """A mock async driver for testing."""

    def __init__(self, model=None, api_key=None):
        self.model = model or "default-mock-model"
        self.api_key = api_key

    async def generate(self, prompt, options):
        return {
            "text": f"Mock async response for {self.model}",
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.0,
                "raw_response": {},
            },
        }


class TestSyncDriverRegistration:
    """Tests for sync driver registration."""

    def test_register_custom_driver(self):
        """Test registering a custom sync driver."""

        # Define a factory function
        def mock_factory(model=None):
            return MockDriver(model=model, api_key="test-key")

        # Register it
        register_driver("mock_provider", mock_factory, overwrite=True)

        # Verify it's registered
        assert is_driver_registered("mock_provider")
        assert "mock_provider" in list_registered_drivers()

        # Get the factory and create a driver
        factory = get_driver_factory("mock_provider")
        driver = factory("custom-model")

        assert isinstance(driver, MockDriver)
        assert driver.model == "custom-model"
        assert driver.api_key == "test-key"

        # Clean up
        unregister_driver("mock_provider")

    def test_get_driver_for_model_with_custom_driver(self):
        """Test that get_driver_for_model works with custom drivers."""
        register_driver(
            "test_custom",
            lambda model=None: MockDriver(model=model),
            overwrite=True,
        )

        driver = get_driver_for_model("test_custom/my-model")
        assert isinstance(driver, MockDriver)
        assert driver.model == "my-model"

        # Clean up
        unregister_driver("test_custom")

    def test_register_duplicate_raises_error(self):
        """Test that registering a duplicate driver raises an error."""
        register_driver(
            "dup_test",
            lambda model=None: MockDriver(model=model),
            overwrite=True,
        )

        with pytest.raises(ValueError, match="already registered"):
            register_driver(
                "dup_test",
                lambda model=None: MockDriver(model=model),
                overwrite=False,
            )

        # Clean up
        unregister_driver("dup_test")

    def test_register_with_overwrite(self):
        """Test that overwrite=True allows replacing a driver."""
        # Register first version
        register_driver(
            "overwrite_test",
            lambda model=None: MockDriver(model="v1"),
            overwrite=True,
        )

        # Overwrite with second version
        register_driver(
            "overwrite_test",
            lambda model=None: MockDriver(model="v2"),
            overwrite=True,
        )

        factory = get_driver_factory("overwrite_test")
        driver = factory()
        assert driver.model == "v2"

        # Clean up
        unregister_driver("overwrite_test")

    def test_unregister_driver(self):
        """Test unregistering a driver."""
        register_driver(
            "unreg_test",
            lambda model=None: MockDriver(model=model),
            overwrite=True,
        )

        assert is_driver_registered("unreg_test")
        result = unregister_driver("unreg_test")
        assert result is True
        assert not is_driver_registered("unreg_test")

        # Unregistering again returns False
        result = unregister_driver("unreg_test")
        assert result is False

    def test_get_unknown_driver_raises_error(self):
        """Test that getting an unknown driver raises an error."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_driver_factory("nonexistent_provider_xyz")

    def test_driver_name_case_insensitive(self):
        """Test that driver names are case-insensitive."""
        register_driver(
            "CaseTEST",
            lambda model=None: MockDriver(model=model),
            overwrite=True,
        )

        assert is_driver_registered("casetest")
        assert is_driver_registered("CASETEST")
        assert is_driver_registered("CaseTEST")

        # Clean up
        unregister_driver("casetest")

    def test_builtin_drivers_registered(self):
        """Test that built-in drivers are registered."""
        drivers = list_registered_drivers()

        # Check some of the built-in drivers
        assert "openai" in drivers
        assert "ollama" in drivers
        assert "claude" in drivers
        assert "google" in drivers
        assert "groq" in drivers


class TestAsyncDriverRegistration:
    """Tests for async driver registration."""

    def test_register_custom_async_driver(self):
        """Test registering a custom async driver."""

        def mock_async_factory(model=None):
            return MockAsyncDriver(model=model, api_key="test-key")

        register_async_driver("mock_async_provider", mock_async_factory, overwrite=True)

        assert is_async_driver_registered("mock_async_provider")
        assert "mock_async_provider" in list_registered_async_drivers()

        factory = get_async_driver_factory("mock_async_provider")
        driver = factory("custom-model")

        assert isinstance(driver, MockAsyncDriver)
        assert driver.model == "custom-model"

        # Clean up
        unregister_async_driver("mock_async_provider")

    def test_get_async_driver_for_model_with_custom_driver(self):
        """Test that get_async_driver_for_model works with custom drivers."""
        register_async_driver(
            "test_async_custom",
            lambda model=None: MockAsyncDriver(model=model),
            overwrite=True,
        )

        driver = get_async_driver_for_model("test_async_custom/my-model")
        assert isinstance(driver, MockAsyncDriver)
        assert driver.model == "my-model"

        # Clean up
        unregister_async_driver("test_async_custom")

    def test_unregister_async_driver(self):
        """Test unregistering an async driver."""
        register_async_driver(
            "async_unreg_test",
            lambda model=None: MockAsyncDriver(model=model),
            overwrite=True,
        )

        assert is_async_driver_registered("async_unreg_test")
        result = unregister_async_driver("async_unreg_test")
        assert result is True
        assert not is_async_driver_registered("async_unreg_test")

    def test_builtin_async_drivers_registered(self):
        """Test that built-in async drivers are registered."""
        drivers = list_registered_async_drivers()

        assert "openai" in drivers
        assert "ollama" in drivers
        assert "claude" in drivers


class TestRegistryBackwardsCompatibility:
    """Tests for backwards compatibility with DRIVER_REGISTRY dict."""

    def test_driver_registry_dict_accessible(self):
        """Test that DRIVER_REGISTRY dict is still accessible."""
        assert isinstance(DRIVER_REGISTRY, dict)
        assert "openai" in DRIVER_REGISTRY
        assert "ollama" in DRIVER_REGISTRY

    def test_async_driver_registry_dict_accessible(self):
        """Test that ASYNC_DRIVER_REGISTRY dict is still accessible."""
        assert isinstance(ASYNC_DRIVER_REGISTRY, dict)
        assert "openai" in ASYNC_DRIVER_REGISTRY
        assert "ollama" in ASYNC_DRIVER_REGISTRY

    def test_registry_dict_reflects_registrations(self):
        """Test that registering a driver updates the dict."""
        register_driver(
            "compat_test",
            lambda model=None: MockDriver(model=model),
            overwrite=True,
        )

        # The registry dict should contain the new driver
        assert "compat_test" in DRIVER_REGISTRY

        # Clean up
        unregister_driver("compat_test")
        assert "compat_test" not in DRIVER_REGISTRY


class TestDriverInterface:
    """Tests verifying the driver interface contract."""

    def test_custom_driver_generate_returns_correct_format(self):
        """Test that a custom driver returns the correct format."""
        register_driver(
            "interface_test",
            lambda model=None: MockDriver(model=model),
            overwrite=True,
        )

        driver = get_driver_for_model("interface_test/test-model")
        result = driver.generate("Test prompt", {})

        # Verify response structure
        assert "text" in result
        assert "meta" in result
        assert "prompt_tokens" in result["meta"]
        assert "completion_tokens" in result["meta"]
        assert "total_tokens" in result["meta"]
        assert "cost" in result["meta"]
        assert "raw_response" in result["meta"]

        # Clean up
        unregister_driver("interface_test")

    @pytest.mark.asyncio
    async def test_custom_async_driver_generate_returns_correct_format(self):
        """Test that a custom async driver returns the correct format."""
        register_async_driver(
            "async_interface_test",
            lambda model=None: MockAsyncDriver(model=model),
            overwrite=True,
        )

        driver = get_async_driver_for_model("async_interface_test/test-model")
        result = await driver.generate("Test prompt", {})

        # Verify response structure
        assert "text" in result
        assert "meta" in result
        assert "prompt_tokens" in result["meta"]
        assert "completion_tokens" in result["meta"]
        assert "total_tokens" in result["meta"]
        assert "cost" in result["meta"]
        assert "raw_response" in result["meta"]

        # Clean up
        unregister_async_driver("async_interface_test")
