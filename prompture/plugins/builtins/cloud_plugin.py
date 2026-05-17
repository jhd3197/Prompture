"""Built-in cloud-provider plugin: Azure + AWS Bedrock."""

from __future__ import annotations

from typing import Any

from ...drivers.provider_descriptors import ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm


def _azure_is_configured(env: Any = None) -> bool:
    """Azure has a complex multi-backend configuration check."""
    import os

    from ...infra.settings import settings

    if env is not None:
        if env.resolve("azure_api_key") and env.resolve("azure_api_endpoint"):
            return True
    else:
        if (getattr(settings, "azure_api_key", None) or os.getenv("AZURE_API_KEY")) and (
            getattr(settings, "azure_api_endpoint", None) or os.getenv("AZURE_API_ENDPOINT")
        ):
            return True

    if settings.azure_claude_api_key or os.getenv("AZURE_CLAUDE_API_KEY"):
        return True
    if settings.azure_mistral_api_key or os.getenv("AZURE_MISTRAL_API_KEY"):
        return True

    try:
        from ...drivers.azure_config import (
            has_azure_config_resolver,
            has_registered_configs,
        )

        if has_registered_configs() or has_azure_config_resolver():
            return True
    except ImportError:
        pass

    return False


def _bedrock_is_configured(env: Any = None) -> bool:
    """Bedrock is configured if AWS creds are set or boto3 finds them in the chain."""
    import os

    from ...infra.settings import settings

    if env is not None:
        if env.resolve("aws_access_key_id") and env.resolve("aws_secret_access_key"):
            return True
    else:
        if (getattr(settings, "aws_access_key_id", None) or os.getenv("AWS_ACCESS_KEY_ID")) and (
            getattr(settings, "aws_secret_access_key", None) or os.getenv("AWS_SECRET_ACCESS_KEY")
        ):
            return True

    try:
        import boto3  # type: ignore[import-not-found]

        session = boto3.Session()
        return session.get_credentials() is not None
    except Exception:
        return False


class CloudPlugin(ProviderPlugin):
    name = "cloud_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        azure_kw = {
            "api_key": "azure_api_key",
            "endpoint": "azure_api_endpoint",
            "deployment_id": "azure_deployment_id",
            "claude_api_key": "azure_claude_api_key",
            "claude_endpoint": "azure_claude_endpoint",
            "mistral_api_key": "azure_mistral_api_key",
            "mistral_endpoint": "azure_mistral_endpoint",
        }
        azure_desc = ProviderDescriptor(
            name="azure",
            **_llm(
                "azure_driver",
                "AzureDriver",
                "async_azure_driver",
                "AsyncAzureDriver",
                azure_kw,
                "gpt-4o-mini",
            ),
            display_name="Azure",
            is_configured_fn=_azure_is_configured,
            models_dev_name="azure",
        )

        bedrock_kw = {
            "aws_access_key_id": "aws_access_key_id",
            "aws_secret_access_key": "aws_secret_access_key",  # nosec B105
            "aws_region": "aws_region",
        }
        bedrock_desc = ProviderDescriptor(
            name="bedrock",
            **_llm(
                "bedrock_driver",
                "BedrockDriver",
                "async_bedrock_driver",
                "AsyncBedrockDriver",
                bedrock_kw,
                "bedrock_model",
            ),
            display_name="AWS Bedrock",
            is_configured_fn=_bedrock_is_configured,
            list_models_kwargs=[
                ("aws_access_key_id", "aws_access_key_id", "AWS_ACCESS_KEY_ID"),
                ("aws_secret_access_key", "aws_secret_access_key", "AWS_SECRET_ACCESS_KEY"),
                ("aws_region", "aws_region", "AWS_REGION"),
            ],
            models_dev_name="amazon-bedrock",
        )

        return [azure_desc, bedrock_desc]
