"""Built-in Google (Gemini + Vertex AI) provider plugin."""

from __future__ import annotations

from typing import Any

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm, build_alias


def _google_vertexai_is_configured(env: Any = None) -> bool:
    """Vertex AI accepts either API key (Gemini) or project_id (Gemini + Claude)."""
    import os

    from ...infra.settings import settings

    if env is not None:
        return bool(env.resolve("google_vertex_api_key") or env.resolve("google_vertex_project_id"))
    return bool(
        getattr(settings, "google_vertex_api_key", None)
        or os.getenv("GOOGLE_VERTEX_API_KEY")
        or getattr(settings, "google_vertex_project_id", None)
        or os.getenv("GOOGLE_VERTEX_PROJECT_ID")
    )


class GooglePlugin(ProviderPlugin):
    name = "google_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        google_kw = {"api_key": "google_api_key"}
        google_desc = ProviderDescriptor(
            name="google",
            **_llm(
                "google_driver",
                "GoogleDriver",
                "async_google_driver",
                "AsyncGoogleDriver",
                google_kw,
                "google_model",
            ),
            img_gen_sync=DriverSpec(
                "google_img_gen_driver.GoogleImageGenDriver",
                google_kw,
                "imagen-3.0-generate-002",
            ),
            img_gen_async=DriverSpec(
                "async_google_img_gen_driver.AsyncGoogleImageGenDriver",
                google_kw,
                "imagen-3.0-generate-002",
            ),
            display_name="Google Gemini",
            is_configured_check="google_api_key",
            list_models_kwargs=[("api_key", "google_api_key", "GOOGLE_API_KEY")],
            models_dev_name="google",
        )

        vertex_kw = {
            "api_key": "google_vertex_api_key",
            "project_id": "google_vertex_project_id",
            "location": "google_vertex_location",
            "access_token": "google_vertex_access_token",  # nosec B105
        }
        # Image gen uses google-genai with vertexai=True; access_token isn't a
        # constructor arg for the image driver, so use a trimmed kwarg map.
        vertex_img_kw = {
            "api_key": "google_vertex_api_key",
            "project_id": "google_vertex_project_id",
            "location": "google_vertex_location",
        }
        vertex_desc = ProviderDescriptor(
            name="google_vertexai",
            **_llm(
                "google_vertexai_driver",
                "GoogleVertexAIDriver",
                "async_google_vertexai_driver",
                "AsyncGoogleVertexAIDriver",
                vertex_kw,
                "google_vertex_model",
            ),
            img_gen_sync=DriverSpec(
                "vertex_img_gen_driver.VertexImageGenDriver",
                vertex_img_kw,
                "gemini-2.5-flash-image",
            ),
            display_name="Google Vertex AI",
            is_configured_fn=_google_vertexai_is_configured,
            list_models_kwargs=[
                ("api_key", "google_vertex_api_key", "GOOGLE_VERTEX_API_KEY"),
                ("project_id", "google_vertex_project_id", "GOOGLE_VERTEX_PROJECT_ID"),
                ("location", "google_vertex_location", "GOOGLE_VERTEX_LOCATION"),
            ],
            models_dev_name="google",
        )

        return [
            google_desc,
            vertex_desc,
            build_alias("gemini", google_desc, {"llm", "img_gen"}),
            build_alias("vertex", vertex_desc, {"llm", "img_gen"}),
            build_alias("vertexai", vertex_desc, {"llm", "img_gen"}),
            # The dashboard (and OpenAI-style "provider/model" ids) use the
            # "vertex_ai" prefix — alias it for both LLM and image gen.
            build_alias("vertex_ai", vertex_desc, {"llm", "img_gen"}),
        ]
