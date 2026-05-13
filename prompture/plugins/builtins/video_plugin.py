"""Built-in video provider plugin: MiniMax (LLM + video), Luma, Pika."""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import _llm, build_alias


class VideoPlugin(ProviderPlugin):
    name = "video_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        descs: list[ProviderDescriptor] = []

        # MiniMax / Hailuo
        mm_kw = {"api_key": "minimax_api_key", "endpoint": "minimax_endpoint"}
        minimax_desc = ProviderDescriptor(
            name="minimax",
            **_llm(
                "minimax_driver",
                "MiniMaxDriver",
                "async_minimax_driver",
                "AsyncMiniMaxDriver",
                mm_kw,
                "minimax_model",
            ),
            video_gen_sync=DriverSpec(
                "minimax_video_gen_driver.MiniMaxVideoGenDriver",
                mm_kw,
                "minimax_video_model",
            ),
            video_gen_async=DriverSpec(
                "async_minimax_video_gen_driver.AsyncMiniMaxVideoGenDriver",
                mm_kw,
                "minimax_video_model",
            ),
            display_name="MiniMax",
            is_configured_check="minimax_api_key",
            list_models_kwargs=[("api_key", "minimax_api_key", "MINIMAX_API_KEY")],
            models_dev_name="minimax",
        )
        descs.append(minimax_desc)

        # Luma AI
        luma_kw = {"api_key": "luma_api_key"}
        luma_desc = ProviderDescriptor(
            name="luma",
            video_gen_sync=DriverSpec("luma_video_gen_driver.LumaVideoGenDriver", luma_kw, "luma_video_model"),
            video_gen_async=DriverSpec(
                "async_luma_video_gen_driver.AsyncLumaVideoGenDriver",
                luma_kw,
                "luma_video_model",
            ),
            display_name="Luma AI",
            is_configured_check="luma_api_key",
            list_models_kwargs=[("api_key", "luma_api_key", "LUMA_API_KEY")],
        )
        descs.append(luma_desc)

        # Pika Labs
        pika_kw = {"api_key": "pika_api_key"}
        descs.append(
            ProviderDescriptor(
                name="pika",
                video_gen_sync=DriverSpec(
                    "pika_video_gen_driver.PikaVideoGenDriver",
                    pika_kw,
                    "pika_video_model",
                ),
                video_gen_async=DriverSpec(
                    "async_pika_video_gen_driver.AsyncPikaVideoGenDriver",
                    pika_kw,
                    "pika_video_model",
                ),
                display_name="Pika Labs",
                is_configured_check="pika_api_key",
                list_models_kwargs=[("api_key", "pika_api_key", "PIKA_API_KEY")],
            )
        )

        # Aliases
        descs.append(build_alias("hailuo", minimax_desc, {"video_gen"}))
        descs.append(build_alias("dream-machine", luma_desc, {"video_gen"}))
        descs.append(build_alias("lumalabs", luma_desc, {"video_gen"}))

        return descs
