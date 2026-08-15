"""Built-in image-focused provider plugin: Stability, Runway (multi-modality), Kling, Fal, Ideogram, BFL."""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin
from ._aliases import build_alias


class ImagePlugin(ProviderPlugin):
    name = "image_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        descs: list[ProviderDescriptor] = []

        # Stability AI
        stab_kw = {"api_key": "stability_api_key", "endpoint": "stability_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="stability",
                img_gen_sync=DriverSpec(
                    "stability_img_gen_driver.StabilityImageGenDriver",
                    stab_kw,
                    "stable-image-core",
                ),
                img_gen_async=DriverSpec(
                    "async_stability_img_gen_driver.AsyncStabilityImageGenDriver",
                    stab_kw,
                    "stable-image-core",
                ),
                display_name="Stability AI",
                is_configured_check="stability_api_key",
            )
        )

        # Runway (img + video + tts)
        runway_kw = {"api_key": "runway_api_key", "endpoint": "runway_endpoint"}
        runway_desc = ProviderDescriptor(
            name="runway",
            img_gen_sync=DriverSpec("runway_img_gen_driver.RunwayImageGenDriver", runway_kw, "gen4_image"),
            img_gen_async=DriverSpec(
                "async_runway_img_gen_driver.AsyncRunwayImageGenDriver",
                runway_kw,
                "gen4_image",
            ),
            video_gen_sync=DriverSpec("runway_video_gen_driver.RunwayVideoGenDriver", runway_kw, "gen4.5"),
            video_gen_async=DriverSpec(
                "async_runway_video_gen_driver.AsyncRunwayVideoGenDriver",
                runway_kw,
                "gen4.5",
            ),
            tts_sync=DriverSpec("runway_tts_driver.RunwayTTSDriver", runway_kw, "eleven_multilingual_v2"),
            tts_async=DriverSpec(
                "async_runway_tts_driver.AsyncRunwayTTSDriver",
                runway_kw,
                "eleven_multilingual_v2",
            ),
            display_name="Runway",
            is_configured_check="runway_api_key",
        )
        descs.append(runway_desc)

        # Kling
        kling_kw = {
            "access_key": "kling_access_key",
            "secret_key": "kling_secret_key",  # nosec B105 — settings attribute name
            "endpoint": "kling_endpoint",
        }
        descs.append(
            ProviderDescriptor(
                name="kling",
                img_gen_sync=DriverSpec(
                    "kling_img_gen_driver.KlingImageGenDriver",
                    kling_kw,
                    "kling_image_model",
                ),
                img_gen_async=DriverSpec(
                    "async_kling_img_gen_driver.AsyncKlingImageGenDriver",
                    kling_kw,
                    "kling_image_model",
                ),
                video_gen_sync=DriverSpec(
                    "kling_video_gen_driver.KlingVideoGenDriver",
                    kling_kw,
                    "kling_video_model",
                ),
                video_gen_async=DriverSpec(
                    "async_kling_video_gen_driver.AsyncKlingVideoGenDriver",
                    kling_kw,
                    "kling_video_model",
                ),
                display_name="Kling AI",
                is_configured_check="kling_access_key",
            )
        )

        # Fal
        fal_kw = {"api_key": "fal_api_key", "endpoint": "fal_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="fal",
                img_gen_sync=DriverSpec("fal_img_gen_driver.FalImageGenDriver", fal_kw, "fal_image_model"),
                img_gen_async=DriverSpec(
                    "async_fal_img_gen_driver.AsyncFalImageGenDriver",
                    fal_kw,
                    "fal_image_model",
                ),
                video_gen_sync=DriverSpec("fal_video_gen_driver.FalVideoGenDriver", fal_kw, "fal_video_model"),
                video_gen_async=DriverSpec(
                    "async_fal_video_gen_driver.AsyncFalVideoGenDriver",
                    fal_kw,
                    "fal_video_model",
                ),
                display_name="Fal.ai",
                is_configured_check="fal_api_key",
            )
        )

        # Muapi.ai (multi-modal aggregator: image + i2i edit + video + lipsync + music)
        muapi_kw = {"api_key": "muapi_api_key", "endpoint": "muapi_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="muapi",
                img_gen_sync=DriverSpec("muapi_aggregator_driver.MuapiImageGenDriver", muapi_kw, "muapi_image_model"),
                img_gen_async=DriverSpec(
                    "async_muapi_aggregator_driver.AsyncMuapiImageGenDriver",
                    muapi_kw,
                    "muapi_image_model",
                ),
                video_gen_sync=DriverSpec("muapi_aggregator_driver.MuapiVideoGenDriver", muapi_kw, "muapi_video_model"),
                video_gen_async=DriverSpec(
                    "async_muapi_aggregator_driver.AsyncMuapiVideoGenDriver",
                    muapi_kw,
                    "muapi_video_model",
                ),
                lipsync_sync=DriverSpec("muapi_aggregator_driver.MuapiLipsyncDriver", muapi_kw, "muapi_lipsync_model"),
                lipsync_async=DriverSpec(
                    "async_muapi_aggregator_driver.AsyncMuapiLipsyncDriver",
                    muapi_kw,
                    "muapi_lipsync_model",
                ),
                music_sync=DriverSpec("muapi_aggregator_driver.MuapiMusicGenDriver", muapi_kw, "muapi_music_model"),
                music_async=DriverSpec(
                    "async_muapi_aggregator_driver.AsyncMuapiMusicGenDriver",
                    muapi_kw,
                    "muapi_music_model",
                ),
                display_name="Muapi.ai",
                is_configured_check="muapi_api_key",
            )
        )

        # Ideogram
        ideogram_kw = {"api_key": "ideogram_api_key"}
        descs.append(
            ProviderDescriptor(
                name="ideogram",
                img_gen_sync=DriverSpec(
                    "ideogram_img_gen_driver.IdeogramImageGenDriver",
                    ideogram_kw,
                    "ideogram_image_model",
                ),
                img_gen_async=DriverSpec(
                    "async_ideogram_img_gen_driver.AsyncIdeogramImageGenDriver",
                    ideogram_kw,
                    "ideogram_image_model",
                ),
                display_name="Ideogram",
                is_configured_check="ideogram_api_key",
            )
        )

        # Black Forest Labs (BFL)
        bfl_kw = {"api_key": "bfl_api_key"}
        bfl_desc = ProviderDescriptor(
            name="bfl",
            img_gen_sync=DriverSpec("bfl_img_gen_driver.BFLImageGenDriver", bfl_kw, "bfl_image_model"),
            img_gen_async=DriverSpec(
                "async_bfl_img_gen_driver.AsyncBFLImageGenDriver",
                bfl_kw,
                "bfl_image_model",
            ),
            display_name="Black Forest Labs",
            is_configured_check="bfl_api_key",
        )
        descs.append(bfl_desc)

        # Aliases
        descs.append(build_alias("runwayml", runway_desc, {"img_gen", "video_gen", "tts"}))
        descs.append(build_alias("flux", bfl_desc, {"img_gen"}))

        return descs
