"""Built-in audio provider plugin: ElevenLabs, Cartesia, Deepgram, AssemblyAI."""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin


class AudioPlugin(ProviderPlugin):
    name = "audio_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        descs: list[ProviderDescriptor] = []

        el_kw = {"api_key": "elevenlabs_api_key", "endpoint": "elevenlabs_endpoint"}
        descs.append(
            ProviderDescriptor(
                name="elevenlabs",
                stt_sync=DriverSpec("elevenlabs_stt_driver.ElevenLabsSTTDriver", el_kw, "scribe_v1"),
                stt_async=DriverSpec(
                    "async_elevenlabs_stt_driver.AsyncElevenLabsSTTDriver",
                    el_kw,
                    "scribe_v1",
                ),
                tts_sync=DriverSpec(
                    "elevenlabs_tts_driver.ElevenLabsTTSDriver",
                    {**el_kw},
                    "elevenlabs_tts_model",
                ),
                tts_async=DriverSpec(
                    "async_elevenlabs_tts_driver.AsyncElevenLabsTTSDriver",
                    {**el_kw},
                    "elevenlabs_tts_model",
                ),
                display_name="ElevenLabs",
                is_configured_check="elevenlabs_api_key",
                models_dev_name="elevenlabs",
            )
        )

        cartesia_kw = {"api_key": "cartesia_api_key"}
        descs.append(
            ProviderDescriptor(
                name="cartesia",
                tts_sync=DriverSpec(
                    "cartesia_tts_driver.CartesiaTTSDriver",
                    cartesia_kw,
                    "cartesia_tts_model",
                ),
                tts_async=DriverSpec(
                    "async_cartesia_tts_driver.AsyncCartesiaTTSDriver",
                    cartesia_kw,
                    "cartesia_tts_model",
                ),
                display_name="Cartesia",
                is_configured_check="cartesia_api_key",
            )
        )

        deepgram_kw = {"api_key": "deepgram_api_key"}
        descs.append(
            ProviderDescriptor(
                name="deepgram",
                stt_sync=DriverSpec(
                    "deepgram_stt_driver.DeepgramSTTDriver",
                    deepgram_kw,
                    "deepgram_stt_model",
                ),
                stt_async=DriverSpec(
                    "async_deepgram_stt_driver.AsyncDeepgramSTTDriver",
                    deepgram_kw,
                    "deepgram_stt_model",
                ),
                tts_sync=DriverSpec(
                    "deepgram_tts_driver.DeepgramTTSDriver",
                    deepgram_kw,
                    "deepgram_tts_model",
                ),
                tts_async=DriverSpec(
                    "async_deepgram_tts_driver.AsyncDeepgramTTSDriver",
                    deepgram_kw,
                    "deepgram_tts_model",
                ),
                display_name="Deepgram",
                is_configured_check="deepgram_api_key",
            )
        )

        assemblyai_kw = {"api_key": "assemblyai_api_key"}
        descs.append(
            ProviderDescriptor(
                name="assemblyai",
                stt_sync=DriverSpec(
                    "assemblyai_stt_driver.AssemblyAISTTDriver",
                    assemblyai_kw,
                    "assemblyai_stt_model",
                ),
                stt_async=DriverSpec(
                    "async_assemblyai_stt_driver.AsyncAssemblyAISTTDriver",
                    assemblyai_kw,
                    "assemblyai_stt_model",
                ),
                display_name="AssemblyAI",
                is_configured_check="assemblyai_api_key",
            )
        )

        return descs
