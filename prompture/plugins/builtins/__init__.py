"""Built-in Prompture provider plugins."""

from __future__ import annotations

from ..base import ProviderPlugin
from .anthropic_plugin import AnthropicPlugin
from .audio_plugin import AudioPlugin
from .cloud_plugin import CloudPlugin
from .embedding_rerank_plugin import EmbeddingRerankPlugin
from .extras_plugin import ExtrasPlugin
from .google_plugin import GooglePlugin
from .image_plugin import ImagePlugin
from .local_plugin import LocalPlugin
from .openai_compatible_plugin import OpenAICompatiblePlugin
from .openai_plugin import OpenAIPlugin
from .video_plugin import VideoPlugin

BUILTIN_PLUGINS: list[ProviderPlugin] = [
    OpenAIPlugin(),
    AnthropicPlugin(),
    GooglePlugin(),
    CloudPlugin(),
    OpenAICompatiblePlugin(),
    LocalPlugin(),
    AudioPlugin(),
    ImagePlugin(),
    VideoPlugin(),
    EmbeddingRerankPlugin(),
    ExtrasPlugin(),
]

__all__ = [
    "BUILTIN_PLUGINS",
    "AnthropicPlugin",
    "AudioPlugin",
    "CloudPlugin",
    "EmbeddingRerankPlugin",
    "ExtrasPlugin",
    "GooglePlugin",
    "ImagePlugin",
    "LocalPlugin",
    "OpenAICompatiblePlugin",
    "OpenAIPlugin",
    "VideoPlugin",
]
