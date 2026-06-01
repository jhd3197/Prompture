"""Provider-agnostic image prompt-engineering helpers.

These shape prompts and normalized options *before* any image driver is called,
so they behave the same across every backend (OpenAI, Google, Stability,
Ideogram, Grok, Fal, BFL, Runway, Kling, Vertex, ...):

- :class:`StylePreset` + a curated default library — reusable look-and-feel.
- :func:`enhance_image_prompt` — LLM expansion of a terse prompt.
- :func:`compose_negative_prompt` — native negative param where supported,
  otherwise folded into the prompt.
- :func:`plan_image_set` — art-director decomposition of a brief into a
  cohesive set of standalone image prompts.
"""

from .enhance import aenhance_image_prompt, enhance_image_prompt
from .negative import compose_negative_prompt, model_supports_negative_prompt
from .set_planner import (
    ImageSetPlan,
    ImageSpec,
    aplan_image_set,
    plan_image_set,
)
from .style_presets import (
    DEFAULT_STYLE_PRESETS,
    StylePreset,
    get_style_preset,
    list_style_presets,
    register_style_preset,
)

__all__ = [
    "DEFAULT_STYLE_PRESETS",
    "ImageSetPlan",
    "ImageSpec",
    "StylePreset",
    "aenhance_image_prompt",
    "aplan_image_set",
    "compose_negative_prompt",
    "enhance_image_prompt",
    "get_style_preset",
    "list_style_presets",
    "model_supports_negative_prompt",
    "plan_image_set",
    "register_style_preset",
]
