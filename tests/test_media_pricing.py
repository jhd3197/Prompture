"""Tests for media pricing + the estimate_media_cost quote (G16)."""

from __future__ import annotations

from prompture.infra.media_pricing import (
    estimate_media_cost,
    get_media_rate,
    register_media_rate,
)


class TestGetMediaRate:
    def test_exact_key(self):
        assert get_media_rate("muapi/nano-banana") == {"per_image": 0.039}

    def test_case_insensitive(self):
        assert get_media_rate("MUAPI/Nano-Banana") == {"per_image": 0.039}

    def test_bare_id_match(self):
        assert get_media_rate("seedream-edit") == {"per_image": 0.03}

    def test_unknown_returns_none(self):
        assert get_media_rate("muapi/no-such-model") is None


class TestEstimateCost:
    def test_per_image_times_n(self):
        assert estimate_media_cost("muapi/nano-banana", n=3) == round(0.039 * 3, 6)

    def test_per_second_times_duration(self):
        assert estimate_media_cost("muapi/kling-video-v2-1", duration_seconds=5) == round(0.10 * 5, 6)

    def test_per_second_via_duration_alias(self):
        assert estimate_media_cost("runway/veo3", duration=2) == round(0.40 * 2, 6)

    def test_per_character(self):
        assert estimate_media_cost("elevenlabs/eleven_multilingual_v2", characters=1000) == round(0.00003 * 1000, 6)

    def test_per_job(self):
        assert estimate_media_cost("muapi/suno-create-music") == 0.10

    def test_unknown_model_zero(self):
        assert estimate_media_cost("muapi/unknown", n=5) == 0.0

    def test_params_dict_form(self):
        assert estimate_media_cost("muapi/nano-banana", {"n": 2}) == round(0.039 * 2, 6)

    def test_params_and_overrides_merge(self):
        # kwargs override params
        assert estimate_media_cost("muapi/nano-banana", {"n": 2}, n=4) == round(0.039 * 4, 6)

    def test_resolution_specific_rate(self):
        register_media_rate("test/res-video", {"per_second_by_resolution": {"720p": 0.07, "1080p": 0.12}})
        assert estimate_media_cost("test/res-video", duration_seconds=4, resolution="1080p") == round(0.12 * 4, 6)


class TestRegisterMediaRate:
    def test_runtime_registration_and_precedence(self):
        register_media_rate("muapi/nano-banana", {"per_image": 1.23})
        try:
            assert get_media_rate("muapi/nano-banana") == {"per_image": 1.23}
            assert estimate_media_cost("muapi/nano-banana", n=2) == round(1.23 * 2, 6)
        finally:
            # restore by clearing the runtime override
            from prompture.infra import media_pricing

            media_pricing._runtime_rates.pop("muapi/nano-banana", None)
        assert get_media_rate("muapi/nano-banana") == {"per_image": 0.039}


def test_top_level_export():
    import prompture

    assert prompture.estimate_media_cost("muapi/nano-banana", n=1) == 0.039
