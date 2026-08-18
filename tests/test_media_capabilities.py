"""Tests for the cross-provider media schema KB (media_capabilities.py)."""

from __future__ import annotations

import pytest

from prompture.drivers.media_capabilities import (
    MediaModelInfo,
    ParamSpec,
    get_aspect_ratios,
    get_audio_model_controls,
    get_durations,
    get_max_images,
    get_model_schema,
    get_models_by_modality,
    get_models_by_op,
    get_video_model_controls,
    register_media_model,
)


class TestParamSpec:
    def test_to_json_schema_enum_and_range(self):
        spec = ParamSpec("width", "int", "W", default=1024, min_value=128, max_value=2048)
        js = spec.to_json_schema()
        assert js["type"] == "integer"
        assert js["minimum"] == 128 and js["maximum"] == 2048 and js["default"] == 1024

    def test_enum_rendered(self):
        spec = ParamSpec("aspect_ratio", enum=["1:1", "16:9"])
        assert spec.to_json_schema()["enum"] == ["1:1", "16:9"]


class TestLookup:
    def test_get_model_schema_provider_model(self):
        info = get_model_schema("muapi/nano-banana")
        assert info is not None
        assert info.modality == "image" and info.op == "text_to_image"

    def test_get_model_schema_bare_id(self):
        info = get_model_schema("nano-banana")
        assert info is not None and info.provider == "muapi"

    def test_unknown_model_returns_none(self):
        assert get_model_schema("muapi/does-not-exist") is None

    def test_runway_bridge(self):
        info = get_model_schema("runway/gen4.5")
        assert info is not None
        assert info.provider == "runway"
        assert info.op == "text_to_video"
        assert info.modality == "video"

    def test_runwayml_alias_bridge(self):
        assert get_model_schema("runwayml/gen4_aleph").op == "video_to_video"


class TestAccessors:
    def test_aspect_ratios(self):
        assert "16:9" in get_aspect_ratios("muapi/nano-banana")

    def test_durations(self):
        assert get_durations("muapi/kling-video-v2-1-image-to-video") == [5, 10]

    def test_max_images(self):
        assert get_max_images("muapi/seedream-edit") == 14
        assert get_max_images("muapi/nano-banana") == 1

    def test_unknown_accessors_empty(self):
        assert get_aspect_ratios("foo/bar") == []
        assert get_durations("foo/bar") == []
        assert get_max_images("foo/bar") == 1


class TestByQuery:
    def test_by_modality(self):
        vids = get_models_by_modality("video")
        assert "muapi/kling-video-v2-1" in vids
        assert all("/" in k for k in vids)

    def test_by_op(self):
        assert "muapi/kling-video-v2-1-image-to-video" in get_models_by_op("image_to_video")

    def test_by_op_unknown_raises(self):
        with pytest.raises(ValueError):
            get_models_by_op("teleport")


class TestControls:
    def test_video_controls(self):
        c = get_video_model_controls("muapi/kling-video-v2-1-image-to-video")
        assert c["durations"] == [5, 10]
        assert c["supports_image_input"] is True
        assert c["op"] == "image_to_video"

    def test_video_controls_unknown(self):
        c = get_video_model_controls("foo/bar")
        assert c["op"] is None and c["aspect_ratios"] == []

    def test_audio_controls(self):
        c = get_audio_model_controls("muapi/suno-create-music")
        assert c["op"] == "music"
        assert "model" in c["params"]


class TestRegistration:
    def test_register_and_lookup(self):
        info = MediaModelInfo(
            "unit-test-model",
            "muapi",
            "image",
            "text_to_image",
            inputs={"aspect_ratio": ParamSpec("aspect_ratio", enum=["1:1"])},
        )
        register_media_model(info)
        assert get_model_schema("muapi/unit-test-model") is info

    def test_register_no_overwrite_raises(self):
        info = MediaModelInfo("dup-model", "muapi", "image", "text_to_image")
        register_media_model(info)
        with pytest.raises(ValueError):
            register_media_model(info, overwrite=False)

    def test_to_dict_serializes_inputs(self):
        d = get_model_schema("muapi/flux-dev").to_dict()
        assert d["inputs"]["width"]["type"] == "integer"
        assert d["provider"] == "muapi"
