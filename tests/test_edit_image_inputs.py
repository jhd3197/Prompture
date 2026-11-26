"""edit_image takes one picture or a sequence, bounded by max_edit_images."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from prompture.drivers.async_img_gen_base import AsyncImageGenDriver
from prompture.drivers.async_muapi_aggregator_driver import AsyncMuapiImageGenDriver
from prompture.drivers.img_gen_base import ImageGenDriver, edit_images_list
from prompture.drivers.muapi_aggregator_driver import MuapiImageGenDriver


def test_edit_images_list_wraps_one_picture():
    assert edit_images_list(b"a", 1) == [b"a"]
    assert edit_images_list(bytearray(b"a"), 1) == [b"a"]


def test_edit_images_list_takes_a_sequence():
    assert edit_images_list([b"a", b"b"], 2) == [b"a", b"b"]
    assert edit_images_list((b"a",), 1) == [b"a"]


def test_edit_images_list_rejects_empty_and_too_many():
    with pytest.raises(ValueError, match="at least one"):
        edit_images_list([], 1)
    with pytest.raises(ValueError, match="at most 1 image"):
        edit_images_list([b"a", b"b"], 1)


def test_async_base_has_edit_capabilities():
    assert AsyncImageGenDriver.supports_edit is False
    assert AsyncImageGenDriver.max_edit_images == 1
    assert ImageGenDriver.max_edit_images == 1
    with pytest.raises(NotImplementedError, match="does not support image editing"):
        asyncio.run(AsyncImageGenDriver().edit_image(b"a", "x", {}))


def test_single_image_aggregator_takes_a_one_item_list():
    d = MuapiImageGenDriver(api_key="k", model="seedream-edit")
    with (
        patch.object(MuapiImageGenDriver, "upload_file", return_value="https://host/in.png") as up,
        patch.object(MuapiImageGenDriver, "_run_image", return_value={"images": [], "meta": {}}),
    ):
        d.edit_image([b"rawpng"], "make it blue", {})
    assert up.call_args.args[0] == b"rawpng"


def test_single_image_aggregator_refuses_several():
    d = MuapiImageGenDriver(api_key="k", model="seedream-edit")
    with patch.object(MuapiImageGenDriver, "upload_file") as up, pytest.raises(ValueError, match="at most 1"):
        d.edit_image([b"a", b"b"], "x", {})
    up.assert_not_called()


def test_async_single_image_aggregator_takes_a_list_and_refuses_several():
    d = AsyncMuapiImageGenDriver(api_key="k", model="seedream-edit")
    with (
        patch.object(AsyncMuapiImageGenDriver, "upload_file", new=AsyncMock(return_value="https://h/i.png")) as up,
        patch.object(AsyncMuapiImageGenDriver, "_run_image", new=AsyncMock(return_value={"images": [], "meta": {}})),
    ):
        asyncio.run(d.edit_image([b"rawpng"], "x", {}))
        assert up.call_args.args[0] == b"rawpng"
        with pytest.raises(ValueError, match="at most 1"):
            asyncio.run(d.edit_image([b"a", b"b"], "x", {}))
