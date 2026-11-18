"""OpenAIImageGenDriver.edit_image: one picture or several, and the model it names."""

import base64
from types import SimpleNamespace

import pytest

from prompture.drivers.openai_img_gen_driver import OpenAIImageGenDriver

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class _Images:
    def __init__(self):
        self.kwargs = None

    def edit(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(PNG).decode(), url=None)])


def _driver(model="dall-e-3"):
    d = OpenAIImageGenDriver(api_key="sk-test", model=model)
    d.client = SimpleNamespace(images=_Images())
    return d


def test_one_picture_is_sent_as_one_file():
    d = _driver()
    d.edit_image(PNG, "make it blue", {})
    image = d.client.images.kwargs["image"]
    assert isinstance(image, tuple) and image[0] == "image1.png"


def test_several_pictures_are_sent_as_a_list():
    d = _driver()
    d.edit_image([PNG, PNG], "put the product from the second image into the first", {})
    image = d.client.images.kwargs["image"]
    assert [f[0] for f in image] == ["image1.png", "image2.png"]


def test_too_many_or_no_pictures_is_an_error():
    d = _driver()
    with pytest.raises(ValueError, match="at most 16"):
        d.edit_image([PNG] * 17, "x", {})
    with pytest.raises(ValueError, match="at least one"):
        d.edit_image([], "x", {})


def test_the_model_comes_from_options_then_the_driver():
    d = _driver("gpt-image-2")
    d.edit_image(PNG, "x", {})
    assert d.client.images.kwargs["model"] == "gpt-image-2"
    d.edit_image(PNG, "x", {"model": "gpt-image-1.5"})
    assert d.client.images.kwargs["model"] == "gpt-image-1.5"


def test_a_dalle_driver_still_edits_on_gpt_image():
    d = _driver("dall-e-3")
    d.edit_image(PNG, "x", {})
    assert d.client.images.kwargs["model"] == "gpt-image-1"


def test_only_the_openai_driver_advertises_several():
    from prompture.drivers.img_gen_base import ImageGenDriver

    assert ImageGenDriver.max_edit_images == 1
    assert OpenAIImageGenDriver.max_edit_images == 16
