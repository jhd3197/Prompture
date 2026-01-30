#!/usr/bin/env python3
"""Vision support example — sending images to LLMs via Prompture.

Demonstrates:
1. Sending an image file to Conversation.ask()
2. Vision-based structured extraction via extract_with_model() with images
3. Multi-turn conversation referencing earlier images

Usage:
    python examples/vision_example.py

Requires:
    - A valid API key for a vision-capable provider (OpenAI, Claude, Google, etc.)
    - An image file to analyze (or uses a URL)
"""

from pydantic import BaseModel, Field

from prompture import Conversation, make_image

# ── Section 1: Basic image description ─────────────────────────────────────

print("=" * 60)
print("Section 1: Basic Image Description")
print("=" * 60)

conv = Conversation(
    "openai/gpt-4o",
    system_prompt="You are a helpful assistant that describes images.",
)

# You can pass images as URLs, file paths, bytes, or base64 strings.
# Here we use a publicly available test image URL.
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

response = conv.ask(
    "What do you see in this image? Be concise.",
    images=[image_url],
)
print(f"Response: {response}\n")


# ── Section 2: Structured extraction from images ───────────────────────────

print("=" * 60)
print("Section 2: Structured Extraction from Images")
print("=" * 60)


class ImageDescription(BaseModel):
    """Structured description extracted from an image."""

    subject: str = Field(description="Main subject of the image")
    setting: str = Field(description="The setting or environment")
    colors: list[str] = Field(description="Dominant colors in the image")
    mood: str = Field(description="Overall mood or feeling")
    objects: list[str] = Field(description="Notable objects visible")


conv2 = Conversation(
    "openai/gpt-4o",
    system_prompt="You are a precise image analyzer.",
)

result = conv2.extract_with_model(
    ImageDescription,
    "Analyze the following image in detail:",
    images=[image_url],
)

model = result["model"]
print(f"Subject:  {model.subject}")
print(f"Setting:  {model.setting}")
print(f"Colors:   {', '.join(model.colors)}")
print(f"Mood:     {model.mood}")
print(f"Objects:  {', '.join(model.objects)}")
print()


# ── Section 3: Multi-turn with images ──────────────────────────────────────

print("=" * 60)
print("Section 3: Multi-turn Conversation with Images")
print("=" * 60)

conv3 = Conversation(
    "openai/gpt-4o",
    system_prompt="You are a helpful assistant.",
)

# First turn: send an image
r1 = conv3.ask("Remember this image for later questions.", images=[image_url])
print(f"Turn 1: {r1}")

# Second turn: ask about the image without re-sending it
r2 = conv3.ask("What animal was in the image you just saw?")
print(f"Turn 2: {r2}")

# Third turn: follow-up
r3 = conv3.ask("What breed might it be?")
print(f"Turn 3: {r3}")

print(f"\nUsage: {conv3.usage_summary()}")


# ── Section 4: Using make_image() to prepare images ───────────────────────

print("\n" + "=" * 60)
print("Section 4: Image Preparation with make_image()")
print("=" * 60)

# make_image() auto-detects the source type:
#   - bytes → base64-encode
#   - str URL → store URL reference
#   - str path → read file from disk
#   - ImageContent → pass through

img = make_image(image_url)
print(f"Source type: {img.source_type}")
print(f"Media type:  {img.media_type}")
print(f"Has URL:     {img.url is not None}")
print(f"Has data:    {bool(img.data)}")

# If you have a local file:
# img_file = make_image(Path("photo.png"))
# img_bytes = make_image(open("photo.png", "rb").read())
