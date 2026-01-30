#!/usr/bin/env python3
"""Conversation persistence example.

Demonstrates saving, loading, and searching conversations using
Prompture's persistence features:

1. In-memory export/import round-trip
2. File-based save/load
3. SQLite ConversationStore with tags and search
4. Auto-save configuration
5. Restoring a conversation and continuing the dialog

Requirements:
    pip install prompture
    # Set OLLAMA_ENDPOINT or OPENAI_API_KEY in .env
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

# ── Section 1: In-memory export/import ──────────────────────────────

print("=" * 60)
print("Section 1: In-memory export / import")
print("=" * 60)

from prompture import Conversation

conv = Conversation(
    model_name="ollama/llama3.1:8b",
    system_prompt="You are a helpful assistant.",
    tags=["demo", "persistence"],
)

# Simulate a conversation (add_context avoids needing a live LLM)
conv.add_context("user", "What is the capital of France?")
conv.add_context("assistant", "The capital of France is Paris.")
conv.add_context("user", "And Germany?")
conv.add_context("assistant", "The capital of Germany is Berlin.")

# Export to dict
exported = conv.export()
print(f"Conversation ID: {conv.conversation_id}")
print(f"Tags: {conv.tags}")
print(f"Export version: {exported['version']}")
print(f"Messages: {len(exported['messages'])}")
print(f"Model: {exported['model_name']}")
print()

# Import from dict (reconstructs a new Conversation)
restored = Conversation.from_export(exported)
print(f"Restored ID: {restored.conversation_id}")
print(f"Restored messages: {len(restored.messages)}")
print(f"Restored system prompt: {restored._system_prompt}")
print()


# ── Section 2: File-based save/load ─────────────────────────────────

print("=" * 60)
print("Section 2: File-based save / load")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "conversation.json"
    conv.save(path)
    print(f"Saved to: {path}")
    print(f"File size: {path.stat().st_size} bytes")

    loaded = Conversation.load(path)
    print(f"Loaded conversation ID: {loaded.conversation_id}")
    print(f"Loaded messages: {len(loaded.messages)}")
    print()

    # Peek at the JSON structure
    data = json.loads(path.read_text())
    print("Export keys:", list(data.keys()))
    print()


# ── Section 3: SQLite ConversationStore ──────────────────────────────

print("=" * 60)
print("Section 3: SQLite ConversationStore")
print("=" * 60)

from prompture import ConversationStore

with tempfile.TemporaryDirectory() as tmpdir:
    store = ConversationStore(db_path=Path(tmpdir) / "conversations.db")

    # Save several conversations with tags
    for topic, tags in [
        ("geography", ["demo", "geography"]),
        ("science", ["demo", "science"]),
        ("history", ["demo", "history"]),
    ]:
        c = Conversation(model_name="ollama/llama3.1:8b", tags=tags)
        c.add_context("user", f"Tell me about {topic}")
        c.add_context("assistant", f"Here's what I know about {topic}...")
        data = c.export()
        store.save(c.conversation_id, data)
        print(f"Saved: {c.conversation_id[:8]}... [{', '.join(tags)}]")

    print()

    # Search by tag
    demo_convs = store.find_by_tag("demo")
    print(f"Conversations tagged 'demo': {len(demo_convs)}")

    science_convs = store.find_by_tag("science")
    print(f"Conversations tagged 'science': {len(science_convs)}")

    # List all
    all_convs = store.list_all()
    print(f"Total conversations: {len(all_convs)}")

    # Find by ID
    first_id = all_convs[0]["id"]
    summary = store.find_by_id(first_id)
    print(f"Found by ID: {summary['id'][:8]}... (tags: {summary['tags']})")

    # Load full data
    full_data = store.load(first_id)
    print(f"Full data messages: {len(full_data['messages'])}")

    # Delete
    deleted = store.delete(first_id)
    print(f"Deleted: {deleted}")
    print(f"After delete: {len(store.list_all())} conversations remain")
    print()


# ── Section 4: Auto-save ────────────────────────────────────────────

print("=" * 60)
print("Section 4: Auto-save")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    auto_path = Path(tmpdir) / "auto_save.json"

    conv = Conversation(
        model_name="ollama/llama3.1:8b",
        system_prompt="You are helpful.",
        auto_save=auto_path,
        tags=["auto-save-demo"],
    )

    # Each add_context doesn't trigger auto-save (only _accumulate_usage does),
    # but save() works explicitly
    conv.add_context("user", "Hello")
    conv.add_context("assistant", "Hi!")
    conv.save(auto_path)

    print(f"Auto-save path: {auto_path}")
    print(f"File exists: {auto_path.exists()}")
    if auto_path.exists():
        data = json.loads(auto_path.read_text())
        print(f"Auto-saved messages: {len(data['messages'])}")
    print()


# ── Section 5: Strip images on export ───────────────────────────────

print("=" * 60)
print("Section 5: Export with strip_images")
print("=" * 60)

conv = Conversation(model_name="ollama/llama3.1:8b")
# Simulate a message with image content
conv._messages.append(
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image",
                "source": {
                    "data": "base64encodeddata...",
                    "media_type": "image/png",
                    "source_type": "base64",
                    "url": None,
                },
            },
        ],
    }
)
conv._messages.append({"role": "assistant", "content": "I see a cat."})

export_with_images = conv.export(strip_images=False)
export_without_images = conv.export(strip_images=True)

print(f"With images - content type: {type(export_with_images['messages'][0]['content']).__name__}")
print(f"Without images - content: {export_without_images['messages'][0]['content']!r}")
print()

print("=" * 60)
print("Done!")
print("=" * 60)
