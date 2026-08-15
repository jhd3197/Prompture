"""Built-in node runners.

Each runner has signature ``run(node, config, ctx) -> NodeResult`` where
``config`` is the node's config with all ``{{...}}`` references already resolved.
Runners own NO provider logic — they delegate to existing Prompture surfaces
(``get_driver_for_model``, the media driver factories, ``ToolRegistry``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import NodeExecutionError
from .model import Handle, HandleType, Node
from .registry import register_node_type
from .state import NodeResult, NodeStatus

if TYPE_CHECKING:
    from .scheduler import ExecutionContext

__all__ = ["register_all"]


def _ok(node_id: str, outputs: dict[str, Any], **kw: Any) -> NodeResult:
    return NodeResult(node_id=node_id, status=NodeStatus.COMPLETED, outputs=outputs, **kw)


# ── pure / local nodes ──────────────────────────────────────────────────────


def run_input(node: Node, config: dict[str, Any], ctx: ExecutionContext) -> NodeResult:
    name = config.get("name", node.id)
    value = ctx.inputs.get(name, config.get("default"))
    return _ok(node.id, {"value": value})


def run_output(node: Node, config: dict[str, Any], ctx: ExecutionContext) -> NodeResult:
    return _ok(node.id, {"value": config.get("value")})


def run_constant(node: Node, config: dict[str, Any], ctx: ExecutionContext) -> NodeResult:
    return _ok(node.id, {"value": config.get("value")})


def run_transform(node: Node, config: dict[str, Any], ctx: ExecutionContext) -> NodeResult:
    op = config.get("op", "value")
    if op in ("value", "identity"):
        value = config.get("value")
    elif op == "concat":
        value = config.get("sep", "").join(str(x) for x in config.get("values", []))
    elif op == "format":
        value = str(config.get("template", "")).format(*config.get("args", []), **config.get("kwargs", {}))
    elif op == "get":
        value = config.get("value")
        for step in config.get("path", []):
            value = value[step]
    else:
        raise NodeExecutionError(node.id, f"Unknown transform op {op!r}")
    return _ok(node.id, {"value": value})


# ── LLM node ────────────────────────────────────────────────────────────────


def run_llm(node: Node, config: dict[str, Any], ctx: ExecutionContext) -> NodeResult:
    from ..drivers import get_driver_for_model

    model = config.get("model")
    if not model:
        raise NodeExecutionError(node.id, "llm node requires config['model']")
    prompt = config.get("prompt", "")
    options = dict(config.get("options", {}))

    driver = get_driver_for_model(model)
    resp = driver.generate(prompt, options)
    text = resp.get("text", "")
    meta = resp.get("meta", {}) or {}
    usage = {
        "prompt_tokens": meta.get("prompt_tokens", 0),
        "completion_tokens": meta.get("completion_tokens", 0),
        "total_tokens": meta.get("total_tokens", 0),
        "cost": meta.get("cost", 0.0),
    }
    return NodeResult(
        node_id=node.id,
        status=NodeStatus.COMPLETED,
        outputs={"text": text},
        usage=usage,
        cost=float(meta.get("cost", 0.0) or 0.0),
        strategy=meta.get("strategy"),
        raw=meta if isinstance(meta, dict) else None,
    )


# ── Tool node ───────────────────────────────────────────────────────────────


def run_tool(node: Node, config: dict[str, Any], ctx: ExecutionContext) -> NodeResult:
    registry = ctx.options.get("tool_registry")
    if registry is None:
        raise NodeExecutionError(node.id, "tool node requires a 'tool_registry' run option")
    tool_name = config.get("tool")
    if not tool_name:
        raise NodeExecutionError(node.id, "tool node requires config['tool']")
    args = config.get("args", {})
    result = registry.execute(tool_name, args if isinstance(args, dict) else {"value": args})
    return _ok(node.id, {"result": result})


# ── Media node (blocking) ───────────────────────────────────────────────────

_MEDIA = {
    "image": ("img_gen_registry", "get_img_gen_driver_for_model", "generate_image", "images"),
    "video": ("video_gen_registry", "get_video_gen_driver_for_model", "generate_video", "videos"),
    "music": ("music_registry", "get_music_driver_for_model", "generate_music", "audio"),
}


def run_media_gen(node: Node, config: dict[str, Any], ctx: ExecutionContext) -> NodeResult:
    import importlib

    modality = config.get("modality", "image")
    if modality not in _MEDIA:
        raise NodeExecutionError(node.id, f"Unsupported media modality {modality!r} (image|video|music)")
    model = config.get("model")
    if not model:
        raise NodeExecutionError(node.id, "media_gen node requires config['model']")
    prompt = config.get("prompt", "")
    options = dict(config.get("options", {}))

    mod_name, factory_name, method_name, items_key = _MEDIA[modality]
    factory = getattr(importlib.import_module(f"..drivers.{mod_name}", __package__), factory_name)
    driver = factory(model)
    resp = getattr(driver, method_name)(prompt, options)
    items = resp.get(items_key, []) or []
    urls = [getattr(c, "url", None) for c in items if getattr(c, "url", None)]
    meta = resp.get("meta", {}) or {}
    cost = float(meta.get("cost", 0.0) or 0.0)
    return NodeResult(
        node_id=node.id,
        status=NodeStatus.COMPLETED,
        outputs={"urls": urls, "url": urls[0] if urls else None, "model": meta.get("model_name", model)},
        usage={"cost": cost},
        cost=cost,
        raw=meta if isinstance(meta, dict) else None,
    )


def register_all() -> None:
    """Register every built-in node type (idempotent via overwrite)."""
    register_node_type("input", run_input, output_handles=(Handle("value"),), overwrite=True)
    register_node_type("output", run_output, output_handles=(Handle("value"),), overwrite=True)
    register_node_type("constant", run_constant, output_handles=(Handle("value"),), overwrite=True)
    register_node_type("transform", run_transform, output_handles=(Handle("value"),), overwrite=True)
    register_node_type("llm", run_llm, output_handles=(Handle("text", HandleType.TEXT),), overwrite=True)
    register_node_type("tool", run_tool, output_handles=(Handle("result"),), overwrite=True)
    register_node_type(
        "media_gen",
        run_media_gen,
        output_handles=(Handle("urls", HandleType.LIST), Handle("url", HandleType.TEXT)),
        produces_job=True,
        overwrite=True,
    )
