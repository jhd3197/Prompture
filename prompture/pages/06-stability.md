# API Stability

Prompture follows [Semantic Versioning](https://semver.org/) (SemVer).

## Stability Tiers

| Surface | Stability | Notes |
|---------|-----------|-------|
| `extract_with_model`, `stepwise_extract_with_model` | **Stable** | Pydantic-based extraction. Signature and return shape will not change without deprecation. |
| `ask_for_json`, `extract_and_jsonify` | **Stable** | Schema-enforced JSON extraction. |
| `extract_from_data`, `extract_from_pandas` | **Stable** | TOON input conversion + extraction. |
| `StructuredOutputStrategy` | **Stable** | Enum: `provider_native`, `tool_call`, `prompted_repair`. |
| `ProviderCapabilities`, `get_capabilities` | **Stable** | Capability registry for providers and models. |
| `Driver` interface | **Stable** | All drivers implement `generate(prompt, options)`. |
| Response shape (`json_string`, `json_object`, `usage`) | **Stable** | New keys may be added; existing keys preserved. |
| `Conversation`, `ToolRegistry` | **Stable** | Multi-turn sessions and tool registration. |
| `SkillPipeline`, agent abstractions | Experimental | May change between minor versions. |
| `groups/` (consensus, debate) | Experimental | May change between minor versions. |
| `pipeline/` (model routing) | Experimental | May change between minor versions. |

## What "Stable" Means

- The function signature will not change in a backward-incompatible way within the same major version.
- New **optional** parameters or new keys in returned dicts may be added in minor releases.
- Breaking changes go through the deprecation process.

## Deprecation Policy

1. **Deprecation warning** -- The old API continues to work but emits a `DeprecationWarning`.
2. **Minimum one minor version** -- Kept for at least one minor release cycle.
3. **Breaking changes log** -- Recorded in `BREAKING_CHANGES.md`.
4. **Migration guidance** -- Instructions included in warnings and changelog.
