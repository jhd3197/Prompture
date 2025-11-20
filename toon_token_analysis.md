# TOON Token Comparison Findings

## Test Setup
- **Script:** `examples/output_format_comparison.py`
- **Model:** `lmstudio/deepseek/deepseek-r1-0528-qwen3-8b`
- **Prompt text:** multi-paragraph profile for Alice Johnson (see script)
- **Schema:** resume-style schema (name, age, location, skills, etc.)
- **Run date:** 2025-11-20

## Results (most recent run)
| Format | Success | Prompt Tokens | Completion Tokens | Total Tokens |
| ------ | ------- | ------------- | ----------------- | ------------ |
| JSON | ✅ | 344 | 408 | 752 |
| TOON | ✅ | 312 | 1 885 | 2 197 |

Even with the compact TOON instructions, the completion length exploded, so TOON consumed **+1 445 tokens** overall compared to JSON.

## Why TOON Blew Up This Time
1. **Completion verbosity dominates.** Although the new instructions shaved 32 prompt tokens (344 → 312), the model responded with long, multi-line prose for TOON arrays, inflating completions from 408 → 1 885.
2. **Format drift persists.** The model occasionally adds blank lines, semicolons, and descriptive phrases, forcing the decoder (and our cleanup) to carry more text before conversion.
3. **Schema coverage pressure.** Requiring every schema field with `null` fallbacks encourages the model to repeat context when it is unsure, producing paragraphs instead of terse TOON lines.

## Takeaways
- Shorter prompts alone are not enough; the bottleneck is completion discipline.
- Without TOON-aware weights, models default to natural-language explanations inside the TOON structure.
- The theoretical token savings only materialize when the model reliably emits compact, delimiter-only output.

## Paths to Actual Savings
1. **Fine-tune or LoRA adapters.** Provide TOON-formatted training pairs so the model intrinsically emits concise lines without reminders.
2. **Stricter decoding or validators.** Reject responses containing blank lines/semicolons and ask the model to retry, trading latency for compactness.
3. **Post-generation compression.** Accept JSON (cheaper) and convert to TOON locally when humans need that view.
4. **Prompt packs per provider.** Maintain provider-specific TOON instructions tuned for each model’s quirks to minimize unnecessary narration.

## Recommendation
Keep JSON as the production default. TOON can remain an opt-in experiment, but advise users that current models often use *more* tokens unless paired with TOON-finetuned checkpoints or adapters.
