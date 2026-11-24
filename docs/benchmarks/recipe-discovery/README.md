# Finding a saved recipe from unfamiliar wording

Claude Sonnet 4.6 selected the right recipe for **27 of 30** supported warehouse requests and declined **all 18** requests that needed clarification or another capability. It returned an invalid choice for the other three. With the same six recipes, Qwen2.5 1.5B selected **11 of 30** and Llama 3.1 8B selected **2 of 30**. Both older models used fewer tokens than Claude. Neither matched its accuracy.

## Matched results

Each row covers the same 48 requests. A correct decision means either selecting the exact expected recipe or explicitly declining a request labeled for abstention. An invalid answer never executes and earns no credit. False accepts count selections on the 18 unsupported requests. Wrong recipe counts selections of a different recipe for one of the 30 supported requests.

| Model | Choices | Correct / 48 | Right recipe / 30 | Correct declines / 18 | False accepts | Wrong recipe | Invalid | Input + output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude Sonnet 4.6 | 6 | 45 | 27 | 18 | 0 | 0 | 3 | 10,269 + 289 |
| Claude Sonnet 4.6 | 3 | 44 | 26 | 18 | 0 | 1 | 1 | 7,189 + 255 |
| Qwen2.5 1.5B | 6 | 25 | 11 | 14 | 4 | 3 | 0 | 9,174 + 96 |
| Qwen2.5 1.5B | 3 | 21 | 10 | 11 | 7 | 2 | 0 | 6,635 + 96 |
| Llama 3.1 8B | 6 | 17 | 2 | 15 | 1 | 6 | 3 | 9,269 + 138 |
| Llama 3.1 8B | 3 | 19 | 3 | 16 | 0 | 4 | 3 | 6,730 + 137 |

Showing Claude three candidates reduced total tokens from **10,558 to 7,444 (29.5%)**. One receiving question then got a packing suggestion. For Qwen, the smaller list also saved tokens but increased false accepts from four to seven. For example, it assigned “Check that the paperwork matches” to the inbound-count recipe despite the missing context. A lower token count alone gives a poor picture of the user experience.

**Where retrieval matters.** Before inference, local word matching ranked the correct recipe first for 22 of the 30 supported requests and in the top three for 27. “Subtract the units already spoken for from the shelf balance at this site” missed the top-three cutoff for free stock. Claude selected it with the full catalog and declined with the shorter list. Returning a local shortlist costs zero model tokens and is available through `store.search()`. Taking its first result unconditionally would also assign a recipe to every unsupported request.

**What was measured.** Six host-authored descriptions and six reviewed aliases were stored in SQLite. The fixture contained five new phrasings per recipe and 18 ambiguous, multi-intent, write or out-of-scope requests. Every model received the same shuffled requests, candidate order, selection instructions, requested temperature of zero and 16-token output allowance. Each decision used one inference. These are provider-reported routing tokens for an authored English fixture with an empty use history. Tokenizers differ. The experiment measures neither end-to-end order accuracy nor the cost of execution or escalation. The earlier packing experiment measured execution using Claude and Anthropic hosted programmatic tool calling. [Packing benchmark](../pack-checks/README.md)

**Format correction.** The first run accepted only bare numbers. In two diagnosis calls, Claude returned a correct number in Markdown bold. The final parser accepts a single bare or bolded choice, rejects multiple answers and prose, and assigns zero to abstention. We reran all three models on the same fixture. Claude's six-choice score rose from 28 to 45. This fixture informed the parser correction. A fresh blind sample from actual operators is needed to estimate production accuracy. [First-pass records](strict-trial/) · [Final Claude](sonnet.json) · [Final Qwen](qwen.json) · [Final Llama](llama.json)

## Calling discovery

Start with a store containing reviewed recipes. In this example, `store` is the packing store in the linked setup, `inputs` come from the host, and `verify` checks fresh business evidence. [Create the packing store](../pack-checks/README.md#using-saved-recipes)

```python
match = store.discover(
    "Does what we packed agree with the shipment lines?",
    model="claude/claude-sonnet-4-6",
    limit=6,
)
if match.recipe:
    print(match.status, match.recipe.name, match.recipe.description, match.usage)
    # Once the host confirms this recipe applies to the request:
    # output = store.run_match(match, inputs, verify=verify, event_id=event_id)
else:
    print("Clarify the request or choose another workflow.", match.status)
```

A confirmed exact name or alias resolves locally. For unfamiliar wording, Prompture ranks the compatible recipes in the project using their names, descriptions and reviewed aliases. Verified usage breaks score ties. The driver receives the request and a numbered shortlist of names and descriptions. A completion-only model can make this choice. Results distinguish an exact hit, a suggestion, an explicit abstention, invalid output and an empty catalog. A suggestion adds no alias or popularity credit. `run_match()` rechecks the recipe fingerprint, validates the supplied inputs and records use after the verifier passes. A verifier that only checks the output shape cannot establish that the user intended the selected recipe.

For a known warehouse event, use the exact route directly. For a person asking a new question, surface the proposed recipe and let the host resolve ambiguity. For deployments restricted to the two local models tested here, showing the local shortlist gives the person a choice without spending any model tokens. Tukuy tools can participate through the existing registry. Discovery itself adds no Tukuy requirement.

## Reproduction

Run from the repository root with the corresponding provider configured. The Ollama runs used version 0.35.1 and Q4_K_M model manifests beginning with `65ec06548149` for Qwen and `46e0c10c039e` for Llama. Each JSON contains the fixture hash, candidate order, expected and actual decision, token counts and elapsed time for every request. [Cases](../../../scripts/data/recipe_discovery_cases.json) · [Benchmark runner and catalog](../../../scripts/benchmark_recipe_discovery.py)

```sh
python -m scripts.benchmark_recipe_discovery --model ollama/qwen2.5:1.5b --out qwen.json
python -m scripts.benchmark_recipe_discovery --model ollama/llama3.1:8b --out llama.json
python -m scripts.benchmark_recipe_discovery --model claude/claude-sonnet-4-6 --out sonnet.json
```
