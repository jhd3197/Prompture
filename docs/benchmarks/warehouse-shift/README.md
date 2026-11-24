# Routing fresh warehouse requests

On **60 new requests**, Claude Sonnet 4.6 with three recipes retrieved from reviewed examples made **57 correct decisions using 9,664 routing tokens**. Sending all six recipes yielded **51 correct with 13,624 tokens**. That is six additional correct decisions and **29.1% fewer tokens**. The largest local improvement came from Laya. With the example shortlist it selected the right recipe for **39 of 42** supported requests, matching Claude. It also proposed a recipe for **12 of 18** requests that required a decline. Claude declined all 18.

## What a shortlist changed

The two fresh sets shared the same catalog and the two arms below. Each cell in the first two result columns gives correct decisions out of 60 and the sum of provider-reported input and output tokens. Invalid outputs receive no credit. A correct abstention requires an explicit decline.

| Model | Full catalog · correct / tokens | Example shortlist · correct / tokens | Right recipe / 42, shortlist | Correct declines / 18, shortlist |
| --- | ---: | ---: | ---: | ---: |
| Claude Sonnet 4.6 | 51 / 13,624 | **57 / 9,664** | 39 | 18 |
| Laya typed-decisions | 33 / 10,862 | **45 / 7,417** | 39 | 6 |
| Qwen2.5 1.5B | 29 / 11,945 | 28 / 8,751 | 10 | 18 |
| Llama 3.1 8B | 16 / 12,078 | 21 / 8,884 | 5 | 16 |

The local ranker uses five previously labeled requests per recipe. It weights uncommon words and character trigrams, then scores a recipe against its closest example. The model sees only the selected names and descriptions. Across the two sets the correct recipe appeared in the example top three on **42/42**, and ranked first on **32/42**. These retrieval figures apply only to supported requests. Selecting the first recipe unconditionally would also accept all 18 unsupported requests.

## Did categories help?

The first set had 24 supported requests and 12 requests involving changes, ambiguity, multiple jobs or unavailable information. Catalog means all six choices. Lexical means the shipped top three using names, descriptions and aliases. Examples means the top three using reviewed wording. Categories starts from the same example scores, infers the area from the strongest match, favors up to two recipes in that area, then fills the remaining places. The three areas were inventory, packaging and transport. Each had two recipes.

| Model | Catalog · correct / tokens | Lexical | Examples | Categories |
| --- | ---: | ---: | ---: | ---: |
| Claude | 33 / 8,162 | 29 / 5,846 | **34 / 5,807** | 34 / 5,811 |
| Laya | 18 / 6,531 | 16 / 4,478 | **26 / 4,461** | 23 / 4,470 |
| Qwen | 19 / 7,182 | 17 / 5,287 | 17 / 5,265 | 15 / 5,268 |
| Llama | 12 / 7,232 | 15 / 5,351 | **17 / 5,315** | 14 / 5,318 |

Lexical retrieval included the correct recipe on 18/24 supported requests, examples on 24/24 and categories on 23/24. Grouping provided no overall accuracy gain for any model. Six recipes give little room for a 13-way warehouse taxonomy. A larger catalog with multiple plausible recipes per department would be a better test of that idea. In this experiment, Laya selected carton contents for a question about humidity in a cheese cage. Its ability to find the correct supported lookup gives little protection against that kind of mistake.

## Did asking a simpler question help?

After observing the first set, we fixed a separate set of 24 requests with 18 supported jobs and six required declines. We compared showing three examples-ranked recipes with showing only the strongest and asking for confirmation. Text models answered yes or no. Laya used a typed choice between one recipe and abstention. The top candidate was correct for 13/18 supported requests, setting the maximum possible number of correct recipe selections for confirmation.

| Model | Full · correct / tokens | Three examples | Confirm first | Correct recipe / 18, confirm | Correct declines / 6, confirm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude | 18 / 5,462 | **23 / 3,857** | 7 / 2,179 | 4 | 3 |
| Laya | 15 / 4,331 | **19 / 2,956** | 17 / 2,036 | 13 | 4 |
| Qwen | 10 / 4,763 | 11 / 3,486 | **11 / 2,118** | 8 | 3 |
| Llama | 4 / 4,846 | 4 / 3,569 | **8 / 2,117** | 3 | 5 |

On this set Qwen matched its three-choice score with **39% fewer tokens** under confirmation. Llama improved from four to eight correct, while Claude fell from 23 to seven. A single proposal removes the opportunity to rescue a poor first match. Invalid yes/no responses also counted against the score. The confirmation mode is opt-in.

**What about putting Laya in front of an older model?** In an exploratory replay of these 24 requests, Laya proposed a recipe on 22. A live Qwen review of each proposal yielded **15/24 correct with 4,868 combined tokens**. Llama review yielded **10/24 with 4,906**. Both reduced false accepts on unsupported requests from four for Laya alone to zero. The extra review also rejected correct proposals. Claude with the three examples-ranked candidates still led at 23/24 and used fewer tokens than either two-model route. Counts include the measured Laya pass on every request and the review call whenever it was needed. [Qwen review](review-qwen.json) · [Llama review](review-llama.json)

## Using it in Prompture

Supply reviewed example phrases for the recipe IDs in your existing store. Construct the index once per project. It stays in process and the caller controls its lifetime. Exact aliases continue to resolve locally. Search, model selection and an independent verifier remain separate steps.

```python
from prompture.workflow import RecipeExampleIndex

index = RecipeExampleIndex({
    "inbound_count": ["Did the vendor deliver the units on our purchase order?"],
    "free_stock": ["How many units can we still promise after reservations?"],
})
request = "Do we have any sellable units left at the north building?"
choices = store.search(request, index=index, limit=3)
match = store.discover(request, index=index, limit=3, model="claude/claude-sonnet-4-6")
```

For a local typed selector, pass `model="laya/typed-decisions"` with the Laya extra installed. A configured Jev account can use `model="typesafe/jev-latest"`. This machine had no Jev credential, so every typed-model number above is Laya's. Use `confirm_first=True` to test a single suggestion on your own requests. To evaluate a taxonomy, pass a recipe-name-to-category mapping as `categories=` when constructing the index. Consider further validation before promoting free-text decisions to automatic execution. [Creating a store and verifying execution](../pack-checks/README.md#using-saved-recipes)

## Evidence and reproduction

Both sets were authored for this evaluation. Their labels and primary arms were committed before the first model call on each set (`82c80e2` and `a5db73a`). Only positive phrases from the original 48-case benchmark built the index. The following runs used identical shuffled requests and matched menus per arm, independent calls without conversation history, and no automatically learned aliases. Each report records the fixture hash, candidate list, predictions and input/output usage per request. Freshness here means unseen in the experiment and example index. Model pretraining provenance and representativeness of real shifts are unknown. These are routing outcomes on synthetic English requests, with no inventory execution, cost-in-dollars or end-to-end latency claim.

Claude used the Messages API. The two older models ran through Ollama with their Q4_K_M checkpoints. Laya used the local typed-decisions checkpoint from revision `fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1` and package 0.5.3. It reports encoder input tokens and zero generated output tokens. The models use different tokenizers. Relative savings within one model are the more interpretable comparison. No confidence threshold was fitted.

| Records | Claude | Qwen | Llama | Laya |
| --- | --- | --- | --- | --- |
| First 36 | [Decisions](sonnet.json) | [Decisions](qwen.json) | [Decisions](llama.json) | [Decisions](laya.json) |
| Second 24 | [Decisions](confirm-sonnet.json) | [Decisions](confirm-qwen.json) | [Decisions](confirm-llama.json) | [Decisions](confirm-laya.json) |

```sh
python -m scripts.benchmark_recipe_discovery --shift-holdout --model ollama/qwen2.5:1.5b --out qwen.json
python -m scripts.benchmark_recipe_discovery --confirmation-holdout --model claude/claude-sonnet-4-6 --out confirm-sonnet.json
python -m scripts.benchmark_recipe_discovery --confirmation-holdout --decision --model laya/typed-decisions --out confirm-laya.json
python -m scripts.benchmark_recipe_review --proposals confirm-laya.json --model ollama/llama3.1:8b --out review-llama.json
```

**Related research.** ToolRerank found that accounting for hierarchy can improve retrieval and downstream execution on a larger tool library. AdaMenu compared retrieved example-based menus with hierarchical selection over larger label sets and found that one compact menu could reduce input tokens while retaining or improving accuracy on several datasets. These findings motivated testing retrieval recall and final selection separately. Neither study establishes an optimal number of warehouse categories. [ToolRerank, LREC-COLING 2024](https://arxiv.org/abs/2403.06551) · [AdaMenu preprint](https://arxiv.org/abs/2611.19005) · [Laya model card](https://huggingface.co/convaiinnovations/laya)
