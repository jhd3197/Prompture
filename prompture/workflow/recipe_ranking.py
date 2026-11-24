"""Local retrieval from host-reviewed examples of each recipe's intent."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from .discovery import RecipeCandidate


def _terms(text: str) -> set[str]:
    words = re.findall(r"[^\W_]+", text.casefold())
    features = set(words)
    for word in words:
        padded = f"^{word}$"
        features.update("#" + padded[i : i + 3] for i in range(len(padded) - 2))
    return features


class RecipeExampleIndex:
    """Rank by nearest reviewed wording. Optionally favor recipes in the leading category.

    Construct one index per project and reuse it. Unindexed recipes remain eligible
    in the store's original lexical ordering. Examples stay in the application process.
    """

    def __init__(self, examples: Mapping[str, Sequence[str]], *, categories: Mapping[str, str] | None = None):
        self._examples = {}
        frequency: Counter[str] = Counter()
        documents = 0
        for name, phrases in examples.items():
            if isinstance(phrases, str):
                raise TypeError("Supply a sequence of example requests for each recipe")
            terms = [_terms(phrase) for phrase in phrases if phrase.strip()]
            self._examples[name] = terms
            for document in terms:
                frequency.update(document)
                documents += 1
        self._weights = {term: (1 + math.log((1 + documents) / (1 + count))) ** 2 for term, count in frequency.items()}
        self._categories = dict(categories or {})
        self._norms = {name: [self._norm(doc) for doc in docs] for name, docs in self._examples.items()}

    def _norm(self, terms: set[str]) -> float:
        return math.sqrt(sum(self._weights.get(term, 0.0) for term in terms))

    def shortlist(self, request: str, candidates: Sequence[RecipeCandidate], *, limit: int) -> list[RecipeCandidate]:
        """Rerank compatible candidates, using lexical order to break ties."""
        if limit < 1:
            raise ValueError("limit must be positive")
        query = _terms(request)
        scores = {}
        for candidate in candidates:
            name = candidate.name
            # Query norm is constant across documents and cancels for ranking.
            scores[name] = max(
                (
                    sum(self._weights[t] for t in query & doc) / norm
                    for doc, norm in zip(self._examples.get(name, ()), self._norms.get(name, ()), strict=True)
                    if norm
                ),
                default=0.0,
            )
        ranked = sorted(candidates, key=lambda candidate: -scores[candidate.name])
        if not ranked or not scores[ranked[0].name] or limit < 3:
            return ranked[:limit]
        category = self._categories.get(ranked[0].name)
        if category is None:
            return ranked[:limit]
        same = [c for c in ranked if self._categories.get(c.name) == category][: limit - 1]
        selected = {c.name for c in same}
        return (same + [c for c in ranked if c.name not in selected])[:limit]
