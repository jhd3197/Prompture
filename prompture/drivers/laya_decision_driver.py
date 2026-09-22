"""Laya decision driver — in-process, open-weights typed decisions.

Laya (Apache 2.0) is a non-autoregressive System 1 decision engine.  Unlike
TypeSafe and Kev it exposes no HTTP endpoint: the weights run in your own
process, so there is no network hop, no API key and no per-token cost.

The ``laya`` package is a lazy dependency — it pulls ``torch`` and
``transformers``, so it is imported on first :meth:`decide` call and installed
separately::

    pip install prompture[laya]

Model strings map to Laya's checkpoints:

===================  ====================================================
``laya/router``      ``Router`` — detects script/language and dispatches
                     to the right checkpoint per request (recommended)
``laya/english``     ModernBERT-large, English only
``laya/multilingual``mmBERT-base, 100+ languages, faster
``laya/typed-decisions``
                     the checkpoint fine-tuned on typed-decision workflows
===================  ====================================================

Two notes worth knowing before you rely on it, both from Laya's own README:
the base checkpoints score near chance on typed decisions zero-shot (it is a
fast base to fine-tune, not a zero-shot engine), and the English checkpoint
stays confident while being wrong outside Latin scripts — which is why
``laya/router`` is the default here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from .decision_base import (
    DecisionDriver,
    DecisionResponse,
    QuestionLike,
    StateLike,
    calculate_decision_cost,
    normalize_questions,
    parse_decision_response,
)

logger = logging.getLogger(__name__)

#: Model string → (checkpoint kwarg for ``Router.predict``, HF subfolder).
_CHECKPOINTS: dict[str, str | None] = {
    "router": None,
    "english": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}

_HF_REPO = "convaiinnovations/laya"


class LayaDecisionDriver(DecisionDriver):
    """In-process Laya decision driver.

    Args:
        model: One of ``router``, ``english``, ``multilingual``,
            ``typed-decisions``, or a raw Hugging Face repo id.
        preload: In ``router`` mode, load every checkpoint up front.  Leave
            this on for anything server-shaped: at the library default a
            request in a new language rebuilds a checkpoint, which Laya
            measures at 7–10 s.
        device: Passed through to Laya (e.g. ``"cuda"``, ``"cpu"``).
    """

    supports_async = False

    PROVIDER = "laya"
    DEFAULT_MODEL = "router"

    KNOWN_MODELS: tuple[str, ...] = tuple(_CHECKPOINTS)

    def __init__(
        self,
        model: str | None = None,
        preload: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        self.model = model or self.DEFAULT_MODEL
        self.preload = preload
        self.device = device
        self._agent: Any = None
        self._router: Any = None

    # ── Lazy model loading ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the Laya checkpoint (or router) on first use."""
        if self._agent is not None or self._router is not None:
            return

        try:
            import laya
        except ImportError:
            raise ImportError(
                "The 'laya' package is required for the Laya decision driver. "
                "Install it with: pip install prompture[laya]"
            ) from None

        if self.model == "router":
            kwargs: dict[str, Any] = {"preload": self.preload}
            if self.device:
                kwargs["device"] = self.device
            self._router = laya.Router(**kwargs)
            return

        subfolder = _CHECKPOINTS.get(self.model, "__unknown__")
        if subfolder == "__unknown__":
            # Not one of the named checkpoints — treat it as a raw repo id.
            self._agent = laya.load(self.model)
        elif subfolder is None:
            self._agent = laya.load(_HF_REPO)
        else:
            self._agent = laya.load(_HF_REPO, subfolder=subfolder)

    # ── Public API ─────────────────────────────────────────────────────────

    def decide(
        self,
        state: StateLike,
        questions: Mapping[str, QuestionLike],
        **options: Any,
    ) -> DecisionResponse:
        payload_questions = normalize_questions(questions)
        model = options.pop("model", self.model)
        self._ensure_loaded()

        self._fire_callback(
            "on_request",
            {"provider": self.PROVIDER, "model": model, "payload": {"state": state, "questions": payload_questions}},
        )

        try:
            if self._router is not None:
                resp = self._router.predict(state, payload_questions, **options)
            else:
                resp = self._agent.predict(state, payload_questions, **options)
        except Exception as e:
            error_msg = f"Laya decision call failed: {e!s}"
            self._fire_callback("on_error", {"provider": self.PROVIDER, "model": model, "error": error_msg})
            raise RuntimeError(error_msg) from e

        if not isinstance(resp, Mapping):
            raise RuntimeError(f"Laya returned {type(resp).__name__}, expected a mapping with an 'answers' key.")

        routing = resp.get("routing") or {}
        # Report the checkpoint the router actually picked, not the alias.
        fallback = str(routing.get("model") or model)
        answered_model, answers = parse_decision_response(resp, fallback_model=fallback)
        if self._router is not None and routing:
            answered_model = str(routing.get("model") or answered_model)

        cost, pricing_unknown = calculate_decision_cost(self.PROVIDER, model)
        self.last_usage = {
            "model_name": f"{self.PROVIDER}/{model}",
            "questions": len(payload_questions),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": cost,
            "pricing_unknown": pricing_unknown,
            "routing": dict(routing) if routing else None,
            "raw_response": dict(resp),
        }
        self._fire_callback("on_response", {"provider": self.PROVIDER, "model": model, "usage": self.last_usage})

        return DecisionResponse(
            model=answered_model,
            answers=answers,
            usage=dict(self.last_usage),
            raw_response=dict(resp),
        )

    def unload(self) -> None:
        """Release the resident checkpoints and free memory."""
        if self._router is not None and hasattr(self._router, "unload"):
            self._router.unload()
        self._router = None
        self._agent = None
