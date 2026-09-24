"""Fusion: ask a panel of models in parallel, then have a judge write one answer.

``FusionDriver`` is a regular driver, so it drops in anywhere (Conversation,
extraction, ``prompture serve``, a gateway). Register one by name and it
becomes the virtual model ``fusion/<name>``::

    register_fusion("council", panel=["openai/gpt-4o", "claude/claude-sonnet-4-5", "google/gemini-2.5-pro"],
                    judge="claude/claude-opus-4-6")
    Conversation(model_name="fusion/council").ask("Design a rate limiter for ...")

Panel answers are shown to the judge anonymously ("Answer A", "Answer B")
in shuffled order so the synthesis isn't biased toward a brand. Panel
members that fail or miss ``timeout`` are skipped; with a single surviving
answer the judge is bypassed.
"""

from __future__ import annotations

import concurrent.futures as cf
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..drivers.base import Driver
from ..exceptions import DriverError

JUDGE_PROMPT = """You are given several candidate answers to the same request, written independently.
Write the single best final answer to the request:
- keep what is correct and useful from each candidate; resolve disagreements by reasoning about which is right
- drop errors, hedging and repetition
- answer the request directly in the user's language and format, without mentioning the candidates

Request:
{request}

{answers}"""


def _default_factory(model: str) -> Any:
    from ..drivers import get_driver_for_model

    return get_driver_for_model(model)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            return " ".join(b.get("text", "") for b in content or [] if isinstance(b, dict))
    return ""


class FusionDriver(Driver):
    supports_messages = True

    def __init__(
        self,
        panel: Sequence[str | Any],
        judge: str | Any,
        *,
        timeout: float = 90.0,
        min_answers: int = 1,
        judge_prompt: str = JUDGE_PROMPT,
        factory: Callable[[str], Any] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if not panel:
            raise ValueError("Fusion needs at least one panel model")
        self._factory = factory or _default_factory
        self._panel_specs = list(panel)
        self._judge_spec = judge
        self._drivers: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.timeout = timeout
        self.min_answers = max(1, min_answers)
        self.judge_prompt = judge_prompt
        self._rng = rng or random.Random()  # nosec B311 - shuffling for bias, not crypto
        self.model = "fusion"
        self.last_fusion: dict[str, Any] | None = None

    def _driver(self, spec: Any) -> Any:
        if not isinstance(spec, str):
            return spec
        with self._lock:
            if spec not in self._drivers:
                self._drivers[spec] = self._factory(spec)
            return self._drivers[spec]

    @staticmethod
    def _name(spec: Any) -> str:
        return spec if isinstance(spec, str) else str(getattr(spec, "model", type(spec).__name__))

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self.generate_messages([{"role": "user", "content": prompt}], options)

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        results: list[dict[str, Any]] = []

        def ask(spec: Any) -> dict[str, Any]:
            t0 = time.perf_counter()
            resp = self._driver(spec).generate_messages(messages, dict(options or {}))
            return {"model": self._name(spec), "resp": resp, "elapsed_ms": (time.perf_counter() - t0) * 1000}

        pool = cf.ThreadPoolExecutor(max_workers=len(self._panel_specs))
        futures = {pool.submit(ask, spec): spec for spec in self._panel_specs}
        try:
            for fut in cf.as_completed(futures, timeout=self.timeout):
                spec = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append({"model": self._name(spec), "error": str(exc)})
        except cf.TimeoutError:
            for fut, spec in futures.items():
                if not fut.done():
                    results.append({"model": self._name(spec), "error": f"timed out after {self.timeout:g}s"})
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        answers = [r for r in results if "resp" in r and (r["resp"].get("text") or "").strip()]
        panel_meta = [
            {
                "model": r["model"],
                "ok": "resp" in r,
                "error": r.get("error"),
                "elapsed_ms": round(r.get("elapsed_ms", 0.0), 1),
                "cost": float((r.get("resp") or {}).get("meta", {}).get("cost", 0.0) or 0.0),
            }
            for r in results
        ]
        if len(answers) < self.min_answers:
            raise DriverError(
                f"Fusion got {len(answers)} usable answer(s), needs {self.min_answers}: "
                + "; ".join(f"{p['model']}: {p['error'] or 'empty'}" for p in panel_meta)
            )

        judge_resp: dict[str, Any] | None = None
        if len(answers) == 1:
            text = answers[0]["resp"]["text"]
            judge_name = None
        else:
            shuffled = answers[:]
            self._rng.shuffle(shuffled)
            labelled = "\n\n".join(
                f"Answer {chr(65 + i)}:\n{a['resp']['text'].strip()}" for i, a in enumerate(shuffled)
            )
            prompt = self.judge_prompt.format(request=_last_user_text(messages), answers=labelled)
            judge_messages = [m for m in messages if m.get("role") == "system"] + [{"role": "user", "content": prompt}]
            judge_resp = self._driver(self._judge_spec).generate_messages(judge_messages, dict(options or {}))
            text = judge_resp.get("text", "")
            judge_name = self._name(self._judge_spec)

        metas = [a["resp"].get("meta", {}) or {} for a in answers]
        if judge_resp is not None:
            metas.append(judge_resp.get("meta", {}) or {})

        def total(field: str) -> Any:
            return sum((m.get(field, 0) or 0) for m in metas)

        fusion = {
            "panel": panel_meta,
            "judge": judge_name,
            "answers_used": len(answers),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        self.last_fusion = fusion
        return {
            "text": text,
            "meta": {
                "prompt_tokens": total("prompt_tokens"),
                "completion_tokens": total("completion_tokens"),
                "total_tokens": total("total_tokens"),
                "cost": float(total("cost")),
                "raw_response": {},
                "model_name": judge_name or answers[0]["model"],
                "fusion": fusion,
            },
        }


class AsyncFusionDriver:
    """Async facade over :class:`FusionDriver` (panel calls already run in threads)."""

    supports_messages = True
    supports_json_mode = False
    supports_json_schema = False
    supports_tool_use = False
    supports_streaming = False
    supports_streaming_tool_use = False
    supports_vision = False
    callbacks = None

    def __init__(self, sync: FusionDriver) -> None:
        self._sync = sync
        self.model = sync.model

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self._sync.generate, prompt, options)

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self._sync.generate_messages, messages, options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sync, name)


@dataclass
class FusionSpec:
    name: str
    panel: list[str]
    judge: str
    timeout: float = 90.0
    min_answers: int = 1

    def driver(self) -> FusionDriver:
        drv = FusionDriver(self.panel, self.judge, timeout=self.timeout, min_answers=self.min_answers)
        drv.model = f"fusion/{self.name}"
        return drv


_fusions: dict[str, FusionSpec] = {}
_fusions_lock = threading.Lock()


def register_fusion(
    name: str, panel: Sequence[str], judge: str, *, timeout: float = 90.0, min_answers: int = 1
) -> FusionSpec:
    """Register ``fusion/<name>`` as a virtual model."""
    spec = FusionSpec(name.removeprefix("fusion/"), list(panel), judge, timeout, min_answers)
    with _fusions_lock:
        _fusions[spec.name] = spec
    return spec


def get_fusion(name: str) -> FusionSpec | None:
    with _fusions_lock:
        return _fusions.get(name.removeprefix("fusion/"))


def list_fusions() -> list[FusionSpec]:
    with _fusions_lock:
        return list(_fusions.values())


def clear_fusions() -> None:
    with _fusions_lock:
        _fusions.clear()
