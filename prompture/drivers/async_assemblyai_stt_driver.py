"""Async AssemblyAI STT driver."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

try:
    import httpx
except Exception:
    httpx = None  # type: ignore[assignment]

from .assemblyai_stt_driver import _format_result
from .async_stt_base import AsyncSTTDriver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.assemblyai.com"
_POLL_INTERVAL_SECONDS = 2.0
_MAX_POLL_ATTEMPTS = 30


class AsyncAssemblyAISTTDriver(AsyncSTTDriver):
    """Async speech-to-text via AssemblyAI REST API."""

    supports_timestamps = True
    supports_language_detection = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "universal",
        endpoint: str = _DEFAULT_ENDPOINT,
    ):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("AssemblyAI API key is required (set ASSEMBLYAI_API_KEY).")
        headers = {"Authorization": self.api_key}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    async def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        if httpx is None:
            raise RuntimeError("httpx package is not installed")

        model_id = options.get("model_id", self.model)
        poll_interval = float(options.get("poll_interval_seconds", _POLL_INTERVAL_SECONDS))
        max_attempts = int(options.get("max_poll_attempts", _MAX_POLL_ATTEMPTS))

        async with httpx.AsyncClient() as client:
            # Step 1: Upload.
            upload_resp = await client.post(
                f"{self.endpoint}/v2/upload",
                content=audio,
                headers=self._headers("application/octet-stream"),
                timeout=120,
            )
            if upload_resp.status_code >= 400:
                raise RuntimeError(f"AssemblyAI upload failed [{upload_resp.status_code}]: {upload_resp.text}")
            upload_data = upload_resp.json()
            audio_url = upload_data.get("upload_url")
            if not audio_url:
                raise RuntimeError(f"AssemblyAI upload returned no upload_url: {upload_data!r}")

            # Step 2: Submit.
            submit_body: dict[str, Any] = {
                "audio_url": audio_url,
                "speech_model": model_id,
            }
            if "language_code" in options:
                submit_body["language_code"] = options["language_code"]
            if options.get("punctuate") is not None:
                submit_body["punctuate"] = bool(options["punctuate"])
            if options.get("format_text") is not None:
                submit_body["format_text"] = bool(options["format_text"])

            submit_resp = await client.post(
                f"{self.endpoint}/v2/transcript",
                json=submit_body,
                headers=self._headers("application/json"),
                timeout=60,
            )
            if submit_resp.status_code >= 400:
                raise RuntimeError(
                    f"AssemblyAI transcript submit failed [{submit_resp.status_code}]: {submit_resp.text}"
                )
            submit_data = submit_resp.json()
            transcript_id = submit_data.get("id")
            if not transcript_id:
                raise RuntimeError(f"AssemblyAI submit returned no id: {submit_data!r}")

            # Step 3: Poll.
            poll_url = f"{self.endpoint}/v2/transcript/{transcript_id}"
            for _ in range(max_attempts):
                poll_resp = await client.get(poll_url, headers=self._headers(), timeout=30)
                if poll_resp.status_code >= 400:
                    raise RuntimeError(f"AssemblyAI poll failed [{poll_resp.status_code}]: {poll_resp.text}")
                poll_data = poll_resp.json()
                status = poll_data.get("status")
                if status == "completed":
                    return _format_result(poll_data, model_id)
                if status == "error":
                    raise RuntimeError(f"AssemblyAI transcription error: {poll_data.get('error', 'unknown')}")
                await asyncio.sleep(poll_interval)

        raise RuntimeError(
            f"AssemblyAI transcription timed out after {max_attempts} attempts "
            f"({max_attempts * poll_interval:.0f}s); transcript_id={transcript_id}"
        )
