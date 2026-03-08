"""Ollama server lifecycle and LLM query helpers.

Usage pattern::

    with OllamaClient() as client:
        response = client.generate("Tell me a joke")
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any

import requests

from notorg import config

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT = 30  # seconds to wait for Ollama to become ready
_HEALTH_INTERVAL = 0.5  # polling interval while waiting


class OllamaError(RuntimeError):
    """Raised when communication with Ollama fails."""


class OllamaClient:
    """Context manager that starts an Ollama server, yields, then stops it.

    If Ollama is already running on the configured port it will be used
    directly without being restarted, and it will **not** be stopped on
    exit (since we did not start it).
    """

    def __init__(self) -> None:
        cfg = config.load()
        self.model: str = cfg["ollama_model"]
        self.host: str = cfg["ollama_host"]
        self._process: subprocess.Popen | None = None
        self._started_by_us: bool = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "OllamaClient":
        self._ensure_server_running()
        return self

    def __exit__(self, *_: Any) -> None:
        if self._started_by_us:
            self._stop_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _health_url(self) -> str:
        return f"{self.host}/api/tags"

    def _is_running(self) -> bool:
        try:
            r = requests.get(self._health_url(), timeout=3)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def _ensure_server_running(self) -> None:
        if self._is_running():
            logger.debug("Ollama already running – reusing existing server.")
            self._started_by_us = False
            return

        logger.debug("Starting Ollama server …")
        # Redirect stdout/stderr so they don't clutter CLI output
        popen_kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys.platform != "win32":
            # Start in a new process group so we can cleanly terminate it
            popen_kwargs["preexec_fn"] = os.setsid
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self._process = subprocess.Popen(["ollama", "serve"], **popen_kwargs)
        self._started_by_us = True
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._is_running():
                logger.debug("Ollama server is ready.")
                return
            time.sleep(_HEALTH_INTERVAL)
        raise OllamaError(
            f"Ollama server did not become ready within {_STARTUP_TIMEOUT}s."
        )

    def _stop_server(self) -> None:
        if self._process is None:
            return
        logger.debug("Stopping Ollama server …")
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()
            self._process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                self._process.kill()
            except ProcessLookupError:
                pass
        finally:
            self._process = None

    # ------------------------------------------------------------------
    # LLM API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Send *prompt* to the model and return the full response text."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        data = resp.json()
        return data.get("response", "").strip()

    def generate_json(self, prompt: str) -> Any:
        """Like :meth:`generate` but parse the response as JSON.

        The model is instructed to respond with JSON only.
        """
        full_prompt = (
            prompt
            + "\n\nRespond ONLY with valid JSON. Do not include any explanation, "
            "markdown fences, or extra text — just the raw JSON object."
        )
        raw = self.generate(full_prompt, temperature=0.0)
        # Strip potential markdown code fences
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            # Remove first and last fence lines
            inner = [l for l in lines[1:] if l.strip() != "```"]
            raw = "\n".join(inner).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaError(
                f"LLM returned non-JSON response: {raw!r}"
            ) from exc
