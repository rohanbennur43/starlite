import os
import json
import logging
import urllib.request
import urllib.error

from .provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"
_OLLAMA_URL = "http://localhost:11434/api/generate"


class OllamaProvider(LLMProvider):
    """Ollama provider using the local REST API.

    The model name is read from the ``OLLAMA_MODEL`` environment variable
    (default: ``"llama3"``).  No third-party SDK is required — only
    ``urllib`` from the stdlib.
    """

    def __init__(self, model: str | None = None):
        self._model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)

    def generate_response(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _OLLAMA_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            logger.error("Ollama HTTP %s: %s", exc.code, detail)
            raise LLMProviderError(
                f"Ollama API returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            logger.error("Ollama network error: %s", exc.reason)
            raise LLMProviderError(
                f"Ollama network error: {exc.reason}"
            ) from exc

        try:
            return data["response"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Ollama response shape: %s", data)
            raise LLMProviderError(
                f"Could not parse Ollama response: {exc}"
            ) from exc
