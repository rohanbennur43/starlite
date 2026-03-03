import os
import json
import logging
import urllib.request
import urllib.error

from .provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

_ENV_KEY = "GEMINI_API_KEY"
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)
_DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the REST API.

    The API key is read from the ``GEMINI_API_KEY`` environment variable.
    No third-party SDK is required — only ``urllib`` from the stdlib.
    """

    def __init__(self, model: str = _DEFAULT_MODEL):
        self._api_key = os.environ.get(_ENV_KEY)
        if not self._api_key:
            raise LLMProviderError(
                f"Environment variable {_ENV_KEY} is not set. "
                "Obtain a key at https://aistudio.google.com/apikey"
            )
        self._model = model

    def generate_response(self, prompt: str) -> str:
        url = _GEMINI_URL.format(model=self._model, key=self._api_key)
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            logger.error("Gemini HTTP %s: %s", exc.code, detail)
            raise LLMProviderError(
                f"Gemini API returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            logger.error("Gemini network error: %s", exc.reason)
            raise LLMProviderError(
                f"Gemini network error: {exc.reason}"
            ) from exc

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected Gemini response shape: %s", data)
            raise LLMProviderError(
                f"Could not parse Gemini response: {exc}"
            ) from exc
