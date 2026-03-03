import logging
import os

from .provider import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER = "gemini"

# Registry of provider name -> callable that returns an LLMProvider instance.
# Each entry is a zero-arg factory so that imports and API-key validation are
# deferred until the provider is actually requested.
_PROVIDERS = {
    "gemini": lambda: _make_gemini(),
    "ollama": lambda: _make_ollama(),
}


def _make_gemini() -> LLMProvider:
    from .gemini_provider import GeminiProvider
    return GeminiProvider()


def _make_ollama() -> LLMProvider:
    from .ollama_provider import OllamaProvider
    return OllamaProvider()


class LLMFactory:
    """Instantiate an :class:`LLMProvider` by name.

    Supported names (case-insensitive):
        * ``"gemini"`` — Google Gemini (requires ``GEMINI_API_KEY``)
        * ``"ollama"`` — Local Ollama (uses ``OLLAMA_MODEL``, default ``llama3``)

    Future providers (``"openai"``, ``"claude"``, ...) can be added by
    registering an entry in ``_PROVIDERS``.
    """

    @staticmethod
    def get_provider(name: str) -> LLMProvider:
        """Return a ready-to-use provider instance.

        Raises:
            LLMProviderError: if *name* is unknown or construction fails.
        """
        key = name.strip().lower()
        builder = _PROVIDERS.get(key)
        if builder is None:
            supported = ", ".join(sorted(_PROVIDERS))
            raise LLMProviderError(
                f"Unknown LLM provider '{name}'. Supported: {supported}"
            )
        return builder()

    @staticmethod
    def get_default_provider() -> LLMProvider:
        """Return a provider selected by the ``LLM_PROVIDER`` env var.

        Falls back to ``"gemini"`` when the variable is unset or names an
        unknown provider.
        """
        name = os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
        if name not in _PROVIDERS:
            logger.warning(
                "Unknown LLM_PROVIDER '%s', falling back to '%s'",
                name, _DEFAULT_PROVIDER,
            )
            name = _DEFAULT_PROVIDER
        return LLMFactory.get_provider(name)
