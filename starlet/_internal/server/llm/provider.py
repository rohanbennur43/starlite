from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base interface for LLM providers.

    Every concrete provider (Gemini, OpenAI, Claude, etc.) must implement
    ``generate_response`` so that callers remain provider-agnostic.
    """

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Send *prompt* to the underlying LLM and return the text response.

        Raises:
            LLMProviderError: on any communication or API failure.
        """


class LLMProviderError(Exception):
    """Raised when an LLM provider fails to produce a response."""
