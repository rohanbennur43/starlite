import json
import logging
import re
from typing import List

from .factory import LLMFactory
from .provider import LLMProviderError

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a geospatial data visualization assistant.

Dataset: {dataset}
User query: {query}

Based on the dataset name and the user's query, suggest a list of HTML
visualization page filenames that would be useful.  Each filename must:
  - use lowercase snake_case
  - end with .html
  - be descriptive of the visualization it provides

Respond with ONLY a JSON array of strings.  No explanation, no markdown
fences, no extra text.  Example:

["population_heatmap.html", "county_borders.html"]
"""


def generate_dataset_html_suggestions(
    dataset: str,
    user_query: str,
    provider_name: str = "gemini",
) -> List[str]:
    """Ask an LLM for HTML visualization filenames relevant to *dataset* and *user_query*.

    Returns:
        A list of suggested ``.html`` filenames (e.g. ``["density_view.html"]``).

    Raises:
        LLMProviderError: if the provider cannot be created or the call fails.
        ValueError: if the LLM response cannot be parsed into a filename list.
    """
    provider = LLMFactory.get_provider(provider_name)
    prompt = _PROMPT_TEMPLATE.format(dataset=dataset, query=user_query)

    raw = provider.generate_response(prompt)
    logger.debug("LLM raw response: %s", raw)

    return _parse_filename_list(raw)


def _parse_filename_list(text: str) -> List[str]:
    """Extract a JSON string list from the LLM output.

    Tolerates markdown fences and leading/trailing prose.
    """
    # Strip optional ```json ... ``` wrapping
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    # Find the first JSON array in the text
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in LLM response: {text!r}")

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in LLM response: {exc}"
        ) from exc

    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        raise ValueError(f"Expected a list of strings, got: {parsed!r}")

    # Enforce .html suffix
    return [name for name in parsed if name.endswith(".html")]
