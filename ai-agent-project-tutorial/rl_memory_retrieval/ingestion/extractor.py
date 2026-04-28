"""Optional metadata extraction using langextract.

Extracts structured metadata (entities, topics, key facts) from text chunks.
When extraction is disabled (default), returns empty metadata dicts.
"""

from __future__ import annotations

import os
from typing import Any

from tqdm import tqdm

from rl_memory_retrieval.models import MemoryItem


def extract_metadata(
    items: list[MemoryItem],
    instruction: str = "Extract entities, topics, and key facts",
    model_id: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> list[MemoryItem]:
    """Enrich MemoryItems with extracted entities, topics, and metadata.

    Uses langextract to pull structured information from each chunk's text.
    Updates the `entity`, `topic`, and `metadata` fields in place.

    Args:
        items: List of MemoryItems to enrich.
        instruction: Extraction prompt for langextract.
        model_id: The LLM model to use for extraction.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

    Returns:
        The same list of MemoryItems with updated metadata fields.
    """
    import langextract as lx

    key = api_key or os.environ.get("OPENAI_API_KEY", "")

    for item in tqdm(items, desc="Extracting metadata"):
        try:
            result = lx.extract(
                text_or_documents=item.text,
                prompt_description=instruction,
                model_id=model_id,
                api_key=key,
                fence_output=True,
                use_schema_constraints=False,
            )
            parsed = _parse_extraction(result)
            item.entity = parsed.get("entity", "")
            item.topic = parsed.get("topic", "")
            item.metadata = parsed
        except Exception:
            # Extraction failure is non-fatal; leave fields at defaults
            pass

    return items


def _parse_extraction(result: Any) -> dict[str, Any]:
    """Parse the langextract result into a flat metadata dict.

    langextract returns various structures depending on the prompt.
    We attempt to extract entity and topic fields in a best-effort manner.

    Args:
        result: The raw langextract result.

    Returns:
        A dict with at minimum 'entity' and 'topic' keys.
    """
    metadata: dict[str, Any] = {}

    # langextract may return a dict, list, or string
    if isinstance(result, dict):
        metadata = dict(result)
    elif isinstance(result, list) and len(result) > 0:
        # Take the first result if it's a list of dicts
        if isinstance(result[0], dict):
            metadata = dict(result[0])
        else:
            metadata["raw"] = result
    elif isinstance(result, str):
        metadata["raw"] = result

    # Normalize entity/topic fields
    entity = metadata.get("entity", metadata.get("entities", ""))
    if isinstance(entity, list):
        entity = ", ".join(str(e) for e in entity)
    metadata["entity"] = str(entity)

    topic = metadata.get("topic", metadata.get("topics", ""))
    if isinstance(topic, list):
        topic = ", ".join(str(t) for t in topic)
    metadata["topic"] = str(topic)

    return metadata
