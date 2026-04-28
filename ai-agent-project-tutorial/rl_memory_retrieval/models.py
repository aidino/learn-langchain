"""Data models for the RL Memory Retrieval pipeline."""

from dataclasses import dataclass, field


@dataclass
class MemoryItem:
    """A single memory chunk from the knowledge bank."""

    memory_id: int
    text: str
    topic: str = ""
    entity: str = ""
    slot: str = ""
    value: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryItem:
    """A query with its gold (correct) memory reference."""

    query_id: int
    query: str
    gold_memory_id: int
    gold_value: str
    topic: str = ""
    entity: str = ""
