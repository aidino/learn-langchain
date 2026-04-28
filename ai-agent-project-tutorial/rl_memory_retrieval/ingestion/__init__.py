"""Ingestion layer: loader and chunker."""

from .loader import load_source
from .chunker import chunk_text

__all__ = ["load_source", "chunk_text"]
