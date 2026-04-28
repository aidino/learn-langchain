"""Ingestion layer: loader, chunker, and optional extractor."""

from .loader import load_source
from .chunker import chunk_text
from .extractor import extract_metadata

__all__ = ["load_source", "chunk_text", "extract_metadata"]

