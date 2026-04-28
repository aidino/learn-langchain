"""Unified source loader for URLs, files, and directories."""

from __future__ import annotations

import asyncio
from pathlib import Path


def load_source(source: str) -> str:
    """Load content from a URL, file, or directory.

    Args:
        source: A URL (http/https), file path, or directory path.

    Returns:
        The content as a string (markdown format when converted).

    Raises:
        FileNotFoundError: If the file or directory does not exist.
        ValueError: If the source type cannot be determined.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return _load_url(source)
    path = Path(source)
    if path.is_dir():
        return _load_directory(path)
    if path.is_file():
        return _load_file(path)
    raise FileNotFoundError(f"Source not found: {source}")


def _load_url(url: str) -> str:
    """Load content from a web URL using Crawl4AI."""
    from crawl4ai import AsyncWebCrawler

    async def _crawl() -> str:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            return result.markdown

    return asyncio.run(_crawl())


def _load_file(path: Path) -> str:
    """Load content from a single file.

    Plain text files (.md, .txt) are read directly.
    Other formats are converted to markdown via markitdown.
    """
    if path.suffix in (".md", ".txt"):
        return path.read_text()
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(path))
    return result.markdown


def _load_directory(path: Path) -> str:
    """Load content from all supported files in a directory (recursive)."""
    supported_suffixes = {".md", ".txt", ".pdf", ".docx", ".pptx"}
    texts: list[str] = []
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix in supported_suffixes:
            texts.append(_load_file(f))
    return "\n\n".join(texts)
