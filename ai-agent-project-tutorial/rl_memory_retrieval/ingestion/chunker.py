"""Recursive character text chunker.

Splits text into chunks of approximately `chunk_size` characters with
`chunk_overlap` characters of overlap between consecutive chunks.
"""

from __future__ import annotations

# Separators ordered from strongest to weakest boundary
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into chunks using recursive character splitting.

    Args:
        text: The text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []

    return _split_recursive(text, _SEPARATORS, chunk_size, chunk_overlap)


def _split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split text using the given separators."""
    # Base case: text fits in a single chunk
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator that actually appears in the text
    separator = ""
    remaining_seps = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining_seps = []
            break
        if sep in text:
            separator = sep
            remaining_seps = separators[i + 1 :]
            break

    # Split using the chosen separator
    if separator:
        pieces = text.split(separator)
    else:
        # Character-level split as last resort
        pieces = list(text)

    # Merge pieces into chunks respecting chunk_size
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        # Build the candidate chunk
        if current:
            candidate = current + separator + piece
        else:
            candidate = piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Current chunk is full — save it
            if current:
                chunks.append(current)
            # If the piece itself is too long, recursively split it
            if len(piece) > chunk_size and remaining_seps:
                sub_chunks = _split_recursive(
                    piece, remaining_seps, chunk_size, chunk_overlap
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = piece

    # Don't forget the last piece
    if current:
        chunks.append(current)

    # Apply overlap: prepend tail of previous chunk to next chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:]
            merged = overlap_text + chunks[i]
            # Trim if the overlap pushes us over chunk_size
            if len(merged) > chunk_size:
                merged = merged[:chunk_size]
            overlapped.append(merged)
        chunks = overlapped

    return chunks
