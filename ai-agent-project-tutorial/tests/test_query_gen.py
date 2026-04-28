"""Tests for query generation — parsing and deduplication logic.

These tests verify the JSON parsing and dedup logic without making API calls.
"""

from rl_memory_retrieval.training.query_gen import parse_qa_response


def test_parse_valid_json():
    """Valid JSON array should be parsed correctly."""
    content = '[{"question": "What is X?", "answer": "X is Y."}]'
    result = parse_qa_response(content)
    assert len(result) == 1
    assert result[0]["question"] == "What is X?"
    assert result[0]["answer"] == "X is Y."


def test_parse_multiple_pairs():
    """Multiple QA pairs should all be parsed."""
    content = """[
        {"question": "Q1?", "answer": "A1"},
        {"question": "Q2?", "answer": "A2"},
        {"question": "Q3?", "answer": "A3"}
    ]"""
    result = parse_qa_response(content)
    assert len(result) == 3


def test_parse_with_code_fence():
    """JSON wrapped in markdown code fence should be parsed."""
    content = """```json
[{"question": "What?", "answer": "That."}]
```"""
    result = parse_qa_response(content)
    assert len(result) == 1
    assert result[0]["question"] == "What?"


def test_parse_with_surrounding_text():
    """JSON array embedded in surrounding text should be extracted."""
    content = """Here are the QA pairs:
[{"question": "How?", "answer": "Like this."}]
That's all."""
    result = parse_qa_response(content)
    assert len(result) == 1


def test_parse_empty_response():
    """Empty response should return empty list."""
    assert parse_qa_response("") == []
    assert parse_qa_response("no json here") == []


def test_parse_invalid_json():
    """Invalid JSON should return empty list."""
    assert parse_qa_response("[{broken json}]") == []


def test_parse_missing_keys():
    """Items missing required keys should be filtered out."""
    content = """[
        {"question": "Valid?", "answer": "Yes"},
        {"question": "No answer key"},
        {"other_key": "value"}
    ]"""
    result = parse_qa_response(content)
    assert len(result) == 1
    assert result[0]["question"] == "Valid?"
