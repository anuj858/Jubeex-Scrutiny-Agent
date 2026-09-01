import pytest

from extraction_review.llm import LLMError, _parse_json


def test_parse_json_object() -> None:
    assert _parse_json('{"check_id": "D018", "status": "compliant"}') == {
        "check_id": "D018",
        "status": "compliant",
    }


def test_parse_json_fenced_block() -> None:
    raw = """```json
{"check_id": "D018", "status": "compliant"}
```"""
    assert _parse_json(raw)["check_id"] == "D018"


def test_parse_json_truncated_raises() -> None:
    with pytest.raises(LLMError, match="truncated or not JSON"):
        _parse_json(
            '{\n  "check_id": "D018",\n  "status": "compliant",\n'
        )
