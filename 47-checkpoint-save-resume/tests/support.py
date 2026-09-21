from __future__ import annotations


def assert_plain_python_value(value: object) -> None:
    """Assert that a value uses only types accepted by weights-only loading."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for item in value:
            assert_plain_python_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AssertionError(f"non-string mapping key: {key!r}")
            assert_plain_python_value(item)
        return
    raise AssertionError(f"non-plain value: {type(value).__name__}")
