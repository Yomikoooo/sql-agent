"""Replace PII in text on its way to disk or off the machine.

Placeholders, not lengths or hashes: the full `messages` in results/*.jsonl
are training material for rejection sampling, so redaction has to preserve
structure and meaning and swap only the identifier itself.

Redaction is a *replacing* operation, which makes a too-wide pattern
destructive and silent -- see tests/test_redact.py for the cases that must
survive untouched.
"""
import re

PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
]


def redact_text(s: str) -> str:
    for pattern, placeholder in PATTERNS:
        s = pattern.sub(placeholder, s)
    return s


def redact_record(obj):
    """Redact every string in an arbitrarily nested structure, returning a
    new object.

    The whole record, not just `messages`: PII also shows up in the agent's
    answer, in the question, and inside `WHERE email = '...'` SQL. Redacting
    one field is guaranteed to miss some.

    A new object rather than an in-place edit: scoring and statistics read the
    original data, and writing to disk is where the boundary actually is.
    """
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact_record(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_record(x) for x in obj]
    return obj  # int / float / bool / None