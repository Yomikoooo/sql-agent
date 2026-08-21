r"""Regression tests for redaction. The point is the negative cases.

Why this file has to exist: `redact.py` is down to a single email pattern, but
it once also carried a phone pattern, `\+?\d[\d\s-]{7,}\d`, which swallowed
dates like `2026-08-15` into `<PHONE>`.  That lesson lived only in a commit
message; nothing in the code guarded it.

Adding to PATTERNS is a planned change, not a hypothetical one (column-level
redaction of `users.name` before pointing this at a real database), so the
lesson is due to be paid for twice unless something holds it down.

Redaction is a substitution, so writing a pattern too wide is destructive and
symptom-free: the messages in `results/*.jsonl` are the raw material for
rejection sampling, and a date quietly rewritten to `<PHONE>` raises nothing --
it just spoils the batch.
"""
import copy

from sql_agent.redact import PATTERNS, redact_record, redact_text

# Run this table before adding any pattern. These strings must survive untouched.
MUST_SURVIVE = [
    "2026-08-15",                                   # the one that actually broke
    "2026-08-15T19-08-40",                          # the run_id format under results/
    "WHERE o.created_at BETWEEN '2026-01-01' AND '2026-12-31'",
    "strftime('%Y-%m', o.created_at)",
    "1234567890",                                   # an order id or an amount, not a phone number
    "SELECT COUNT(*) FROM orders WHERE total > 100",
    "sql_024 hit the 8 turn ceiling",               # a task id inside prose
]

MUST_BE_REDACTED = [
    "user7@example.com",
    "WHERE email = 'liu.lei+tag@sub.example.co.uk'",
    "Müller a.b-c@example.org danke",             # non-ASCII prose around the address
]


def test_things_that_must_not_be_touched():
    """This is the body of the file.

    Testing only that emails get replaced would pass an implementation that
    rewrites every character to `<EMAIL>`.  The negative cases are the half
    that pins down "not too wide".
    """
    for s in MUST_SURVIVE:
        assert redact_text(s) == s, f"damaged: {s!r} -> {redact_text(s)!r}"


def test_things_that_must_be_replaced():
    for s in MUST_BE_REDACTED:
        out = redact_text(s)
        assert "@" not in out, f"not fully redacted: {s!r} -> {out!r}"
        assert "<EMAIL>" in out


def test_the_placeholder_keeps_the_sentence_readable():
    """The placeholder cannot be a length or a hash.

    The messages are training material; replacing an address with something
    like `<len:17>` strips the sentence of its meaning.
    """
    assert redact_text("mail me at a@b.com ok?") == "mail me at <EMAIL> ok?"


def test_redact_record_reaches_every_nesting_level():
    """PII shows up in the answer, the question and the SQL, so a single field is never enough."""
    record = {"question": "who is a@b.com?",
              "messages": [{"role": "tool", "content": "email: c@d.com"}],
              "nested": {"deep": ["e@f.com"]}}
    assert "@" not in str(redact_record(record))


def test_redact_record_leaves_non_strings_alone():
    """Numbers and None come back untouched: turning an int into a str would silently make downstream stats compare strings."""
    record = {"passed": True, "turns": 3, "elapsed": 1.5, "last_sql": None}
    assert redact_record(record) == record


def test_redact_record_does_not_mutate_the_original():
    """Grading and statistics read the original data; writing to disk is the boundary.

    Redacting in place would also change the copy in memory -- the one the
    grader is reading.
    """
    record = {"question": "who is a@b.com?", "messages": [{"content": "x@y.com"}]}
    before = copy.deepcopy(record)
    redact_record(record)
    assert record == before, "redact_record mutated its input"


def test_every_pattern_is_checked_against_the_survivor_list():
    """For the future: however many entries PATTERNS grows to, the survivor table must still pass.

    Adding a pattern re-runs the table above automatically -- the step that
    was skipped the first time.
    """
    assert PATTERNS, "PATTERNS is empty, so nothing is being redacted"
    for s in MUST_SURVIVE:
        for pattern, _placeholder in PATTERNS:
            assert not pattern.search(s), \
                f"pattern {pattern.pattern!r} matched {s!r}, which must be left alone"
