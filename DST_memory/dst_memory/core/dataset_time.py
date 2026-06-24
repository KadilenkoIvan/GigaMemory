"""
LongMemEval-style question_date parsing for simulated wall clock in validation.

Typical LongMemEval formats::

    2023/05/20 (Sat) 02:21
    2023/05/30 (Tue) 23:40

Also accepted: no weekday (``2023/05/20 02:21``), optional spaces, and compact ``2023/05/20(Tue)23:40``.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

# Date, then one of: " (Dow) HH:MM" | "(Dow)HH:MM" | " HH:MM"
_LMEM_DATE_RE = re.compile(
    r"^\s*(\d{4}/\d{2}/\d{2})(?:\s+\([^)]+\)\s+|\([^)]+\)\s*|\s+)(\d{1,2}:\d{2})\s*$",
)


def parse_longmemeval_question_date_to_iso(raw: Any) -> str | None:
    """
    Parse LongMemEval ``question_date`` into ISO 8601 string (naive local semantics).

    Returns None if missing or unparseable.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Already ISO-like
    for candidate in (s, s.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate).isoformat()
        except ValueError:
            continue

    m = _LMEM_DATE_RE.match(s)
    if m:
        date_part, time_part = m.group(1), m.group(2)
        ymd = date_part.replace("/", "-")
        try:
            dt = datetime.strptime(f"{ymd} {time_part}", "%Y-%m-%d %H:%M")
            return dt.isoformat()
        except ValueError:
            return None

    # Plain YYYY/MM/DD without time → midnight
    m2 = re.match(r"^\s*(\d{4}/\d{2}/\d{2})\s*$", s)
    if m2:
        ymd = m2.group(1).replace("/", "-")
        try:
            return datetime.strptime(ymd, "%Y-%m-%d").isoformat()
        except ValueError:
            return None

    return None


def format_clock_for_final_llm_prompt(iso_str: str) -> str:
    """Same visual style as ``FinalLLMClient`` default: ``%Y-%m-%d %H:%M``."""
    dt = datetime.fromisoformat(iso_str)
    return dt.strftime("%Y-%m-%d %H:%M")


def fact_clock_iso_for_haystack_session(
    use_dataset_datetime: bool,
    haystack_dates: Any,
    session_index: int,
    question_date_raw: Any,
) -> str | None:
    """
    ISO timestamp for facts extracted from ``haystack_sessions[session_index]``.

    Uses ``haystack_dates[session_index]`` when present and parseable; otherwise
    falls back to ``question_date_raw``. Returns None when ``use_dataset_datetime``
    is False (caller should use wall clock in DST).
    """
    if not use_dataset_datetime:
        return None
    raw: Any = None
    if isinstance(haystack_dates, list) and 0 <= session_index < len(haystack_dates):
        raw = haystack_dates[session_index]
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        raw = question_date_raw
    return parse_longmemeval_question_date_to_iso(raw)


def optional_clock_display_for_validation(
    use_dataset_datetime: bool,
    question_date_raw: Any,
) -> str | None:
    """If dataset-time mode is on and ``question_date`` parses, return prompt clock string."""
    if not use_dataset_datetime:
        return None
    iso = parse_longmemeval_question_date_to_iso(question_date_raw)
    if not iso:
        return None
    return format_clock_for_final_llm_prompt(iso)


if __name__ == "__main__":
    _samples = [
        "2023/05/20 (Sat) 02:21",
        "  2023/05/20 (Sat) 02:21  ",
        "2023/05/30 (Tue) 23:40",
        "2023/05/20(Tue)23:40",
        "2023/05/20 02:21",
    ]
    for _s in _samples:
        _iso = parse_longmemeval_question_date_to_iso(_s)
        assert _iso is not None, _s
        datetime.fromisoformat(_iso)
    assert parse_longmemeval_question_date_to_iso("nope") is None
    _hds = ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 03:24"]
    assert fact_clock_iso_for_haystack_session(  # type: ignore[union-attr]
        True, _hds, 0, "2023/05/22 (Mon) 10:00"
    ).startswith("2023-05-20")
    assert fact_clock_iso_for_haystack_session(  # type: ignore[union-attr]
        True, _hds, 1, "2023/05/22 (Mon) 10:00"
    ).startswith("2023-05-21")
    assert fact_clock_iso_for_haystack_session(  # type: ignore[union-attr]
        True, _hds, 9, "2023/05/22 (Mon) 10:00"
    ).startswith("2023-05-22")
    assert fact_clock_iso_for_haystack_session(False, _hds, 0, "x") is None
    print("dataset_time OK:", len(_samples), "samples + haystack session clocks")
