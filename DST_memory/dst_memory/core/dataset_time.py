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


def parse_longmemeval_question_date_to_iso(raw: Any) -> Optional[str]:
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


def optional_clock_display_for_validation(
    use_dataset_datetime: bool,
    question_date_raw: Any,
) -> Optional[str]:
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
    print("dataset_time OK:", len(_samples), "samples")
