"""
Unit tests for core data models: FactRecord, is_expired(), TTL logic.

Pure Python — no LLM, no GPU.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from dst_memory.core.models import (
    VALID_TTL_VALUES,
    FactRecord,
    is_expired,
    now_iso,
)

# ---------------------------------------------------------------------------
# VALID_TTL_VALUES
# ---------------------------------------------------------------------------


def test_valid_ttl_values_non_empty() -> None:
    assert len(VALID_TTL_VALUES) > 0
    assert "inf" in VALID_TTL_VALUES
    assert "1d" in VALID_TTL_VALUES
    assert "1y" in VALID_TTL_VALUES


# ---------------------------------------------------------------------------
# is_expired() standalone function
# ---------------------------------------------------------------------------


class TestIsExpired:
    def test_inf_never_expires(self) -> None:
        past = (datetime.now() - timedelta(days=3650)).isoformat()
        assert not is_expired("inf", past)

    def test_expired_1d(self) -> None:
        two_days_ago = (datetime.now() - timedelta(days=2)).isoformat()
        assert is_expired("1d", two_days_ago)

    def test_not_expired_1d_recent(self) -> None:
        one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        assert not is_expired("1d", one_hour_ago)

    def test_expired_1m(self) -> None:
        two_months_ago = (datetime.now() - timedelta(days=60)).isoformat()
        assert is_expired("1m", two_months_ago)

    def test_not_expired_3m_recent(self) -> None:
        one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        assert not is_expired("3m", one_week_ago)

    def test_expired_with_as_of(self) -> None:
        created = datetime(2020, 1, 1).isoformat()
        as_of = datetime(2020, 1, 5)  # 4 days later
        assert is_expired("1d", created, as_of=as_of)

    def test_not_expired_with_as_of(self) -> None:
        created = datetime(2020, 1, 1).isoformat()
        as_of = datetime(2020, 1, 1, 12)  # 12 hours later
        assert not is_expired("1d", created, as_of=as_of)

    def test_empty_created_at_not_expired(self) -> None:
        assert not is_expired("1d", "")

    def test_invalid_created_at_not_expired(self) -> None:
        assert not is_expired("1d", "not-a-date")

    def test_unknown_ttl_not_expired(self) -> None:
        past = (datetime.now() - timedelta(days=999)).isoformat()
        assert not is_expired("bad_ttl", past)

    @pytest.mark.parametrize("ttl", list(VALID_TTL_VALUES - {"inf"}))
    def test_all_finite_ttls_expire_eventually(self, ttl: str) -> None:
        very_old = (datetime.now() - timedelta(days=3650)).isoformat()
        assert is_expired(ttl, very_old), f"TTL {ttl!r} should expire after 10 years"


# ---------------------------------------------------------------------------
# FactRecord
# ---------------------------------------------------------------------------


class TestFactRecord:
    def make_record(self, ttl: str = "inf", days_old: int = 0) -> FactRecord:
        created = (datetime.now() - timedelta(days=days_old)).isoformat()
        return FactRecord(
            record_id=1,
            value="пользователь | живёт в | москва",
            source_text="Я живу в Москве",
            created_at_step=1,
            updated_at_step=1,
            subject="пользователь",
            relation="живёт в",
            object="москва",
            ttl=ttl,
            created_at_datetime=created,
        )

    def test_record_not_expired_inf(self) -> None:
        rec = self.make_record(ttl="inf", days_old=9999)
        assert not rec.is_expired()

    def test_record_expired_1d(self) -> None:
        rec = self.make_record(ttl="1d", days_old=5)
        assert rec.is_expired()

    def test_record_not_expired_recent(self) -> None:
        rec = self.make_record(ttl="1m", days_old=1)
        assert not rec.is_expired()

    def test_refresh_ttl_resets_timer(self) -> None:
        rec = self.make_record(ttl="1d", days_old=5)
        assert rec.is_expired()
        rec.refresh_ttl()
        assert not rec.is_expired()

    def test_as_line_format(self) -> None:
        rec = self.make_record()
        assert rec.as_line() == "пользователь | живёт в | москва"

    def test_default_is_active(self) -> None:
        rec = self.make_record()
        assert rec.is_active is True

    def test_record_fields(self) -> None:
        rec = self.make_record(ttl="3m")
        assert rec.subject == "пользователь"
        assert rec.relation == "живёт в"
        assert rec.object == "москва"
        assert rec.ttl == "3m"
        assert rec.record_id == 1


# ---------------------------------------------------------------------------
# now_iso
# ---------------------------------------------------------------------------


def test_now_iso_is_valid_datetime() -> None:
    iso = now_iso()
    dt = datetime.fromisoformat(iso)
    assert dt is not None
    # Should be close to now
    delta = abs((datetime.now() - dt).total_seconds())
    assert delta < 5
