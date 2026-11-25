"""Tests for usage metering (Step 6). All dates pinned; no clock trust."""

from datetime import date

import pytest

from provome.usage import UsageLedger, decide

DAY = date(2026, 11, 25)
NEXT = date(2026, 11, 26)


def test_record_and_used_per_day_and_kind():
    ledger = UsageLedger()
    assert ledger.used("acct", "live_questions", on=DAY) == 0
    ledger.record("acct", "live_questions", on=DAY)
    ledger.record("acct", "live_questions", amount=2, on=DAY)
    ledger.record("acct", "batch_variants", amount=50, on=DAY)
    assert ledger.used("acct", "live_questions", on=DAY) == 3
    assert ledger.used("acct", "batch_variants", on=DAY) == 50
    # A different day is a different count.
    assert ledger.used("acct", "live_questions", on=NEXT) == 0
    # A different account is a different count.
    assert ledger.used("other", "live_questions", on=DAY) == 0


def test_remaining_and_unlimited():
    ledger = UsageLedger()
    ledger.record("acct", "live_questions", amount=3, on=DAY)
    assert ledger.remaining("acct", "live_questions", 10, on=DAY) == 7
    assert ledger.remaining("acct", "live_questions", 2, on=DAY) == 0
    assert ledger.remaining("acct", "live_questions", None, on=DAY) is None


def test_decide_allows_then_denies_with_reset_message():
    ledger = UsageLedger()
    first = decide(ledger, "acct", 2, "live_questions", on=DAY)
    assert first.allowed and first.remaining_after == 1
    ledger.record("acct", "live_questions", on=DAY)
    ledger.record("acct", "live_questions", on=DAY)
    denied = decide(ledger, "acct", 2, "live_questions", on=DAY)
    assert not denied.allowed
    assert denied.used == 2 and denied.limit == 2
    assert "limit reached" in denied.message.lower()
    assert NEXT.isoformat() in denied.message  # names the reset day
    # decide() itself records nothing.
    assert ledger.used("acct", "live_questions", on=DAY) == 2


def test_decide_batch_amount_and_unlimited_tier():
    ledger = UsageLedger()
    big = decide(ledger, "acct", 100, "batch_variants", amount=250, on=DAY)
    assert not big.allowed  # one batch can exceed the whole limit
    unlimited = decide(ledger, "acct", None, "batch_variants", amount=10**6, on=DAY)
    assert unlimited.allowed and unlimited.remaining_after is None


def test_unknown_kind_rejected_everywhere():
    ledger = UsageLedger()
    with pytest.raises(ValueError):
        ledger.record("acct", "emails_sent", on=DAY)
    with pytest.raises(ValueError):
        ledger.used("acct", "emails_sent", on=DAY)
    with pytest.raises(ValueError):
        decide(ledger, "acct", 5, "emails_sent", on=DAY)


def test_save_load_round_trip(tmp_path):
    ledger = UsageLedger()
    ledger.record("acct", "live_questions", amount=4, on=DAY)
    ledger.record("acct", "batch_variants", amount=120, on=NEXT)
    ledger.record("other", "live_questions", amount=1, on=DAY)
    path = tmp_path / "usage.json"
    ledger.save(path)
    loaded = UsageLedger.load(path)
    assert loaded.to_dict() == ledger.to_dict()
    assert loaded.used("acct", "live_questions", on=DAY) == 4
    assert loaded.used("acct", "batch_variants", on=NEXT) == 120
