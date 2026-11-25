"""Usage metering (Step 6): the ledger behind the freemium tiers.

Metering only.  This module counts how much of the product an
account has used today — live questions asked, batch variants
processed — and answers one honest question: is there room left
under this tier's limit?  It does not bill, charge, or pay
anything; there is no money anywhere in this file.  Counts are
local, kept per account per day, and auditable: the whole ledger
serialises to a plain JSON file a human can open and check.

Design rules:

* A denial is never silent.  When a limit is reached,
  :func:`decide` returns a message that says so and names the
  reset (the next day).  Dropping a request without telling the
  user would be exactly the kind of quiet failure Provome exists
  to avoid.
* Limits come from the caller (the tier definition); this module
  only counts and compares.  A limit of ``None`` means unlimited
  and is reported as unlimited, never as a big made-up number.
* Days are explicit.  Every method takes an optional date so
  tests — and audits — can pin the clock instead of trusting it.

Run the demo (a demo account with a limit of 2 asks 3 questions;
the third is denied, with its message printed):

    python -m provome.usage

Research software.  Not for clinical use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Union

KINDS = ("live_questions", "batch_variants")

PathLike = Union[str, Path]


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(
            f"unknown usage kind {kind!r}; expected one of {', '.join(KINDS)}"
        )
    return kind


def _check_amount(amount: int) -> int:
    if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
        raise ValueError(f"amount must be a non-negative integer, got {amount!r}")
    return amount


class UsageLedger:
    """Per-account, per-day usage counts, in memory and on disk.

    Counts are keyed by (account_id, kind, ISO date).  The ledger
    is deliberately dumb: it records what it is told, reports what
    it holds, and round-trips through JSON without loss.
    """

    def __init__(self, counts: Optional[dict] = None) -> None:
        self._counts: dict = dict(counts) if counts else {}

    # -- recording ------------------------------------------------------
    def record(
        self,
        account_id: str,
        kind: str,
        amount: int = 1,
        on: Optional[date] = None,
    ) -> int:
        """Add ``amount`` to today's count; return the new total."""
        _check_kind(kind)
        _check_amount(amount)
        day = (on or date.today()).isoformat()
        key = (account_id, kind, day)
        self._counts[key] = self._counts.get(key, 0) + amount
        return self._counts[key]

    # -- reading ----------------------------------------------------------
    def used(
        self,
        account_id: str,
        kind: str,
        on: Optional[date] = None,
    ) -> int:
        """How much ``account_id`` has used of ``kind`` on that day."""
        _check_kind(kind)
        day = (on or date.today()).isoformat()
        return self._counts.get((account_id, kind, day), 0)

    def remaining(
        self,
        account_id: str,
        kind: str,
        limit: Optional[int],
        on: Optional[date] = None,
    ) -> Optional[int]:
        """Room left under ``limit``; ``None`` when unlimited."""
        if limit is None:
            return None
        return max(limit - self.used(account_id, kind, on=on), 0)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise to plain JSON-compatible data."""
        return {
            "version": 1,
            "counts": [
                {
                    "account_id": account_id,
                    "kind": kind,
                    "date": day,
                    "count": count,
                }
                for (account_id, kind, day), count in sorted(self._counts.items())
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "UsageLedger":
        counts = {}
        for row in payload.get("counts", []):
            _check_kind(row["kind"])
            counts[(row["account_id"], row["kind"], row["date"])] = int(row["count"])
        return cls(counts)

    def save(self, path: PathLike) -> None:
        """Write the ledger to ``path`` as human-readable JSON."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: PathLike) -> "UsageLedger":
        """Read a ledger previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass(frozen=True)
class QuotaDecision:
    """The answer to "may this account do this much of this, today?"

    ``message`` is always human-readable and always honest: an
    approval says how much room remains (or that the tier is
    unlimited); a denial says the limit was reached and when it
    resets.
    """

    allowed: bool
    account_id: str
    kind: str
    amount: int
    used: int
    limit: Optional[int]
    on: date
    message: str

    @property
    def remaining_after(self) -> Optional[int]:
        """Room left if this decision is acted on; None if unlimited."""
        if self.limit is None:
            return None
        projected = self.used + (self.amount if self.allowed else 0)
        return max(self.limit - projected, 0)


def decide(
    ledger: UsageLedger,
    account_id: str,
    tier_limit: Optional[int],
    kind: str,
    amount: int = 1,
    on: Optional[date] = None,
) -> QuotaDecision:
    """Decide whether ``amount`` of ``kind`` fits under the tier limit.

    This function only decides; it records nothing.  The caller
    records with :meth:`UsageLedger.record` once the work is
    actually done, so a denied or abandoned request never burns
    quota.
    """
    _check_kind(kind)
    _check_amount(amount)
    day = on or date.today()
    used = ledger.used(account_id, kind, on=day)
    if tier_limit is None:
        return QuotaDecision(
            allowed=True,
            account_id=account_id,
            kind=kind,
            amount=amount,
            used=used,
            limit=None,
            on=day,
            message=(
                f"Allowed: the tier places no daily limit on {kind} "
                f"({used} used today)."
            ),
        )
    if used + amount <= tier_limit:
        left = tier_limit - used - amount
        return QuotaDecision(
            allowed=True,
            account_id=account_id,
            kind=kind,
            amount=amount,
            used=used,
            limit=tier_limit,
            on=day,
            message=(
                f"Allowed: {used + amount} of {tier_limit} {kind} "
                f"used today after this ({left} remaining)."
            ),
        )
    reset = day + timedelta(days=1)
    return QuotaDecision(
        allowed=False,
        account_id=account_id,
        kind=kind,
        amount=amount,
        used=used,
        limit=tier_limit,
        on=day,
        message=(
            f"Daily limit reached: {used} of {tier_limit} {kind} already "
            f"used today ({day.isoformat()}). The limit resets tomorrow, "
            f"{reset.isoformat()}. Nothing was recorded for this request."
        ),
    )


def _demo() -> None:
    ledger = UsageLedger()
    account = "demo-account"
    limit = 2
    print(f"Demo: {account}, daily limit {limit} live_questions")
    for attempt in (1, 2, 3):
        decision = decide(ledger, account, limit, "live_questions")
        print(f"  Question {attempt}: {decision.message}")
        if decision.allowed:
            ledger.record(account, "live_questions")
    print(f"  Recorded usage today: {ledger.used(account, 'live_questions')}")


if __name__ == "__main__":
    _demo()
