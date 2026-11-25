"""Subscription tiers and the entitlement model (Step 6).

This module is the product's entitlement model: it names the three
tiers the Founder set pricing for, states exactly what each tier
unlocks, and answers the one question every feature gate in the
product asks — *is this account allowed to do this, and if not,
why not, in words a user can act on?*

The shape follows the freemium pattern of the tools researchers
already live in: the Free tier keeps the core evidence tools open
to every student and researcher, and the paid tiers unlock depth —
volume, batch work, the audit trail, team seats, and priority
support.  The cite-or-decline gate itself is never paywalled: a
free answer and a Lab answer pass exactly the same evidence bar.
Tiers govern *how much* and *how conveniently*, never *how true*.

What this module is not: a billing system.  Checkout, payment
processing, invoices, and subscription lifecycle are NOT
implemented here and are not claimed anywhere in the product —
they belong to Step 8 (pilots + money), which fires only after
the company foundation items (formation, contracts, insurance)
are in place.  Until then, tiers exist as a model and a gate,
and no payment claim is made to anyone.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "Tier",
    "Entitlements",
    "entitlements_for",
    "check",
    "FEATURES",
]


class Tier(str, Enum):
    """The three subscription tiers, in ascending order.

    The enum *value* is the tier's monthly price in whole US
    dollars, so ``Tier.RESEARCHER.value == 79`` and the price can
    never drift away from the tier it belongs to.  Ordering
    helpers compare by that price.
    """

    FREE = 0
    RESEARCHER = 79
    LAB = 499

    @property
    def monthly_price_usd(self) -> int:
        """Monthly price in whole US dollars."""
        return int(self.value)

    @property
    def display_name(self) -> str:
        return self.name.capitalize()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return self.monthly_price_usd < other.monthly_price_usd


@dataclass(frozen=True)
class Entitlements:
    """What one tier unlocks.  Frozen: an entitlement set is a
    fact about a tier, not a working document.

    ``None`` on a numeric limit means *unlimited* — the field is
    deliberately not a huge sentinel number, so a limit check can
    never be passed by accident with a bigger integer.
    """

    tier: Tier
    #: Live Copilot questions per day; ``None`` = unlimited.
    max_live_questions_per_day: Optional[int]
    #: Variants accepted in one batch run; ``None`` = unlimited.
    max_batch_variants: Optional[int]
    #: Single-question Drug Discovery access (all tiers: the free
    #: tier keeps core tools open; batch drug work rides the
    #: batch limit above, like every other batch feature).
    drug_discovery_access: bool
    #: Whether the account can read the full audit trail of a run.
    audit_trail_access: bool
    #: Seats included for a team workspace.
    team_seats: int
    #: Priority support channel.
    priority_support: bool


_ENTITLEMENTS: dict[Tier, Entitlements] = {
    Tier.FREE: Entitlements(
        tier=Tier.FREE,
        max_live_questions_per_day=10,
        max_batch_variants=25,
        drug_discovery_access=True,
        audit_trail_access=False,
        team_seats=1,
        priority_support=False,
    ),
    Tier.RESEARCHER: Entitlements(
        tier=Tier.RESEARCHER,
        max_live_questions_per_day=500,
        max_batch_variants=1000,
        drug_discovery_access=True,
        audit_trail_access=True,
        team_seats=1,
        priority_support=False,
    ),
    Tier.LAB: Entitlements(
        tier=Tier.LAB,
        max_live_questions_per_day=None,
        max_batch_variants=None,
        drug_discovery_access=True,
        audit_trail_access=True,
        team_seats=25,
        priority_support=True,
    ),
}


def entitlements_for(tier: Tier) -> Entitlements:
    """Return the entitlement set for ``tier``.

    Raises ``TypeError`` for anything that is not a :class:`Tier`,
    so a stray string or price integer can never silently resolve
    to someone's access level.
    """
    if not isinstance(tier, Tier):
        raise TypeError(f"tier must be a Tier, got {type(tier).__name__}")
    return _ENTITLEMENTS[tier]


#: Features a caller may ask :func:`check` about.  Each maps to the
#: Entitlements field that governs it.
FEATURES = (
    "live_questions",
    "batch_variants",
    "drug_discovery",
    "audit_trail",
    "team_seats",
    "priority_support",
)

#: Numeric features and the Entitlements field holding their limit.
_NUMERIC_FIELDS = {
    "live_questions": "max_live_questions_per_day",
    "batch_variants": "max_batch_variants",
    "team_seats": "team_seats",
}


def _allows(ent: Entitlements, feature: str, amount: int) -> bool:
    """Pure allow/deny predicate — no reasons, no recursion.

    :func:`check` builds its human-readable answers on top of this,
    and :func:`_cheapest_tier_granting` uses it directly, so the
    upgrade search can never recurse back into :func:`check`.
    """
    if feature in _NUMERIC_FIELDS:
        limit = getattr(ent, _NUMERIC_FIELDS[feature])
        return limit is None or amount <= limit
    return {
        "drug_discovery": ent.drug_discovery_access,
        "audit_trail": ent.audit_trail_access,
        "priority_support": ent.priority_support,
    }[feature]


def _cheapest_tier_granting(feature: str, amount: int) -> Optional[Tier]:
    """The lowest-priced tier whose entitlements allow ``feature``
    at ``amount``, or ``None`` if no tier does."""
    for tier in sorted(Tier, key=lambda t: t.monthly_price_usd):
        if _allows(entitlements_for(tier), feature, amount):
            return tier
    return None


def check(
    ent: Entitlements,
    feature: str,
    amount: int = 1,
) -> tuple[bool, str]:
    """Ask whether ``ent`` allows ``feature`` at ``amount``.

    Returns ``(allowed, reason)``.  The reason is written for the
    user, not the log: on a denial it names the current tier, the
    limit or missing access, and the cheapest tier that unlocks
    what was asked for.  On an allowance it states the headroom.

    ``amount`` is the size of the request — questions asked today,
    variants in a batch, seats wanted.  For boolean features it is
    ignored beyond needing to be a positive integer.
    """
    if not isinstance(ent, Entitlements):
        raise TypeError("ent must be an Entitlements")
    if feature not in FEATURES:
        raise ValueError(
            f"unknown feature {feature!r}; known features: {', '.join(FEATURES)}"
        )
    if not isinstance(amount, int) or amount < 1:
        raise ValueError("amount must be a positive integer")

    tier_name = ent.tier.display_name

    if feature in _NUMERIC_FIELDS:
        limit = getattr(ent, _NUMERIC_FIELDS[feature])
        noun, unit = {
            "live_questions": ("live questions", " per day"),
            "batch_variants": ("variants in one batch", ""),
            "team_seats": ("team seats", ""),
        }[feature]
        if limit is None:
            return True, f"The {tier_name} tier has unlimited {noun}."
        if amount <= limit:
            return (
                True,
                f"Allowed on the {tier_name} tier: {amount} of "
                f"{limit} {noun}{unit}.",
            )
        upgrade = _cheapest_tier_granting(feature, amount)
        if upgrade is not None and upgrade is not ent.tier:
            upgrade_limit = getattr(
                entitlements_for(upgrade), _NUMERIC_FIELDS[feature]
            )
            effect = (
                "removes that limit"
                if upgrade_limit is None
                else f"raises it to {upgrade_limit}"
            )
            return (
                False,
                f"The {tier_name} tier allows {limit} {noun}{unit}, "
                f"but {amount} were requested. The "
                f"{upgrade.display_name} tier {effect}.",
            )
        return (
            False,
            f"The {tier_name} tier allows {limit} {noun}{unit}, but "
            f"{amount} were requested, and no tier raises that limit.",
        )

    # Boolean features.
    granted = {
        "drug_discovery": ent.drug_discovery_access,
        "audit_trail": ent.audit_trail_access,
        "priority_support": ent.priority_support,
    }[feature]
    label = {
        "drug_discovery": "Drug Discovery access",
        "audit_trail": "full audit-trail access",
        "priority_support": "priority support",
    }[feature]
    if granted:
        return True, f"The {tier_name} tier includes {label}."
    upgrade = _cheapest_tier_granting(feature, amount)
    if upgrade is not None:
        return (
            False,
            f"The {tier_name} tier does not include {label}. "
            f"The {upgrade.display_name} tier unlocks it.",
        )
    return False, f"No tier currently includes {label}."


def _demo() -> None:
    print("Provome tiers (Step 6 entitlement model; checkout is Step 8)")
    print("=" * 68)
    for tier in Tier:
        ent = entitlements_for(tier)
        daily = (
            "unlimited"
            if ent.max_live_questions_per_day is None
            else str(ent.max_live_questions_per_day)
        )
        batch = (
            "unlimited"
            if ent.max_batch_variants is None
            else str(ent.max_batch_variants)
        )
        print(
            f"{ent.tier.display_name:<11} ${tier.monthly_price_usd:>3}/mo  "
            f"live questions/day: {daily:<9} batch variants: {batch:<9} "
            f"audit trail: {'yes' if ent.audit_trail_access else 'no':<3} "
            f"seats: {ent.team_seats:<2} "
            f"priority support: {'yes' if ent.priority_support else 'no'}"
        )
    print()
    free = entitlements_for(Tier.FREE)
    allowed, reason = check(free, "live_questions", amount=5)
    print(f"Free, 5 live questions today   -> allowed={allowed}: {reason}")
    allowed, reason = check(free, "audit_trail")
    print(f"Free, open the audit trail      -> allowed={allowed}: {reason}")
    allowed, reason = check(free, "batch_variants", amount=500)
    print(f"Free, batch of 500 variants     -> allowed={allowed}: {reason}")
    lab = entitlements_for(Tier.LAB)
    allowed, reason = check(lab, "batch_variants", amount=500)
    print(f"Lab, batch of 500 variants      -> allowed={allowed}: {reason}")


if __name__ == "__main__":
    _demo()
