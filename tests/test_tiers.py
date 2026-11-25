"""Tests for the Step 6 tier entitlement model (provome/tiers.py)."""

import dataclasses

import pytest

from provome.tiers import (
    FEATURES,
    Entitlements,
    Tier,
    check,
    entitlements_for,
)


def test_tier_prices_match_founder_pricing():
    assert Tier.FREE.monthly_price_usd == 0
    assert Tier.RESEARCHER.monthly_price_usd == 79
    assert Tier.LAB.monthly_price_usd == 499
    assert Tier.FREE < Tier.RESEARCHER < Tier.LAB


def test_entitlement_values_per_tier():
    free = entitlements_for(Tier.FREE)
    assert free.max_live_questions_per_day == 10
    assert free.max_batch_variants == 25
    assert free.drug_discovery_access is True
    assert free.audit_trail_access is False
    assert free.team_seats == 1
    assert free.priority_support is False

    researcher = entitlements_for(Tier.RESEARCHER)
    assert researcher.max_live_questions_per_day == 500
    assert researcher.max_batch_variants == 1000
    assert researcher.drug_discovery_access is True
    assert researcher.audit_trail_access is True
    assert researcher.team_seats == 1
    assert researcher.priority_support is False

    lab = entitlements_for(Tier.LAB)
    assert lab.max_live_questions_per_day is None
    assert lab.max_batch_variants is None
    assert lab.drug_discovery_access is True
    assert lab.audit_trail_access is True
    assert lab.team_seats == 25
    assert lab.priority_support is True


def test_entitlements_are_frozen():
    ent = entitlements_for(Tier.FREE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ent.team_seats = 99


def test_entitlements_for_rejects_non_tier():
    with pytest.raises(TypeError):
        entitlements_for("Researcher")
    with pytest.raises(TypeError):
        entitlements_for(79)


def test_check_allows_within_limits():
    free = entitlements_for(Tier.FREE)
    allowed, reason = check(free, "live_questions", amount=10)
    assert allowed is True
    assert "Free" in reason
    allowed, _ = check(free, "batch_variants", amount=25)
    assert allowed is True
    allowed, _ = check(free, "drug_discovery")
    assert allowed is True


def test_check_unlimited_on_lab():
    lab = entitlements_for(Tier.LAB)
    allowed, reason = check(lab, "live_questions", amount=1_000_000)
    assert allowed is True
    assert "unlimited" in reason
    allowed, _ = check(lab, "batch_variants", amount=1_000_000)
    assert allowed is True


def test_check_denial_names_tier_limit_and_upgrade():
    free = entitlements_for(Tier.FREE)
    allowed, reason = check(free, "batch_variants", amount=500)
    assert allowed is False
    assert "Free" in reason and "25" in reason and "500" in reason
    # Researcher (1000) is the cheapest tier that fits 500.
    assert "Researcher" in reason

    allowed, reason = check(free, "batch_variants", amount=5000)
    assert allowed is False
    # Only Lab (unlimited) fits 5000.
    assert "Lab" in reason


def test_check_boolean_denials_name_the_unlocking_tier():
    free = entitlements_for(Tier.FREE)
    allowed, reason = check(free, "audit_trail")
    assert allowed is False
    assert "Free" in reason and "Researcher" in reason

    allowed, reason = check(free, "priority_support")
    assert allowed is False
    assert "Lab" in reason

    researcher = entitlements_for(Tier.RESEARCHER)
    allowed, _ = check(researcher, "audit_trail")
    assert allowed is True
    allowed, reason = check(researcher, "priority_support")
    assert allowed is False
    assert "Lab" in reason


def test_check_team_seats():
    researcher = entitlements_for(Tier.RESEARCHER)
    allowed, reason = check(researcher, "team_seats", amount=5)
    assert allowed is False
    assert "Lab" in reason
    lab = entitlements_for(Tier.LAB)
    allowed, _ = check(lab, "team_seats", amount=25)
    assert allowed is True
    allowed, reason = check(lab, "team_seats", amount=26)
    assert allowed is False
    assert "no tier raises" in reason


def test_check_rejects_bad_inputs():
    free = entitlements_for(Tier.FREE)
    with pytest.raises(ValueError):
        check(free, "teleportation")
    with pytest.raises(ValueError):
        check(free, "live_questions", amount=0)
    with pytest.raises(TypeError):
        check("not-entitlements", "live_questions")


def test_every_feature_checkable_on_every_tier():
    for tier in Tier:
        ent = entitlements_for(tier)
        for feature in FEATURES:
            allowed, reason = check(ent, feature)
            assert isinstance(allowed, bool)
            assert tier.display_name in reason
