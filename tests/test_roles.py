"""Tests for provome.roles: the senior role registry."""

import pytest

from provome.roles import (
    ROLE_ANIMAL_DIVISION_LEAD,
    ROLE_CHIEF_COPILOT,
    ROLE_HUMAN_DIVISION_LEAD,
    ROLE_PLANT_DIVISION_LEAD,
    ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
    ROLE_SENIOR_DATA_ANALYST,
    ROLE_SENIOR_EVIDENCE_REVIEWER,
    ROLE_SENIOR_PRODUCTION_ENGINEER,
    ROLE_SENIOR_PRODUCTION_MANAGER,
    ROLE_SENIOR_QC_LEAD,
    ROLE_SENIOR_RESEARCHER,
    ROLE_SENIOR_SCIENTIST,
    Role,
    RoleRegistry,
    Seniority,
    all_roles,
    default_role_registry,
    get_role,
)

REQUIRED_ROLES = {
    ROLE_CHIEF_COPILOT: "Chief Copilot",
    ROLE_SENIOR_PRODUCTION_MANAGER: "Senior Production Manager",
    ROLE_SENIOR_PRODUCTION_ENGINEER: "Senior Production Engineer",
    ROLE_SENIOR_SCIENTIST: "Senior Scientist",
    ROLE_SENIOR_RESEARCHER: "Senior Researcher",
    ROLE_SENIOR_DATA_ANALYST: "Senior Data Analyst",
    ROLE_SENIOR_BIOINFORMATICS_SCIENTIST: "Senior Bioinformatics Scientist",
    ROLE_SENIOR_EVIDENCE_REVIEWER: "Senior Evidence Reviewer",
    ROLE_SENIOR_QC_LEAD: "Senior QC Lead",
    ROLE_HUMAN_DIVISION_LEAD: "Human Division Lead",
    ROLE_ANIMAL_DIVISION_LEAD: "Animal Division Lead",
    ROLE_PLANT_DIVISION_LEAD: "Plant Division Lead",
}


def test_all_required_senior_roles_present_with_titles():
    registry = default_role_registry()
    for role_id, title in REQUIRED_ROLES.items():
        role = registry.require(role_id)
        assert role.title == title


def test_every_role_is_senior_level():
    # The organisation defines senior roles only: every role in the
    # registry must report senior, whatever its rung on the ladder.
    for role in all_roles():
        assert role.is_senior
        assert role.seniority.is_senior


def test_every_role_has_responsibilities_and_scope():
    for role in all_roles():
        assert role.responsibilities
        assert role.team_scope.strip()


def test_exactly_one_chief():
    registry = default_role_registry()
    chief = registry.chief()
    assert chief.role_id == ROLE_CHIEF_COPILOT
    assert chief.seniority is Seniority.CHIEF
    assert len(registry.by_seniority(Seniority.CHIEF)) == 1


def test_seniority_ladder_ordering():
    assert Seniority.CHIEF.outranks(Seniority.PRINCIPAL)
    assert Seniority.PRINCIPAL.outranks(Seniority.LEAD)
    assert Seniority.LEAD.outranks(Seniority.SENIOR)
    # Even the lowest rung is a senior rung.
    assert Seniority.SENIOR.is_senior


def test_division_leads_carry_their_division():
    registry = default_role_registry()
    assert registry.require(ROLE_HUMAN_DIVISION_LEAD).division == "human"
    assert registry.require(ROLE_ANIMAL_DIVISION_LEAD).division == "animal"
    assert registry.require(ROLE_PLANT_DIVISION_LEAD).division == "plant"
    leads = {r.role_id for r in registry.leads()}
    assert ROLE_HUMAN_DIVISION_LEAD in leads
    assert ROLE_SENIOR_QC_LEAD in leads
    # A plain senior individual contributor has no sign-off authority.
    assert ROLE_SENIOR_RESEARCHER not in leads


def test_registry_rejects_duplicate_role():
    role = get_role(ROLE_SENIOR_SCIENTIST)
    with pytest.raises(ValueError):
        RoleRegistry([role, role])


def test_role_requires_responsibilities():
    with pytest.raises(ValueError):
        Role(
            role_id="empty-role",
            title="Empty Role",
            seniority=Seniority.SENIOR,
            responsibilities=(),
            team_scope="nowhere",
        )


def test_unknown_role_raises():
    with pytest.raises(KeyError):
        default_role_registry().require("no-such-role")
