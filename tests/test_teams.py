"""Tests for omevra.teams: the team registry and question routing."""

import pytest

from omevra.domains import Division, get_division
from omevra.roles import default_role_registry
from omevra.teams import (
    TEAM_ANIMAL_HEALTH,
    TEAM_CALIBRATION_BENCHMARKS,
    TEAM_CANCER,
    TEAM_CARDIAC,
    TEAM_DNA_VARIANT,
    TEAM_EVIDENCE_VERIFICATION,
    TEAM_INFECTIOUS_DISEASE,
    TEAM_LIVESTOCK,
    TEAM_NEURO,
    TEAM_PLANT_CROPS,
    TEAM_PLANT_PATHOLOGY,
    TEAM_QC,
    TEAM_RARE_DISEASE,
    TEAM_TUMOR_BIOLOGY,
    all_teams,
    default_team_registry,
    get_team,
    route_question,
)

EXPECTED_TEAMS = {
    TEAM_CANCER, TEAM_TUMOR_BIOLOGY, TEAM_RARE_DISEASE, TEAM_CARDIAC,
    TEAM_NEURO, TEAM_INFECTIOUS_DISEASE, TEAM_DNA_VARIANT,
    TEAM_ANIMAL_HEALTH, TEAM_LIVESTOCK, TEAM_PLANT_CROPS,
    TEAM_PLANT_PATHOLOGY, TEAM_EVIDENCE_VERIFICATION,
    TEAM_CALIBRATION_BENCHMARKS, TEAM_QC,
}


def test_all_expected_teams_present():
    assert {t.team_id for t in all_teams()} == EXPECTED_TEAMS


def test_every_team_led_by_a_senior_role():
    roles = default_role_registry()
    for team in all_teams():
        lead = roles.require(team.lead_role_id)
        assert lead.is_senior, f"{team.name} lead {lead.title} is not senior"
        for member_id in team.member_role_ids:
            assert roles.require(member_id).is_senior
        assert team.lead_role_id not in team.member_role_ids


def test_shared_services_are_not_routable():
    registry = default_team_registry()
    shared = registry.shared_services()
    assert {t.team_id for t in shared} == {
        TEAM_EVIDENCE_VERIFICATION, TEAM_CALIBRATION_BENCHMARKS, TEAM_QC,
    }
    for team in shared:
        assert not team.routable
        assert team.is_shared_service


def test_division_teams_own_declared_division_areas():
    # A division team's problem areas must be areas its division
    # registry actually declares: the org chart cannot drift from
    # the division registry.
    for team in all_teams():
        if team.division is None or team.is_shared_service:
            continue
        declared = set(get_division(team.division).problem_areas)
        assert set(team.problem_areas) <= declared, team.name


def test_route_cancer_question_to_cancer_team():
    team = route_question(
        Division.HUMAN,
        text="Does this BRCA1 variant cause hereditary breast cancer?",
    )
    assert team.team_id == TEAM_CANCER


def test_route_tumor_question_to_tumor_biology_team():
    team = route_question(
        Division.HUMAN,
        text="What drives tumor growth and metastasis in this neoplasm?",
    )
    assert team.team_id == TEAM_TUMOR_BIOLOGY


def test_route_cystic_fibrosis_to_rare_disease_team():
    team = route_question(
        Division.HUMAN,
        text="What is CFTR p.Phe508del and does it cause cystic fibrosis?",
    )
    assert team.team_id == TEAM_RARE_DISEASE


def test_route_crop_question_to_plant_crops_team():
    team = route_question(
        Division.PLANT,
        text="How can drought tolerance in maize crops be improved?",
    )
    assert team.team_id == TEAM_PLANT_CROPS


def test_route_blight_question_to_plant_pathology_team():
    team = route_question(
        Division.PLANT,
        text="Late blight resistance in potato against Phytophthora",
    )
    assert team.team_id == TEAM_PLANT_PATHOLOGY


def test_route_cattle_question_to_livestock_team():
    team = route_question(
        Division.ANIMAL,
        text="A cattle herd fertility variant in Bos taurus",
    )
    assert team.team_id == TEAM_LIVESTOCK


def test_route_dog_question_to_animal_health_team():
    team = route_question(
        Division.ANIMAL,
        text="Inherited disease in a companion animal: my dog",
    )
    assert team.team_id == TEAM_ANIMAL_HEALTH


def test_route_by_declared_problem_area():
    registry = default_team_registry()
    team = registry.route(Division.HUMAN, problem_area="cardiac disease")
    assert team.team_id == TEAM_CARDIAC
    team = registry.route(
        Division.HUMAN,
        problem_area="neurological and neurodevelopmental disease",
    )
    assert team.team_id == TEAM_NEURO
    team = registry.route(
        Division.HUMAN, problem_area="infectious disease and host genetics"
    )
    assert team.team_id == TEAM_INFECTIOUS_DISEASE


def test_route_fallback_lands_on_a_named_team():
    # A question with no recognised area still lands on a named
    # senior team's desk; "unassigned" is not a permitted state.
    assert route_question(
        Division.HUMAN, text="Tell me about this finding"
    ).team_id == TEAM_DNA_VARIANT
    assert route_question(
        Division.ANIMAL, text="Tell me about this finding"
    ).team_id == TEAM_ANIMAL_HEALTH
    assert route_question(
        Division.PLANT, text="Tell me about this finding"
    ).team_id == TEAM_PLANT_CROPS


def test_routing_never_crosses_divisions():
    # A plant question full of animal-adjacent words still cannot
    # land on an Animal division team.
    team = route_question(
        Division.PLANT, text="Breeding for yield in this herd of traits"
    )
    assert team.division in (Division.PLANT, None)


def test_unknown_team_raises():
    with pytest.raises(KeyError):
        get_team("no-such-team")
