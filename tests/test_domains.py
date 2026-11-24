"""Tests for omevra.domains and division handling in the Copilot."""

import pytest

from omevra.copilot import Copilot, CopilotQuestion
from omevra.domains import (
    ENGINE_ACMG,
    ENGINE_CALIBRATION,
    ENGINE_CONSEQUENCE,
    ENGINE_EVIDENCE_GATE,
    ENGINE_PHENOTYPE_SIMILARITY,
    ENGINE_POPULATION,
    Division,
    DivisionStatus,
    all_divisions,
    default_registry,
    division_for_species,
    get_division,
)

SHARED_ENGINES = (
    ENGINE_CONSEQUENCE,
    ENGINE_POPULATION,
    ENGINE_CALIBRATION,
    ENGINE_EVIDENCE_GATE,
    ENGINE_PHENOTYPE_SIMILARITY,
)


def test_exactly_three_divisions_in_order():
    divisions = all_divisions()
    assert [d.division for d in divisions] == [
        Division.HUMAN, Division.ANIMAL, Division.PLANT,
    ]
    assert len(default_registry()) == 3


def test_division_lookup_by_string_and_enum():
    assert get_division("human").division is Division.HUMAN
    assert get_division("Animal").division is Division.ANIMAL
    assert get_division(Division.PLANT).name == "Plant"
    with pytest.raises(KeyError):
        get_division("fungal")


def test_human_division_is_not_cancer_only():
    human = get_division(Division.HUMAN)
    areas = " ".join(human.problem_areas).lower()
    for expected in ("rare disease", "cardiac", "neuro", "metabolic", "infectious"):
        assert expected in areas
    assert "cancer" in areas  # present, but one area among many


def test_human_is_live_and_others_are_expansion():
    assert get_division(Division.HUMAN).status is DivisionStatus.LIVE_BENCHMARKED
    assert get_division(Division.ANIMAL).status is DivisionStatus.EXPANSION
    assert get_division(Division.PLANT).status is DivisionStatus.EXPANSION


def test_every_division_has_scope_species_and_problem_areas():
    for info in all_divisions():
        assert len(info.scope) > 40
        assert len(info.problem_areas) >= 4
        assert len(info.example_species) >= 1
        for species in info.example_species:
            assert species.division is info.division


@pytest.mark.parametrize("scientific,division", [
    ("Homo sapiens", Division.HUMAN),
    ("Canis lupus familiaris", Division.ANIMAL),
    ("Bos taurus", Division.ANIMAL),
    ("Gallus gallus", Division.ANIMAL),
    ("Salmo salar", Division.ANIMAL),
    ("Arabidopsis thaliana", Division.PLANT),
    ("Oryza sativa", Division.PLANT),
    ("Zea mays", Division.PLANT),
    ("Triticum aestivum", Division.PLANT),
])
def test_species_lookup(scientific, division):
    info = division_for_species(scientific)
    assert info is not None
    assert info.division is division


def test_species_lookup_by_common_name_and_case_insensitive():
    assert division_for_species("domestic dog").division is Division.ANIMAL
    assert division_for_species("RICE").division is Division.PLANT
    assert division_for_species("bos taurus").division is Division.ANIMAL


def test_unknown_species_returns_none_not_a_guess():
    assert division_for_species("Pan troglodytes") is None
    assert division_for_species("") is None


def test_shared_engines_apply_to_all_divisions():
    for info in all_divisions():
        for engine in SHARED_ENGINES:
            assert info.engine_applies(engine), (info.name, engine)


def test_acmg_is_human_only():
    assert get_division(Division.HUMAN).engine_applies(ENGINE_ACMG)
    assert not get_division(Division.ANIMAL).engine_applies(ENGINE_ACMG)
    assert not get_division(Division.PLANT).engine_applies(ENGINE_ACMG)
    # And the registry says why, in data a caller can show a user.
    note = get_division(Division.PLANT).applicability(ENGINE_ACMG).note
    assert "human-clinical" in note


def test_undeclared_engine_raises():
    with pytest.raises(KeyError):
        get_division(Division.HUMAN).engine_applies("time_travel")


def test_copilot_question_defaults_to_human():
    question = CopilotQuestion(text="What is CFTR p.Phe508del?")
    assert question.division is Division.HUMAN


def test_copilot_question_species_sets_division():
    animal_q = CopilotQuestion(text="A question", species="Bos taurus")
    assert animal_q.division is Division.ANIMAL
    plant_q = CopilotQuestion(text="A question", species="Oryza sativa")
    assert plant_q.division is Division.PLANT


def test_copilot_question_accepts_division_string():
    question = CopilotQuestion(text="A question", division="plant")
    assert question.division is Division.PLANT


def test_copilot_answer_carries_division_through():
    copilot = Copilot()
    answer = copilot.ask(CopilotQuestion(
        text="What is CFTR p.Phe508del and does it cause cystic fibrosis?",
        species="Bos taurus",
    ))
    assert answer.division is Division.ANIMAL
    data = answer.to_dict()
    assert data["division"] == "animal"
    assert data["division_name"] == "Animal"
    assert any("ACMG" in w or "human-clinical" in w for w in answer.warnings)


def test_copilot_default_answer_is_human_division():
    copilot = Copilot()
    answer = copilot.ask("What is CFTR p.Phe508del and does it cause cystic fibrosis?")
    assert answer.division is Division.HUMAN
    assert answer.to_dict()["division"] == "human"
