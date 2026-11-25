"""Tests for the three doors (Item 31 packaging v1).

Fixture transports only - no network.  The doors are thin wiring,
so these tests pin the wiring: the right copilot and provider per
door, the right boundary per door, cross-division declines through
the door layer in both directions, and examples that parse to
their own door's division.
"""

from datetime import date
from pathlib import Path

import pytest

from provome import doors
from provome.animal import BOUNDARY as ANIMAL_BOUNDARY
from provome.animal import AnimalCopilot, AnimalEvidenceProvider, parse_animal_query
from provome.converse import Guide
from provome.copilot import Copilot, CopilotQuestion, OfflineDemoProvider, Verdict
from provome.domains import Division
from provome.live import LiveEvidenceProvider, parse_variant_query
from provome.plant import BOUNDARY as PLANT_BOUNDARY
from provome.plant import PlantCopilot, PlantEvidenceProvider, parse_plant_query

FIXED = date(2026, 11, 25)
ROOT = Path(__file__).resolve().parent.parent


# -- fixture transports ---------------------------------------------------


_PUBMED_TITLES = {
    "11111111": "ABCB1 mutation and drug sensitivity in Collies.",
    "22222222": "MDR1 genotyping in herding breed dogs.",
    "44444444": "FLC and the control of flowering time.",
    "55555555": "Vernalization requires FLC silencing.",
}


def _pubmed(url):
    pmid = url.rsplit("id=", 1)[1]
    title = _PUBMED_TITLES.get(pmid)
    if title is None:
        return {"result": {"uids": []}}
    return {"result": {"uids": [pmid], pmid: {"title": title}}}


def animal_transport(url, body=None):
    if "esummary" in url and "db=pubmed" in url:
        return _pubmed(url)
    if "lookup/symbol/canis_lupus_familiaris/ABCB1" in url:
        return {
            "id": "ENSCAFG00000000001",
            "display_name": "ABCB1",
            "biotype": "protein_coding",
            "assembly_name": "CanFam3.1",
            "description": "ATP binding cassette subfamily B member 1",
        }
    if "phenotype/gene/canis_lupus_familiaris" in url:
        # Real recorded payload shape: top-level description,
        # plain-string source.
        return [
            {
                "source": "OMIA",
                "location": "11:1111111-2222222",
                "description": "Ivermectin sensitivity",
                "attributes": {"external_id": "OMIA001234"},
                "Gene": "ENSCAFG00000000001",
            }
        ]
    if "europepmc" in url:
        return {
            "resultList": {
                "result": [
                    {
                        "title": "ABCB1 mutation and drug sensitivity in Collies.",
                        "pmid": "11111111",
                        "pubYear": "2024",
                    },
                    {
                        "title": "MDR1 genotyping in herding breed dogs.",
                        "pmid": "22222222",
                        "pubYear": "2023",
                    },
                ]
            }
        }
    return {}


def plant_transport(url, body=None):
    if "esummary" in url and "db=pubmed" in url:
        return _pubmed(url)
    if "lookup/symbol/arabidopsis_thaliana/FLC" in url:
        return {
            "id": "AT5G10140",
            "display_name": "FLC",
            "biotype": "protein_coding",
            "assembly_name": "TAIR10",
            "description": "K-box region and MADS-box transcription factor",
        }
    if "phenotype/gene/arabidopsis_thaliana" in url:
        return []
    if "europepmc" in url:
        return {
            "resultList": {
                "result": [
                    {
                        "title": "FLC and the control of flowering time.",
                        "pmid": "44444444",
                        "pubYear": "2024",
                    },
                    {
                        "title": "Vernalization requires FLC silencing.",
                        "pmid": "55555555",
                        "pubYear": "2023",
                    },
                ]
            }
        }
    return {}


def make_human_door():
    return doors.HumanDoor(
        guide=Guide(Copilot(provider=OfflineDemoProvider()))
    )


def make_animal_door():
    provider = AnimalEvidenceProvider(
        transport=animal_transport, today=lambda: FIXED
    )
    return doors.AnimalDoor(copilot=AnimalCopilot(provider=provider))


def make_plant_door():
    provider = PlantEvidenceProvider(
        transport=plant_transport, today=lambda: FIXED
    )
    return doors.PlantDoor(copilot=PlantCopilot(provider=provider))


# -- registration -----------------------------------------------------------


def test_three_doors_registered_in_order():
    built = doors.all_doors()
    assert [d.division for d in built] == [
        Division.HUMAN,
        Division.ANIMAL,
        Division.PLANT,
    ]
    assert [d.product_name for d in built] == [
        "Provome Human",
        "Provome Animal",
        "Provome Plant",
    ]
    assert [d.command for d in built] == [
        "provome-human",
        "provome-animal",
        "provome-plant",
    ]
    for door in built:
        assert len(door.examples) == 3
        assert door.boundary.strip()
    boundaries = [d.boundary for d in built]
    assert len(set(boundaries)) == 3  # each door has its OWN boundary


def test_doors_use_their_own_division_boundary():
    assert doors.animal_door().boundary == ANIMAL_BOUNDARY
    assert doors.plant_door().boundary == PLANT_BOUNDARY
    assert "not for clinical use" in doors.HUMAN_BOUNDARY


def test_get_door_by_name():
    assert doors.get_door("animal").division is Division.ANIMAL
    assert doors.get_door(Division.PLANT).division is Division.PLANT


# -- wiring -------------------------------------------------------------------


def test_human_door_wiring():
    door = make_human_door()
    assert isinstance(door.copilot, Copilot)
    assert isinstance(door.copilot.provider, OfflineDemoProvider)
    # The default (live) door runs the live human provider, whose
    # scope is the Copilot default: Human only.
    live_door = doors.HumanDoor()
    assert isinstance(live_door.copilot.provider, LiveEvidenceProvider)
    scoped = getattr(
        live_door.copilot.provider,
        "scoped_divisions",
        frozenset({Division.HUMAN}),
    )
    assert scoped == frozenset({Division.HUMAN})


def test_animal_door_wiring():
    door = make_animal_door()
    assert isinstance(door.provider, AnimalEvidenceProvider)
    assert door.provider.scoped_divisions == frozenset({Division.ANIMAL})
    assert door.copilot.provider is door.provider


def test_plant_door_wiring():
    door = make_plant_door()
    assert isinstance(door.provider, PlantEvidenceProvider)
    assert door.provider.scoped_divisions == frozenset({Division.PLANT})
    assert door.copilot.provider is door.provider


def test_doors_module_has_no_gate_logic_of_its_own():
    """Thin wiring guard: the gate lives in provome.evidence and
    is applied by provome.copilot; doors.py must not re-implement
    or re-state it."""
    source = (ROOT / "provome" / "doors.py").read_text()
    assert "evaluate_gate" not in source
    assert "MIN_VERIFIED_ITEMS" not in source


# -- examples parse to their own division -------------------------------------


def test_human_examples_parse_as_human():
    for question in doors.HumanDoor.examples:
        assert parse_variant_query(question) is not None, question
        assert CopilotQuestion(text=question).division is Division.HUMAN


def test_animal_examples_parse_as_animal():
    for question in doors.AnimalDoor.examples:
        parsed = parse_animal_query(question)
        assert parsed is not None, question
        assert parsed.topic_key.startswith("animal:")


def test_plant_examples_parse_as_plant():
    for question in doors.PlantDoor.examples:
        parsed = parse_plant_query(question)
        assert parsed is not None, question
        assert parsed.topic_key.startswith("plant:")


# -- framed answers -------------------------------------------------------------


def test_human_door_answers_with_label_and_boundary():
    door = make_human_door()
    question = "What is known about CFTR p.Phe508del?"
    assert door.guide.ask(question).kind == "answer"
    text = door.ask(question)
    assert "Provome Human — division: Human" in text
    assert "Boundary: " + doors.HUMAN_BOUNDARY in text


def test_animal_door_answers_with_label_and_boundary():
    door = make_animal_door()
    text = door.ask("What is known about ABCB1 in dogs?")
    assert "Provome Animal — division: Animal" in text
    assert "[ANSWER] division: Animal" in text
    assert "Boundary: " + ANIMAL_BOUNDARY in text


def test_plant_door_answers_with_label_and_boundary():
    door = make_plant_door()
    text = door.ask("What is known about FLC in Arabidopsis?")
    assert "Provome Plant — division: Plant" in text
    assert "[ANSWER] division: Plant" in text
    assert "Boundary: " + PLANT_BOUNDARY in text


# -- cross-division declines through the door layer ------------------------------


def test_animal_door_declines_human_and_plant_questions():
    door = make_animal_door()
    assert "[DECLINE]" in door.ask("What is known about CFTR p.Phe508del?")
    assert "[DECLINE]" in door.ask("What is known about FLC in Arabidopsis?")


def test_plant_door_declines_human_and_animal_questions():
    door = make_plant_door()
    assert "[DECLINE]" in door.ask("What is known about CFTR p.Phe508del?")
    assert "[DECLINE]" in door.ask("What is known about ABCB1 in dogs?")


def test_human_door_declines_animal_and_plant_questions():
    door = make_human_door()
    for question in (
        "What is known about ABCB1 in dogs?",
        "What is known about FLC in Arabidopsis?",
    ):
        assert door.guide.ask(question).kind == "decline", question
        text = door.ask(question)
        assert "I can't answer that yet" in text
        assert "Boundary: " + doors.HUMAN_BOUNDARY in text


def test_scope_guard_both_directions_through_door_copilots():
    """Species-carrying questions put to a door's own copilot:
    the provider scope guard declines out-of-division asks."""
    human = make_human_door()
    answer = human.copilot.ask(
        CopilotQuestion(
            text="What is known about CFTR p.Phe508del?",
            species="Bos taurus",  # Animal division context
        )
    )
    assert answer.verdict is Verdict.DECLINE
    assert answer.division is Division.ANIMAL

    animal = make_animal_door()
    answer = animal.copilot.ask(
        CopilotQuestion(
            text="What is known about ABCB1 in dogs?",
            species="Homo sapiens",  # Human division context
        )
    )
    assert answer.verdict is Verdict.DECLINE
    assert answer.division is Division.HUMAN

    plant = make_plant_door()
    answer = plant.copilot.ask(
        CopilotQuestion(
            text="What is known about FLC in Arabidopsis?",
            species="Canis lupus familiaris",  # Animal context
        )
    )
    assert answer.verdict is Verdict.DECLINE
    assert answer.division is Division.ANIMAL


# -- CLI layer -------------------------------------------------------------------


def test_cli_question(capsys):
    door = make_plant_door()
    assert door.main(["What is known about FLC in Arabidopsis?"]) == 0
    out = capsys.readouterr().out
    assert "[ANSWER] division: Plant" in out
    assert "Boundary: " + PLANT_BOUNDARY in out


def test_cli_question_words_are_joined(capsys):
    door = make_human_door()
    assert (
        door.main(["What", "is", "known", "about", "CFTR", "p.Phe508del?"])
        == 0
    )
    out = capsys.readouterr().out
    assert "Provome Human — division: Human" in out


def test_cli_demo_runs_all_examples(capsys):
    door = make_animal_door()
    assert door.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert out.count("Provome Animal — division: Animal") == 3
    assert out.count("Boundary: " + ANIMAL_BOUNDARY) == 3


def test_cli_no_question_is_a_usage_error(capsys):
    door = make_animal_door()
    with pytest.raises(SystemExit) as excinfo:
        door.main([])
    assert excinfo.value.code == 2


# -- packaging -------------------------------------------------------------------


def test_console_scripts_and_extras_declared():
    pyproject = (ROOT / "pyproject.toml").read_text()
    for script, entry in (
        ("provome-human", "provome.doors:human_main"),
        ("provome-animal", "provome.doors:animal_main"),
        ("provome-plant", "provome.doors:plant_main"),
    ):
        assert f'{script} = "{entry}"' in pyproject
        func_name = entry.split(":")[1]
        assert callable(getattr(doors, func_name))
    for extra in ("human = []", "animal = []", "plant = []"):
        assert extra in pyproject
    # The base install is untouched: the original script remains.
    assert 'provome-copilot = "provome.copilot:main"' in pyproject
