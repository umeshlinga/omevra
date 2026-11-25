"""Tests for the Animal division provider (Step 9). Fixture transport only."""

from datetime import date

import pytest

from provome.animal import (
    BOUNDARY,
    AnimalCopilot,
    AnimalEvidenceProvider,
    parse_animal_query,
)
from provome.copilot import Copilot, CopilotQuestion, Verdict

FIXED = date(2026, 11, 25)


def fixture_transport(url, body=None):
    if "lookup/symbol/canis_lupus_familiaris/ABCB1" in url:
        return {
            "id": "ENSCAFG00000000001",
            "display_name": "ABCB1",
            "biotype": "protein_coding",
            "assembly_name": "CanFam3.1",
            "description": "ATP binding cassette subfamily B member 1",
        }
    if "phenotype/gene/canis_lupus_familiaris" in url:
        return [
            {
                "phenotype": {"description": "Ivermectin sensitivity"},
                "source": {"name": "OMIA"},
            },
            {
                "phenotype": {"description": "Drug sensitivity (MDR1)"},
                "source": {"name": "OMIA"},
            },
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
    raise AssertionError(url)


def empty_transport(url, body=None):
    if "ensembl" in url:
        return {}
    if "europepmc" in url:
        return {"resultList": {"result": []}}
    raise AssertionError(url)


def make(transport=fixture_transport):
    return AnimalEvidenceProvider(transport=transport, today=lambda: FIXED)


def test_parse_requires_species_and_gene():
    query = parse_animal_query("What is known about ABCB1 in dogs?")
    assert query is not None
    assert query.species.key == "dog"
    assert query.subject == "ABCB1"
    assert query.is_gene
    assert query.topic_key == "animal:dog:ABCB1"


def test_parse_scientific_name():
    query = parse_animal_query(
        "What is known about ABCB1 in Canis lupus familiaris?"
    )
    assert query is not None
    assert query.species.key == "dog"
    assert query.species.ensembl == "canis_lupus_familiaris"


def test_parse_no_species_no_topic():
    assert parse_animal_query("What is known about ABCB1?") is None
    provider = make()
    assert provider.topic_for_question("What is known about ABCB1?") is None
    answer = Copilot(provider=provider).ask("What is known about ABCB1?")
    assert answer.verdict is Verdict.DECLINE


def test_parse_condition_topic():
    query = parse_animal_query("What is known about hip dysplasia in dogs?")
    assert query is not None
    assert not query.is_gene
    assert query.topic_key == "animal:dog:hip-dysplasia"


def test_parse_type_error():
    with pytest.raises(TypeError):
        parse_animal_query(None)


def test_items_from_ensembl_and_literature():
    items = make().items_for_topic("animal:dog:ABCB1")
    sources = {i.source_id for i in items}
    assert sources == {"ensembl", "pubmed-literature"}
    assert any("ENSCAFG00000000001" in i.statement for i in items)
    assert any("Ivermectin sensitivity" in i.statement for i in items)
    assert all("2026-11-25" in i.citation for i in items)
    assert all(i.is_verified for i in items)


def test_animal_copilot_answers_and_prints_boundary():
    answer = Copilot(provider=make()).ask(
        CopilotQuestion(
            text="What is known about ABCB1 drug sensitivity in dogs?",
            species="Canis lupus familiaris",
        )
    )
    assert answer.verdict is Verdict.ANSWER
    assert answer.division_name == "Animal"
    assert answer.verified_item_count >= 3
    assert answer.independent_source_count >= 2
    text = AnimalCopilot(provider=make()).ask(
        "What is known about ABCB1 drug sensitivity in dogs?"
    )
    assert "Boundary: " + BOUNDARY in text
    assert "not a diagnosis or treatment recommendation" in text
    assert "in build and unbenchmarked" in text


def test_gate_arithmetic_respected_when_evidence_thin():
    def thin(url, body=None):
        if "lookup/symbol" in url:
            return {
                "id": "ENSCAFG00000000001",
                "display_name": "ABCB1",
                "biotype": "protein_coding",
                "assembly_name": "CanFam3.1",
            }
        if "phenotype/gene" in url:
            return []
        if "europepmc" in url:
            return {
                "resultList": {
                    "result": [
                        {"title": "One paper only.", "pmid": "33333333",
                         "pubYear": "2024"}
                    ]
                }
            }
        raise AssertionError(url)

    provider = make(transport=thin)
    items = provider.items_for_topic("animal:dog:ABCB1")
    assert len(items) == 2  # 1 Ensembl + 1 literature: below the gate
    answer = Copilot(provider=provider).ask(
        "What is known about ABCB1 drug sensitivity in dogs?"
    )
    assert answer.verdict is Verdict.DECLINE


def test_ensembl_empty_declines_honestly():
    provider = make(transport=empty_transport)
    assert provider.items_for_topic("animal:cattle:ZZZQX9") == ()
    answer = Copilot(provider=provider).ask(
        "What is known about ZZZQX9 in cattle?"
    )
    assert answer.verdict is Verdict.DECLINE


def test_division_scope_guard_both_directions():
    """Division scoping (CEO integration fix): the Animal provider
    must not answer a Human-division question, and a Human-scoped
    (default) Copilot must not answer an Animal-division question.
    """
    human_q = Copilot(provider=make()).ask(
        "What is known about ABCB1 drug sensitivity in dogs?"
    )
    assert human_q.verdict is Verdict.DECLINE
    animal_q = Copilot().ask(
        CopilotQuestion(
            text="What is known about ABCB1 drug sensitivity in dogs?",
            species="Canis lupus familiaris",
        )
    )
    assert animal_q.verdict is Verdict.DECLINE
