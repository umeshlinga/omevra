"""Tests for the Plant division provider (Step 9). Fixture transport only."""

from datetime import date

import pytest

from provome.copilot import Copilot, CopilotQuestion, Verdict
from provome.plant import (
    BOUNDARY,
    PlantCopilot,
    PlantEvidenceProvider,
    parse_plant_query,
)

FIXED = date(2026, 11, 25)


_PUBMED_TITLES = {
    "44444444": "FLC and the control of flowering time.",
    "55555555": "Vernalization requires FLC silencing.",
}


def fixture_transport(url, body=None):
    if "esummary" in url and "db=pubmed" in url:
        pmid = url.rsplit("id=", 1)[1]
        title = _PUBMED_TITLES.get(pmid)
        if title is None:
            return {"result": {"uids": []}}
        return {"result": {"uids": [pmid], pmid: {"title": title}}}
    if "lookup/symbol/arabidopsis_thaliana/FLC" in url:
        return {
            "id": "AT5G10140",
            "display_name": "FLC",
            "biotype": "protein_coding",
            "assembly_name": "TAIR10",
            "description": (
                "K-box region and MADS-box transcription factor "
                "family protein"
            ),
        }
    if "phenotype/gene/arabidopsis_thaliana" in url:
        # Realistic for plants: Ensembl holds no trait associations.
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
    raise AssertionError(url)


def empty_transport(url, body=None):
    if "ensembl" in url:
        return {}
    if "europepmc" in url:
        return {"resultList": {"result": []}}
    raise AssertionError(url)


def make(transport=fixture_transport):
    return PlantEvidenceProvider(transport=transport, today=lambda: FIXED)


def test_parse_requires_species_and_gene():
    query = parse_plant_query("What is known about FLC in Arabidopsis?")
    assert query is not None
    assert query.species.key == "arabidopsis"
    assert query.subject == "FLC"
    assert query.is_gene
    assert query.topic_key == "plant:arabidopsis:FLC"


def test_parse_scientific_name():
    query = parse_plant_query("What is known about FLC in Arabidopsis thaliana?")
    assert query is not None
    assert query.species.key == "arabidopsis"
    assert query.species.ensembl == "arabidopsis_thaliana"


def test_parse_common_aliases():
    query = parse_plant_query("What is known about SD1 in rice?")
    assert query is not None
    assert query.species.key == "rice"
    assert query.species.ensembl == "oryza_sativa"
    query = parse_plant_query("What is known about TB1 in corn?")
    assert query is not None
    assert query.species.key == "maize"


def test_parse_no_species_no_topic():
    assert parse_plant_query("What is known about FLC?") is None
    provider = make()
    assert provider.topic_for_question("What is known about FLC?") is None
    answer = Copilot(provider=provider).ask("What is known about FLC?")
    assert answer.verdict is Verdict.DECLINE


def test_parse_trait_topic():
    query = parse_plant_query("What is known about drought tolerance in wheat?")
    assert query is not None
    assert not query.is_gene
    assert query.topic_key == "plant:wheat:drought-tolerance"


def test_parse_type_error():
    with pytest.raises(TypeError):
        parse_plant_query(None)


def test_items_from_ensembl_plants_and_literature():
    items = make().items_for_topic("plant:arabidopsis:FLC")
    sources = {i.source_id for i in items}
    assert sources == {"ensembl-plants", "pubmed-literature"}
    assert any("AT5G10140" in i.statement for i in items)
    assert all("2026-11-25" in i.citation for i in items)
    assert all(i.is_verified for i in items)


def test_plant_copilot_answers_and_prints_boundary():
    answer = Copilot(provider=make()).ask(
        CopilotQuestion(
            text="What is known about FLC in Arabidopsis?",
            species="Arabidopsis thaliana",
        )
    )
    assert answer.verdict is Verdict.ANSWER
    assert answer.division_name == "Plant"
    assert answer.verified_item_count >= 3
    assert answer.independent_source_count >= 2
    text = PlantCopilot(provider=make()).ask(
        "What is known about FLC in Arabidopsis?"
    )
    assert "division: Plant" in text
    assert "Boundary: " + BOUNDARY in text
    assert "not agronomic or commercial growing advice" in text
    assert "in build and unbenchmarked" in text


def test_gate_arithmetic_respected_when_evidence_thin():
    def thin(url, body=None):
        if "lookup/symbol" in url:
            return {
                "id": "AT5G10140",
                "display_name": "FLC",
                "biotype": "protein_coding",
                "assembly_name": "TAIR10",
            }
        if "phenotype/gene" in url:
            return []
        if "europepmc" in url:
            return {
                "resultList": {
                    "result": [
                        {"title": "One paper only.", "pmid": "66666666",
                         "pubYear": "2024"}
                    ]
                }
            }
        raise AssertionError(url)

    provider = make(transport=thin)
    items = provider.items_for_topic("plant:arabidopsis:FLC")
    assert len(items) == 2  # 1 Ensembl Plants + 1 literature: below the gate
    answer = Copilot(provider=provider).ask(
        CopilotQuestion(
            text="What is known about FLC in Arabidopsis?",
            species="Arabidopsis thaliana",
        )
    )
    assert answer.verdict is Verdict.DECLINE


def test_ensembl_empty_declines_honestly():
    provider = make(transport=empty_transport)
    assert provider.items_for_topic("plant:rice:ZZZQX9") == ()
    answer = Copilot(provider=provider).ask(
        CopilotQuestion(
            text="What is known about ZZZQX9 in rice?",
            species="Oryza sativa",
        )
    )
    assert answer.verdict is Verdict.DECLINE


def test_division_scope_guard_both_directions():
    """The Plant provider must not answer a Human-division question,
    and a Human-scoped (default) Copilot must not answer a
    Plant-division question."""
    human_q = Copilot(provider=make()).ask("What is known about FLC in Arabidopsis?")
    assert human_q.verdict is Verdict.DECLINE
    plant_q = Copilot().ask(
        CopilotQuestion(
            text="What is known about FLC in Arabidopsis?",
            species="Arabidopsis thaliana",
        )
    )
    assert plant_q.verdict is Verdict.DECLINE


def test_parse_mixed_case_gene_symbols():
    """CEO regression (2026-11-25): real plant symbols that are
    not all-uppercase must be extractable - the Plant benchmark
    construction found Rht-B1 and GmFT2a unaskable."""
    q = parse_plant_query("What is known about Rht-B1 in wheat?")
    assert q is not None and q.subject == "Rht-B1" and q.is_gene
    q = parse_plant_query("What is known about GmFT2a in soybean?")
    assert q is not None and q.subject == "GmFT2a" and q.is_gene
    q = parse_plant_query("What is known about tb1 in maize?")
    assert q is not None and q.subject == "tb1" and q.is_gene


def test_parse_plain_words_are_not_genes():
    """The widened pattern must not turn English into genes."""
    q = parse_plant_query("What is known about flowering in Arabidopsis?")
    assert q is None or not q.is_gene
    q = parse_plant_query("Tell me about rice in general terms please")
    assert q is None or not q.is_gene
