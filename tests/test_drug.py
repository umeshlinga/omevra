"""Tests for the Drug Discovery provider (Step 5). Fixture transport only."""

from datetime import date

import pytest

from provome.copilot import Copilot, Verdict
from provome.drug import DrugCopilot, DrugEvidenceProvider, parse_drug_query

FIXED = date(2026, 11, 25)


_PUBMED_TITLES = {
    "12345678": "Targeted therapy for CFTR.",
    "23456789": "CFTR potentiators reviewed.",
}


def fixture_transport(url, body=None):
    if "esummary" in url and "db=pubmed" in url:
        pmid = url.rsplit("id=", 1)[1]
        title = _PUBMED_TITLES.get(pmid)
        if title is None:
            return {"result": {"uids": []}}
        return {"result": {"uids": [pmid], pmid: {"title": title}}}
    if "uniprot" in url:
        return {"results": [{"primaryAccession": "P13569"}]}
    if "target/search" in url:
        return {
            "targets": [
                {
                    "target_chembl_id": "CHEMBL4051",
                    "organism": "Homo sapiens",
                    "pref_name": "Cystic fibrosis transmembrane conductance regulator",
                    "target_type": "SINGLE PROTEIN",
                }
            ]
        }
    if "target/CHEMBL4051" in url:
        return {"target_components": [{"accession": "P13569"}]}
    if "activity" in url:
        return {"page_meta": {"total_count": 3728}}
    if "mechanism" in url:
        return {
            "mechanisms": [
                {
                    "molecule_chembl_id": "CHEMBL2103872",
                    "mechanism_of_action": "CFTR positive modulator",
                }
            ]
        }
    if "molecule/CHEMBL2103872" in url:
        return {"pref_name": "IVACAFTOR", "max_phase": 4.0}
    if "europepmc" in url:
        return {
            "resultList": {
                "result": [
                    {"title": "Targeted therapy for CFTR.", "pmid": "12345678", "pubYear": "2026"},
                    {"title": "CFTR potentiators reviewed.", "pmid": "23456789", "pubYear": "2025"},
                ]
            }
        }
    raise AssertionError(url)


def make():
    return DrugEvidenceProvider(transport=fixture_transport, today=lambda: FIXED)


def test_parse_requires_intent_and_gene():
    assert parse_drug_query("What drug discovery evidence exists for targeting CFTR?") == "CFTR"
    assert parse_drug_query("What is known about CFTR p.Phe508del?") is None
    assert parse_drug_query("Tell me about the weather today") is None
    with pytest.raises(TypeError):
        parse_drug_query(None)


def test_items_from_chembl_and_literature():
    items = make().items_for_topic("drug:CFTR")
    sources = {i.source_id for i in items}
    assert sources == {"chembl", "pubmed-literature"}
    assert any("IVACAFTOR" in i.statement for i in items)
    assert all("2026-11-25" in i.citation for i in items)


def test_drug_copilot_answers_and_prints_boundary():
    copilot = Copilot(provider=make())
    answer = copilot.ask("What drug discovery evidence exists for targeting CFTR?")
    assert answer.verdict is Verdict.ANSWER
    text = DrugCopilot(provider=make()).ask(
        "What drug discovery evidence exists for targeting CFTR?"
    )
    assert "not a drug recommendation" in text


def test_unknown_gene_declines():
    def empty(url, body=None):
        if "uniprot" in url:
            return {"results": []}
        if "europepmc" in url:
            return {"resultList": {"result": []}}
        raise AssertionError(url)

    provider = DrugEvidenceProvider(transport=empty, today=lambda: FIXED)
    answer = Copilot(provider=provider).ask(
        "What drug discovery evidence exists for targeting ZZZQX9?"
    )
    assert answer.verdict is Verdict.DECLINE
