"""Tests for the live evidence provider (Step 3).

All tests run on recorded fixture payloads through an injected
transport; the suite never touches the network.
"""

from datetime import date

import pytest

from omevra.copilot import Copilot, Verdict
from omevra.live import (
    LiveEvidenceProvider,
    acceptable_notations,
    one_letter_notation,
    parse_variant_query,
)

FIXED_DATE = date(2026, 11, 25)


def fixture_transport(url, body=None):
    if "esearch" in url:
        return {"esearchresult": {"idlist": ["15333"]}}
    if "esummary" in url:
        return {
            "result": {
                "uids": ["15333"],
                "15333": {
                    "title": "NM_000518.5(HBB):c.20A>T (p.Glu7Val)",
                    "protein_change": "E7V",
                    "germline_classification": {
                        "description": "Pathogenic",
                        "review_status": "criteria provided, multiple submitters, no conflicts",
                    },
                    "variation_set": [
                        {
                            "variation_loc": [
                                {
                                    "assembly_name": "GRCh38",
                                    "chr": "11",
                                    "start": "5227002",
                                    "ref": "A",
                                    "alt": "T",
                                }
                            ]
                        }
                    ],
                },
            }
        }
    if "gnomad" in url:
        return {
            "data": {
                "variant": {
                    "variant_id": "11-5227002-A-T",
                    "genome": {"ac": 12000, "an": 150000, "af": 0.08},
                    "exome": None,
                }
            }
        }
    if "europepmc" in url:
        return {
            "resultList": {
                "result": [
                    {
                        "title": "Sickle cell disease review.",
                        "pmid": "11111111",
                        "pubYear": "2024",
                    },
                    {
                        "title": "HBB variant cohort study.",
                        "pmid": "22222222",
                        "pubYear": "2025",
                    },
                ]
            }
        }
    raise AssertionError(f"unexpected URL: {url}")


def make_provider():
    return LiveEvidenceProvider(
        transport=fixture_transport, today=lambda: FIXED_DATE
    )


def test_parse_requires_gene_and_variant():
    assert parse_variant_query("What is known about HBB p.Glu6Val?") is not None
    assert parse_variant_query("Tell me about the weather in Indianapolis") is None
    assert parse_variant_query("CFTR is a gene I like a lot") is None
    assert parse_variant_query("p.Phe508del is interesting to me") is None
    with pytest.raises(TypeError):
        parse_variant_query(None)


def test_notation_helpers():
    assert one_letter_notation("Phe508del") == "F508del"
    forms = acceptable_notations("Glu6Val")
    assert {"glu6val", "e6v", "glu7val", "e7v"} <= forms


def test_live_items_all_sources_verified_with_fetch_date():
    provider = make_provider()
    items = provider.items_for_topic("HBB:p.Glu6Val")
    sources = {item.source_id for item in items}
    assert sources == {"clinvar", "gnomad", "pubmed-literature"}
    for item in items:
        assert item.status.value == "verified"
        assert "2026-11-25" in item.citation


def test_live_copilot_answers_from_fixtures():
    copilot = Copilot(provider=make_provider())
    answer = copilot.ask("What is known about HBB p.Glu6Val?")
    assert answer.verdict is Verdict.ANSWER


def test_live_copilot_declines_non_variant_questions():
    copilot = Copilot(provider=make_provider())
    answer = copilot.ask("What is the capital of France?")
    assert answer.verdict is Verdict.DECLINE


def test_empty_sources_contribute_nothing():
    def empty_transport(url, body=None):
        if "esearch" in url:
            return {"esearchresult": {"idlist": []}}
        if "europepmc" in url:
            return {"resultList": {"result": []}}
        raise AssertionError(url)

    provider = LiveEvidenceProvider(
        transport=empty_transport, today=lambda: FIXED_DATE
    )
    assert provider.items_for_topic("XYZ1:p.Ala1Val") == ()
