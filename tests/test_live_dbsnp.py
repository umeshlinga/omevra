"""Tests for Step 3b: dbSNP allele resolution in the live provider.

Real ClinVar esummary locations carry chromosome and position but
empty ref/alt strings, so the gnomAD item needs the alleles resolved
by cross-reference: ClinVar dbSNP xref -> NCBI Variation API refsnp
SPDI -> (for indels) left-normalisation against GRCh38 reference
sequence.  All tests run on recorded fixture payloads through an
injected transport; the suite never touches the network.  The
Ensembl fixture slices a real GRCh38 chr7 segment fetched from
Ensembl REST on 2026-11-25 (region 7:117559500..117559700).
"""

import re
from datetime import date

import pytest

from provome.live import (
    LiveEvidenceProvider,
    dbsnp_rsid,
    normalize_vcf,
    parse_spdi,
    refsnp_grch38_alleles,
    select_variant_allele,
)

FIXED_DATE = date(2026, 11, 25)

# Real GRCh38 chr7 sequence, 117559500..117559700 (Ensembl REST).
CHR7_SEGMENT_START = 117559500
CHR7_SEGMENT = (
    "CCTTCAGAGGGTAAAATTAAGCACAGTGGAAGAATTTCATTCTGTTCTCAGTTTTC"
    "CTGGATTATGCCTGGCACCATTAAAGAAAATATCATCTTTGGTGTTTCCTATGAT"
    "GAATATAGATACAGAAGCGTCATCAAAGCATGCCAACTAGAAGAGGTAAGAAACT"
    "ATGTGAAAACTTTTTGATTATGCATATGAACCCTT"
)


def ensembl_fixture(url):
    match = re.search(r"/7:(\d+)\.\.(\d+):1", url)
    assert match, f"unexpected Ensembl URL: {url}"
    start, end = int(match.group(1)), int(match.group(2))
    seq = CHR7_SEGMENT[start - CHR7_SEGMENT_START : end - CHR7_SEGMENT_START + 1]
    assert len(seq) == end - start + 1
    return {"seq": seq, "id": f"chromosome:GRCh38:7:{start}:{end}:1"}


def literature_fixture():
    return {
        "resultList": {
            "result": [
                {"title": "A relevant paper.", "pmid": "33333333", "pubYear": "2024"}
            ]
        }
    }




def pubmed_fixture(url):
    """Honest PubMed for the fixture world: the one literature
    PMID these fixtures serve (33333333) resolves with its
    recorded title; anything else does not exist."""
    pmid = url.rsplit("id=", 1)[1]
    if pmid == "33333333":
        return {"result": {"uids": [pmid], pmid: {"title": "A relevant paper."}}}
    return {"result": {"uids": []}}
# -- unit tests: parsing and selection --------------------------------------


def test_parse_spdi():
    assert parse_spdi("NC_000007.14:117559590:TCTT:T") == (
        "NC_000007.14",
        117559590,
        "TCTT",
        "T",
    )
    # Pure deletion keeps its empty inserted sequence.
    assert parse_spdi("NC_000007.14:117559592:TTT:") == (
        "NC_000007.14",
        117559592,
        "TTT",
        "",
    )
    assert parse_spdi("NM_000518.5:69:A:T") is None  # transcript, not genomic
    assert parse_spdi("NC_000011.10:notanint:T:A") is None
    assert parse_spdi("") is None
    assert parse_spdi(None) is None


def test_dbsnp_rsid():
    measure = {
        "variation_xrefs": [
            {"db_source": "OMIM", "db_id": "141900.0039"},
            {"db_source": "dbSNP", "db_id": "334"},
        ]
    }
    assert dbsnp_rsid(measure) == 334
    assert dbsnp_rsid({"variation_xrefs": []}) is None
    assert dbsnp_rsid({}) is None
    assert dbsnp_rsid({"variation_xrefs": [{"db_source": "dbSNP", "db_id": "rs334"}]}) is None


def _allele(seq, pos, deleted, inserted):
    return {
        "allele": {
            "spdi": {
                "seq_id": seq,
                "position": pos,
                "deleted_sequence": deleted,
                "inserted_sequence": inserted,
            }
        }
    }


def test_refsnp_grch38_alleles_prefers_ptlp_grch38():
    payload = {
        "primary_snapshot_data": {
            "placements_with_allele": [
                {
                    "seq_id": "NC_000011.9",
                    "is_ptlp": False,
                    "placement_annot": {
                        "seq_id_traits_by_assembly": [
                            {"assembly_name": "GRCh37.p13"}
                        ]
                    },
                    "alleles": [_allele("NC_000011.9", 5248231, "T", "A")],
                },
                {
                    "seq_id": "NM_000518.5",
                    "is_ptlp": False,
                    "placement_annot": {"seq_id_traits_by_assembly": []},
                    "alleles": [_allele("NM_000518.5", 69, "A", "T")],
                },
                {
                    "seq_id": "NC_000011.10",
                    "is_ptlp": True,
                    "placement_annot": {
                        "seq_id_traits_by_assembly": [
                            {"assembly_name": "GRCh38.p14"}
                        ]
                    },
                    "alleles": [
                        _allele("NC_000011.10", 5227001, "T", "T"),
                        _allele("NC_000011.10", 5227001, "T", "A"),
                    ],
                },
            ]
        }
    }
    assert refsnp_grch38_alleles(payload) == [
        ("NC_000011.10", 5227001, "T", "T"),
        ("NC_000011.10", 5227001, "T", "A"),
    ]
    assert refsnp_grch38_alleles({}) == []
    assert refsnp_grch38_alleles({"primary_snapshot_data": {}}) == []


def test_select_variant_allele():
    ref = ("NC_000011.10", 5227001, "T", "T")
    alt_a = ("NC_000011.10", 5227001, "T", "A")
    alt_c = ("NC_000011.10", 5227001, "T", "C")
    alt_g = ("NC_000011.10", 5227001, "T", "G")
    # Single candidate: selected outright.
    assert select_variant_allele([ref, alt_a], set()) == alt_a
    # Multi-allelic: only the canonical-SPDI match is the variant.
    assert select_variant_allele([ref, alt_a, alt_c, alt_g], {alt_a}) == alt_a
    # Multi-allelic with no (or several) matches: ambiguous -> None.
    assert select_variant_allele([ref, alt_a, alt_c], set()) is None
    assert select_variant_allele([ref, alt_a, alt_c], {alt_a, alt_c}) is None
    # require_match (fallback rsIDs): even a single candidate must match.
    assert select_variant_allele([ref, alt_a], set(), require_match=True) is None
    assert select_variant_allele([ref, alt_a], {alt_a}, require_match=True) == alt_a
    # Reference allele only: nothing to select.
    assert select_variant_allele([ref], {ref}) is None


def test_normalize_vcf_left_aligns_cftr_deletion():
    def base_at(pos1):
        index = pos1 - CHR7_SEGMENT_START
        return CHR7_SEGMENT[index] if 0 <= index < len(CHR7_SEGMENT) else None

    # dbSNP's placed form for rs113993960 -> gnomAD's indexed form.
    assert normalize_vcf(117559591, "TCTT", "T", base_at) == (
        117559590,
        "ATCT",
        "A",
    )
    # An SNV is already normalised.
    assert normalize_vcf(117559600, "T", "C", base_at) == (117559600, "T", "C")
    # Running off the fetched window is an honest failure, not a guess.
    assert normalize_vcf(117559500, "CC", "C", base_at) is None


# -- provider fixtures: HBB SNV through a multi-allelic rsID -----------------

HBB_REFSNP_334 = {
    "primary_snapshot_data": {
        "placements_with_allele": [
            {
                "seq_id": "NC_000011.10",
                "is_ptlp": True,
                "placement_annot": {
                    "seq_id_traits_by_assembly": [
                        {"assembly_name": "GRCh38.p14"}
                    ]
                },
                "alleles": [
                    _allele("NC_000011.10", 5227001, "T", "T"),
                    _allele("NC_000011.10", 5227001, "T", "A"),
                    _allele("NC_000011.10", 5227001, "T", "C"),
                    _allele("NC_000011.10", 5227001, "T", "G"),
                ],
            }
        ]
    }
}


def hbb_transport(url, body=None):
    if "esearch" in url and "db=clinvar" in url:
        return {"esearchresult": {"idlist": ["15333"]}}
    if "esummary" in url and "db=pubmed" in url:
        return pubmed_fixture(url)
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
                            "variation_xrefs": [
                                {"db_source": "dbSNP", "db_id": "334"}
                            ],
                            "canonical_spdi": "NC_000011.10:5227001:T:A",
                            "variation_loc": [
                                {
                                    "assembly_name": "GRCh38",
                                    "chr": "11",
                                    "start": "5227002",
                                    "ref": "",
                                    "alt": "",
                                }
                            ],
                        }
                    ],
                },
            }
        }
    if "variation/v0/refsnp/334" in url:
        return HBB_REFSNP_334
    if "rest.ensembl.org" in url:
        raise AssertionError("SNV resolution must not fetch reference sequence")
    if "gnomad" in url:
        assert body["variables"]["v"] == "11-5227002-T-A", body
        return {
            "data": {
                "variant": {
                    "variant_id": "11-5227002-T-A",
                    "genome": {"ac": 1937, "an": 152294, "af": 0.0127},
                    "exome": None,
                }
            }
        }
    if "europepmc" in url:
        return literature_fixture()
    raise AssertionError(f"unexpected URL: {url}")


def test_hbb_gnomad_fires_via_dbsnp_resolution():
    provider = LiveEvidenceProvider(
        transport=hbb_transport, today=lambda: FIXED_DATE
    )
    items = provider.items_for_topic("HBB:p.Glu6Val")
    sources = {item.source_id for item in items}
    assert sources == {"clinvar", "gnomad", "pubmed-literature"}
    gnomad = next(i for i in items if i.source_id == "gnomad")
    assert "11-5227002-T-A" in gnomad.statement
    assert gnomad.status.value == "verified"
    assert "2026-11-25" in gnomad.citation


# -- provider fixtures: CFTR deletion through normalisation ------------------

CFTR_REFSNP_113993960 = {
    "primary_snapshot_data": {
        "placements_with_allele": [
            {
                "seq_id": "NC_000007.14",
                "is_ptlp": True,
                "placement_annot": {
                    "seq_id_traits_by_assembly": [
                        {"assembly_name": "GRCh38.p14"}
                    ]
                },
                "alleles": [
                    _allele("NC_000007.14", 117559590, "TCTT", "TCTT"),
                    _allele("NC_000007.14", 117559590, "TCTT", "T"),
                    _allele("NC_000007.14", 117559590, "TCTT", "TCTTCTT"),
                ],
            }
        ]
    }
}


def _cftr_record(uid, title, protein_change, xrefs, spdi):
    return {
        "title": title,
        "protein_change": protein_change,
        "germline_classification": {
            "description": "Pathogenic",
            "review_status": "criteria provided, multiple submitters, no conflicts",
        },
        "variation_set": [
            {
                "variation_xrefs": xrefs,
                "canonical_spdi": spdi,
                "variation_loc": [
                    {
                        "assembly_name": "GRCh38",
                        "chr": "7",
                        "start": "117559593",
                        "ref": "",
                        "alt": "",
                    }
                ],
            }
        ],
    }


def cftr_transport(url, body=None):
    if "esearch" in url and "db=clinvar" in url:
        return {"esearchresult": {"idlist": ["4072070", "7105"]}}
    if "esummary" in url and "db=pubmed" in url:
        return pubmed_fixture(url)
    if "esummary" in url:
        return {
            "result": {
                "uids": ["4072070", "7105"],
                # The headline record: no dbSNP xref, and a canonical
                # SPDI in the unanchored 3' form that matches no
                # refsnp allele exactly.
                "4072070": _cftr_record(
                    "4072070",
                    "NM_000492.4(CFTR):c.1522_1524del (p.Phe508del)",
                    "F508del",
                    [],
                    "NC_000007.14:117559592:TTT:",
                ),
                # The older record: carries the rsID and the anchored
                # canonical SPDI that identifies the deletion allele
                # among the refsnp candidates (deletion vs duplication).
                "7105": _cftr_record(
                    "7105",
                    "NM_000492.3(CFTR):c.1521_1523del (p.Phe508del)",
                    "",
                    [{"db_source": "dbSNP", "db_id": "113993960"}],
                    "NC_000007.14:117559590:TCTT:T",
                ),
            }
        }
    if "variation/v0/refsnp/113993960" in url:
        return CFTR_REFSNP_113993960
    if "rest.ensembl.org" in url:
        return ensembl_fixture(url)
    if "gnomad" in url:
        assert body["variables"]["v"] == "7-117559590-ATCT-A", body
        return {
            "data": {
                "variant": {
                    "variant_id": "7-117559590-ATCT-A",
                    "genome": {"ac": 1200, "an": 152206, "af": 0.00788},
                    "exome": None,
                }
            }
        }
    if "europepmc" in url:
        return literature_fixture()
    raise AssertionError(f"unexpected URL: {url}")


def test_cftr_gnomad_fires_after_left_normalisation():
    provider = LiveEvidenceProvider(
        transport=cftr_transport, today=lambda: FIXED_DATE
    )
    items = provider.items_for_topic("CFTR:p.Phe508del")
    gnomad = [i for i in items if i.source_id == "gnomad"]
    assert len(gnomad) == 1
    assert "7-117559590-ATCT-A" in gnomad[0].statement
    assert gnomad[0].status.value == "verified"


def test_both_notation_searches_feed_resolution():
    """Regression: the two ClinVar notation searches return
    different record sets.  The record carrying the dbSNP xref may
    appear only in the second search, so resolution must see the
    union — stopping at the first non-empty search starves it.
    """
    records = {
        "4072070": _cftr_record(
            "4072070",
            "NM_000492.4(CFTR):c.1522_1524del (p.Phe508del)",
            "F508del",
            [],
            "NC_000007.14:117559592:TTT:",
        ),
        "7105": _cftr_record(
            "7105",
            "NM_000492.3(CFTR):c.1521_1523del (p.Phe508del)",
            "",
            [{"db_source": "dbSNP", "db_id": "113993960"}],
            "NC_000007.14:117559590:TCTT:T",
        ),
    }

    def transport(url, body=None):
        if "esearch" in url and "db=clinvar" in url:
            if "Phe508del" in url:
                return {"esearchresult": {"idlist": ["4072070"]}}
            return {"esearchresult": {"idlist": ["7105"]}}
        if "esummary" in url:
            ids = re.search(r"[?&]id=([0-9,]+)", url).group(1).split(",")
            return {
                "result": {
                    "uids": ids,
                    **{uid: records[uid] for uid in ids},
                }
            }
        return cftr_transport(url, body)

    provider = LiveEvidenceProvider(
        transport=transport, today=lambda: FIXED_DATE
    )
    items = provider.items_for_topic("CFTR:p.Phe508del")
    assert "gnomad" in {i.source_id for i in items}


# -- honest failures: no gnomAD item, no exception, no guess -----------------


def _single_record_transport(refsnp_payload=None, refsnp_error=False):
    def transport(url, body=None):
        if "esearch" in url and "db=clinvar" in url:
            return {"esearchresult": {"idlist": ["15333"]}}
        if "esummary" in url and "db=pubmed" in url:
            return pubmed_fixture(url)
        if "esummary" in url:
            record = {
                "title": "NM_000518.5(HBB):c.20A>T (p.Glu7Val)",
                "protein_change": "E7V",
                "germline_classification": {"description": "Pathogenic"},
                "variation_set": [
                    {
                        "variation_xrefs": [
                            {"db_source": "dbSNP", "db_id": "334"}
                        ],
                        # No canonical_spdi: the multi-allelic rsID
                        # cannot be disambiguated.
                        "variation_loc": [
                            {
                                "assembly_name": "GRCh38",
                                "chr": "11",
                                "start": "5227002",
                                "ref": "",
                                "alt": "",
                            }
                        ],
                    }
                ],
            }
            return {"result": {"uids": ["15333"], "15333": record}}
        if "variation/v0/refsnp/" in url:
            if refsnp_error:
                raise ConnectionError("variation api down")
            return refsnp_payload
        if "esearch" in url and "db=snp" in url:
            return {"esearchresult": {"idlist": []}}
        if "gnomad" in url:
            raise AssertionError("gnomAD must not be queried without alleles")
        if "europepmc" in url:
            return literature_fixture()
        raise AssertionError(f"unexpected URL: {url}")

    return transport


def test_ambiguous_multiallelic_without_canonical_match_gives_no_gnomad():
    provider = LiveEvidenceProvider(
        transport=_single_record_transport(refsnp_payload=HBB_REFSNP_334),
        today=lambda: FIXED_DATE,
    )
    items = provider.items_for_topic("HBB:p.Glu6Val")
    sources = {item.source_id for item in items}
    assert "gnomad" not in sources
    assert "clinvar" in sources  # the rest of the answer is unaffected


def test_refsnp_failure_gives_no_gnomad_and_no_exception():
    calls = {"refsnp": 0}

    def counting(url, body=None):
        if "variation/v0/refsnp/" in url:
            calls["refsnp"] += 1
        return _single_record_transport(refsnp_error=True)(url, body)

    provider = LiveEvidenceProvider(
        transport=counting, today=lambda: FIXED_DATE
    )
    items = provider.items_for_topic("HBB:p.Glu6Val")
    assert {i.source_id for i in items} == {"clinvar", "pubmed-literature"}
    assert calls["refsnp"] == 2  # one retry, then silence


def test_fallback_snp_search_requires_canonical_match():
    # No rsID on the ClinVar measure; the snp esearch fallback finds
    # rs334, but with no canonical SPDI on the measure the
    # multi-allelic record must not be trusted.
    def transport(url, body=None):
        if "esearch" in url and "db=snp" in url:
            return {"esearchresult": {"idlist": ["334"]}}
        return _single_record_transport(refsnp_payload=HBB_REFSNP_334)(
            url, body
        )

    # Remove the xref so only the fallback can find the rsID.
    base = _single_record_transport(refsnp_payload=HBB_REFSNP_334)

    def no_xref_transport(url, body=None):
        if "esummary" in url:
            payload = base(url, body)
            measure = payload["result"]["15333"]["variation_set"][0]
            measure["variation_xrefs"] = []
            return payload
        return transport(url, body)

    provider = LiveEvidenceProvider(
        transport=no_xref_transport, today=lambda: FIXED_DATE
    )
    items = provider.items_for_topic("HBB:p.Glu6Val")
    assert "gnomad" not in {i.source_id for i in items}
