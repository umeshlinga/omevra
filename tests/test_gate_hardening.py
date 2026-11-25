"""Gate hardening regression tests (2026-11-25 red-team follow-up).

An outside review demonstrated that fabricated papers handed in
through a faked data feed were accepted as VERIFIED: every live
provider marked literature VERIFIED on the strength of the single
Europe PMC search response that supplied it, and a "PMID" that was
not even numeric passed.  These tests pin the fixes:

* literature in every live provider is independently re-confirmed
  against PubMed (NCBI E-utilities) with a title match before it
  counts as VERIFIED (provome.confirmation);
* a fabricated or title-swapped paper is FAILED, an uncompletable
  lookup is UNVERIFIED, and neither counts toward the gate;
* the same paper returned twice by a feed counts once at the gate;
* ClinVar's transcript-version restatements of one interpretation
  emit one item, while genuinely different classifications do not
  collapse;
* the RAG bridge's default ingest-trust rule is pinned honestly,
  and the bridge with a live confirmer wired in demotes fabricated
  corpus documents.

Fixture transports only; the suite never touches the network.  In
these fixture worlds PubMed is honest: it resolves exactly the
PMIDs recorded as real, with their real titles.
"""

from datetime import date

import pytest

from provome.animal import AnimalEvidenceProvider
from provome.confirmation import (
    confirm_literature,
    make_pubmed_confirmer,
    titles_plausibly_match,
)
from provome.copilot import Copilot, Verdict
from provome.drug import DrugEvidenceProvider
from provome.evidence import VerificationStatus, evaluate_gate
from provome.live import LiveEvidenceProvider
from provome.plant import PlantEvidenceProvider
from provome.rag import (
    Document,
    LexicalEmbedder,
    RetrievalIndex,
    evidence_from_retrieval,
)

FIXED = date(2026, 11, 25)

FAKE_LIT = [
    {"title": "Fabricated breakthrough: gene therapy cures everything.",
     "pmid": "99999991", "pubYear": "2026"},
    {"title": "Invented cohort study with impossible results.",
     "pmid": "99999992", "pubYear": "2026"},
]


def honest_pubmed(known):
    """A transport fragment: PubMed esummary for `known` pmid->title."""

    def serve(url):
        pmid = url.rsplit("id=", 1)[1]
        title = known.get(pmid)
        if title is None:
            return {"result": {"uids": []}}
        return {"result": {"uids": [pmid], pmid: {"title": title}}}

    return serve


def _lit(records):
    return {"resultList": {"result": records}}


# -- confirmation unit behaviour -------------------------------------------


def test_titles_plausibly_match_normalises_markup_and_entities():
    claimed = "Antagonistic regulation by <i>HY5</i> &amp; BPC1 in plants."
    confirmed = "Antagonistic regulation by HY5 & BPC1 in plants"
    assert titles_plausibly_match(claimed, confirmed)


def test_titles_plausibly_match_subtitle_superset():
    assert titles_plausibly_match(
        "Sickle cell disease review",
        "Sickle cell disease review: an update for clinicians",
    )


def test_titles_plausibly_match_rejects_a_different_paper():
    assert not titles_plausibly_match(
        "Sickle cell disease review", "Quantum computing advances"
    )
    assert not titles_plausibly_match("", "Anything at all")


def test_confirm_malformed_pmid_fails_without_any_lookup():
    calls = []

    def transport(url, body=None):
        calls.append(url)
        raise AssertionError("must not be called")

    status, note = confirm_literature(transport, "not-a-pmid", "Any title")
    assert status is VerificationStatus.FAILED
    assert calls == []


def test_confirm_unknown_pmid_fails_and_lookup_error_is_unverified():
    serve = honest_pubmed({})
    status, _ = confirm_literature(
        lambda url, body=None: serve(url), "99999991", "Fabricated."
    )
    assert status is VerificationStatus.FAILED

    def down(url, body=None):
        raise ConnectionError("pubmed unreachable")

    status, _ = confirm_literature(down, "11111111", "A real title.")
    assert status is VerificationStatus.UNVERIFIED


# -- live provider: fake feed attacks ---------------------------------------


def _live_transport(lit_records, pubmed_known):
    serve = honest_pubmed(pubmed_known)

    def transport(url, body=None):
        if "esummary" in url and "db=pubmed" in url:
            return serve(url)
        if "esearch" in url and "db=clinvar" in url:
            return {"esearchresult": {"idlist": ["999"]}}
        if "esearch" in url and "db=snp" in url:
            return {"esearchresult": {"idlist": []}}
        if "esummary" in url:
            return {"result": {"uids": ["999"], "999": {
                "title": "NM_000492.4(CFTR):c.1521_1523del (p.Phe508del)",
                "protein_change": "F508del",
                "germline_classification": {"description": "Pathogenic"},
                "variation_set": [{"variation_loc": [
                    {"assembly_name": "GRCh38", "chr": "7",
                     "start": "117559590", "ref": "", "alt": ""}]}],
            }}}
        if "europepmc" in url:
            return _lit(lit_records)
        raise AssertionError(url)

    return transport


def test_live_fabricated_papers_never_count():
    provider = LiveEvidenceProvider(
        transport=_live_transport(FAKE_LIT, {}), today=lambda: FIXED
    )
    items = provider.items_for_topic("CFTR:p.Phe508del")
    lit = [i for i in items if i.source_id == "pubmed-literature"]
    assert len(lit) == 2
    assert all(i.status is VerificationStatus.FAILED for i in lit)
    answer = Copilot(provider=provider).ask(
        "What is known about CFTR p.Phe508del?"
    )
    assert answer.verdict is Verdict.DECLINE


def test_live_title_swapped_paper_fails_confirmation():
    """A real PMID wearing a fabricated title: PubMed's own title
    for the PMID exposes the swap."""
    swapped = [{"title": "Fabricated claim riding on a real PMID.",
                "pmid": "11111111", "pubYear": "2026"}]
    provider = LiveEvidenceProvider(
        transport=_live_transport(
            swapped, {"11111111": "Sickle cell disease review."}
        ),
        today=lambda: FIXED,
    )
    lit = [i for i in provider.items_for_topic("CFTR:p.Phe508del")
           if i.source_id == "pubmed-literature"]
    assert lit[0].status is VerificationStatus.FAILED
    assert "different title" in lit[0].notes


def test_live_non_numeric_pmid_fails():
    weird = [{"title": "Shape-only identifier paper.",
              "pmid": "not-a-pmid", "pubYear": "2026"}]
    provider = LiveEvidenceProvider(
        transport=_live_transport(weird, {}), today=lambda: FIXED
    )
    lit = [i for i in provider.items_for_topic("CFTR:p.Phe508del")
           if i.source_id == "pubmed-literature"]
    assert lit[0].status is VerificationStatus.FAILED


def test_live_confirmation_outage_demotes_to_unverified_not_verified():
    def transport(url, body=None):
        if "esummary" in url and "db=pubmed" in url:
            raise ConnectionError("pubmed down")
        return _live_transport(
            [{"title": "A real paper.", "pmid": "11111111",
              "pubYear": "2024"}], {}
        )(url, body)

    provider = LiveEvidenceProvider(transport=transport, today=lambda: FIXED)
    lit = [i for i in provider.items_for_topic("CFTR:p.Phe508del")
           if i.source_id == "pubmed-literature"]
    assert lit[0].status is VerificationStatus.UNVERIFIED
    gate = evaluate_gate(
        provider.items_for_topic("CFTR:p.Phe508del"), "CFTR:p.Phe508del"
    )
    assert not gate.passed


def test_live_same_paper_twice_counts_once_at_the_gate():
    rec = {"title": "Sickle cell disease review.", "pmid": "11111111",
           "pubYear": "2024"}
    provider = LiveEvidenceProvider(
        transport=_live_transport(
            [rec, dict(rec)], {"11111111": "Sickle cell disease review."}
        ),
        today=lambda: FIXED,
    )
    items = provider.items_for_topic("CFTR:p.Phe508del")
    assert len(items) == 3  # provider emits both copies
    gate = evaluate_gate(items, "CFTR:p.Phe508del")
    # 1 ClinVar + the paper counted ONCE = 2, below the gate.
    assert gate.verified_count == 2
    assert not gate.passed


# -- live provider: ClinVar restatement dedupe -------------------------------


def _clinvar_transport(classifications):
    serve = honest_pubmed({"11111111": "One real paper."})

    def record(uid, title, classification):
        return {
            "title": title,
            "protein_change": "F508del",
            "germline_classification": {"description": classification},
            "variation_set": [{"variation_loc": [
                {"assembly_name": "GRCh38", "chr": "7",
                 "start": "117559590", "ref": "", "alt": ""}]}],
        }

    def transport(url, body=None):
        if "esummary" in url and "db=pubmed" in url:
            return serve(url)
        if "esearch" in url and "db=clinvar" in url:
            return {"esearchresult": {"idlist": ["7105", "4072070"]}}
        if "esearch" in url and "db=snp" in url:
            return {"esearchresult": {"idlist": []}}
        if "esummary" in url:
            return {"result": {
                "uids": ["7105", "4072070"],
                "7105": record(
                    "7105",
                    "NM_000492.3(CFTR):c.1521_1523del (p.Phe508del)",
                    classifications[0],
                ),
                "4072070": record(
                    "4072070",
                    "NM_000492.4(CFTR):c.1522_1524del (p.Phe508del)",
                    classifications[1],
                ),
            }}
        if "europepmc" in url:
            return _lit([{"title": "One real paper.", "pmid": "11111111",
                          "pubYear": "2024"}])
        raise AssertionError(url)

    return transport


def test_clinvar_transcript_restatement_emits_one_item():
    provider = LiveEvidenceProvider(
        transport=_clinvar_transport(["Pathogenic", "Pathogenic"]),
        today=lambda: FIXED,
    )
    clinvar = [i for i in provider.items_for_topic("CFTR:p.Phe508del")
               if i.source_id == "clinvar"]
    assert len(clinvar) == 1


def test_clinvar_different_classifications_do_not_collapse():
    provider = LiveEvidenceProvider(
        transport=_clinvar_transport(
            ["Pathogenic", "Uncertain significance"]
        ),
        today=lambda: FIXED,
    )
    clinvar = [i for i in provider.items_for_topic("CFTR:p.Phe508del")
               if i.source_id == "clinvar"]
    assert len(clinvar) == 2


# -- animal / plant / drug: fake feed attacks --------------------------------


def test_animal_fabricated_papers_never_count():
    serve = honest_pubmed({})

    def transport(url, body=None):
        if "esummary" in url and "db=pubmed" in url:
            return serve(url)
        if "lookup/symbol" in url:
            return {"id": "ENSCAFG00000000001", "display_name": "ABCB1",
                    "biotype": "protein_coding",
                    "assembly_name": "CanFam3.1",
                    "description": "ATP binding cassette B member 1"}
        if "phenotype/gene" in url:
            return []
        if "europepmc" in url:
            return _lit(FAKE_LIT)
        raise AssertionError(url)

    provider = AnimalEvidenceProvider(transport=transport, today=lambda: FIXED)
    items = provider.items_for_topic("animal:dog:ABCB1")
    lit = [i for i in items if i.source_id == "pubmed-literature"]
    assert all(i.status is VerificationStatus.FAILED for i in lit)
    assert not evaluate_gate(items, "animal:dog:ABCB1").passed


def test_plant_fabricated_papers_never_count():
    serve = honest_pubmed({})

    def transport(url, body=None):
        if "esummary" in url and "db=pubmed" in url:
            return serve(url)
        if "lookup/symbol" in url:
            return {"id": "AT5G10140", "display_name": "FLC",
                    "biotype": "protein_coding",
                    "assembly_name": "TAIR10",
                    "description": "MADS-box transcription factor"}
        if "phenotype/gene" in url:
            return []
        if "europepmc" in url:
            return _lit(FAKE_LIT)
        raise AssertionError(url)

    provider = PlantEvidenceProvider(transport=transport, today=lambda: FIXED)
    items = provider.items_for_topic("plant:arabidopsis:FLC")
    lit = [i for i in items if i.source_id == "pubmed-literature"]
    assert all(i.status is VerificationStatus.FAILED for i in lit)
    assert not evaluate_gate(items, "plant:arabidopsis:FLC").passed


def test_drug_fabricated_papers_never_count():
    serve = honest_pubmed({})

    def transport(url, body=None):
        if "esummary" in url and "db=pubmed" in url:
            return serve(url)
        if "uniprot" in url:
            return {"results": [{"primaryAccession": "P13569"}]}
        if "target/search" in url:
            return {"targets": [{
                "target_chembl_id": "CHEMBL4051",
                "organism": "Homo sapiens", "pref_name": "CFTR",
                "target_type": "SINGLE PROTEIN"}]}
        if "target/CHEMBL4051" in url:
            return {"target_components": [{"accession": "P13569"}]}
        if "activity" in url:
            return {"page_meta": {"total_count": 3728}}
        if "mechanism" in url:
            return {"mechanisms": [{
                "molecule_chembl_id": "CHEMBL2103872",
                "mechanism_of_action": "CFTR positive modulator"}]}
        if "molecule/CHEMBL2103872" in url:
            return {"pref_name": "IVACAFTOR", "max_phase": 4.0}
        if "europepmc" in url:
            return _lit(FAKE_LIT)
        raise AssertionError(url)

    provider = DrugEvidenceProvider(transport=transport, today=lambda: FIXED)
    items = provider.items_for_topic("drug:CFTR")
    lit = [i for i in items if i.source_id == "pubmed-literature"]
    assert all(i.status is VerificationStatus.FAILED for i in lit)
    assert not evaluate_gate(items, "drug:CFTR").passed


# -- RAG bridge: ingest trust pinned, confirmer demotes fabrications ---------


def _fabricated_index():
    index = RetrievalIndex(LexicalEmbedder())
    for n in (1, 2, 3):
        index.add(Document(
            doc_id=f"pmid:9999999{n}",
            title=f"Fabricated paper number {n} about CFTR Phe508del",
            text="CFTR Phe508del fabricated findings " * 5,
            source="pubmed-literature",
            url=f"https://pubmed.ncbi.nlm.nih.gov/9999999{n}/",
            fetched_on=FIXED,
        ))
    return index


def test_rag_bridge_default_rule_trusts_corpus_ingest():
    """Pinned honestly: with no confirmer, the bridge's rule is
    identifier-carriage — its trust boundary is corpus ingest.
    This test documents that boundary; it is not an endorsement."""
    items = evidence_from_retrieval(
        _fabricated_index(), "CFTR Phe508del",
        topic_key="CFTR:p.Phe508del", k=3,
    )
    assert len(items) == 3
    assert all(i.is_verified for i in items)


def test_rag_bridge_with_confirmer_demotes_fabricated_corpus():
    serve = honest_pubmed({})
    confirmer = make_pubmed_confirmer(lambda url, body=None: serve(url))
    items = evidence_from_retrieval(
        _fabricated_index(), "CFTR Phe508del",
        topic_key="CFTR:p.Phe508del", k=3, confirmer=confirmer,
    )
    assert len(items) == 3
    assert all(i.status is VerificationStatus.FAILED for i in items)
    gate = evaluate_gate(items, "CFTR:p.Phe508del")
    assert gate.verified_count == 0
    assert not gate.passed
