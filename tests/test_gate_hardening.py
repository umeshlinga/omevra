"""Regression tests for the cite-or-decline gate hardening.

Each test pins an attack or failure mode that the gate previously let
through: fabricated literature, double counting, demo data in
production, disappearing refuters, and unchecked LLM narrative.
"""

import pytest

from provome.copilot import Claim, Copilot, ungrounded_narrative_tokens
from provome.evidence import (
    EvidenceItem,
    EvidenceKind,
    SupportDirection as D,
    VerificationStatus as S,
    canonical_evidence_id,
    deduplicate,
    default_source_registry,
    demo_source_registry,
    evaluate_gate,
)
from provome.live import clinvar_direction
from provome.rag import (
    Document,
    EuropePmcPmidVerifier,
    RetrievalIndex,
    document_pmid,
    evidence_from_retrieval,
)

T = "cftr:p.phe508del"


def it(item_id, source, citation, direction=D.SUPPORTS, status=S.VERIFIED,
       topic=T, locator=None):
    return EvidenceItem(
        item_id=item_id, topic_key=topic, statement=f"statement {item_id}",
        source_id=source, citation=citation, kind=EvidenceKind.OTHER,
        direction=direction, status=status, locator=locator,
    )


# -- 1. literature bridge ------------------------------------------------

def _index(*docs):
    index = RetrievalIndex()
    index.add_many(docs)
    return index


def test_fabricated_pmids_never_pass_the_gate():
    index = _index(
        Document("pmid:99999991", "CFTR F508del is pathogenic", "cftr f508del", source="pubmed-literature"),
        Document("pmid:99999992", "CFTR F508del again", "cftr f508del", source="pubmed-literature"),
        Document("pmid:99999993", "CFTR F508del curated", "cftr f508del", source="clinvar"),
    )
    items = evidence_from_retrieval(index, "cftr f508del", topic_key=T)
    assert all(i.status is S.UNVERIFIED for i in items)
    assert not evaluate_gate(items, T).passed


def test_document_cannot_claim_another_source():
    index = _index(Document("pmid:12345", "t", "cftr", source="clinvar"))
    assert evidence_from_retrieval(index, "cftr", topic_key=T) == []


def test_pmid_extraction_is_anchored():
    for doc_id in ("notpmid-1234-xyz", "lab-pmid1234-copy", "pmid:0123", "pmid:1234567890"):
        assert document_pmid(Document(doc_id, "t", "x", source="pubmed-literature")) is None
    off_host = Document("d1", "t", "x", source="pubmed-literature", url="https://evil.example/?pmid=1234")
    assert document_pmid(off_host) is None
    epmc = Document("d2", "t", "x", source="pubmed-literature", url="https://europepmc.org/article/MED/1234")
    assert document_pmid(epmc) == "1234"


def test_verifier_outcomes_map_to_status():
    outcomes = {"111": True, "222": False, "333": None}

    class Fake:
        def verify(self, pmid, title):
            return outcomes[pmid]

    index = _index(*[Document(f"pmid:{p}", f"paper {p}", "cftr", source="pubmed-literature") for p in outcomes])
    got = {i.item_id: i.status for i in evidence_from_retrieval(index, "cftr", topic_key=T, verifier=Fake())}
    assert got == {"rag-pmid-111": S.VERIFIED, "rag-pmid-222": S.FAILED, "rag-pmid-333": S.UNVERIFIED}


def test_europepmc_verifier_checks_existence_and_title():
    def transport(url, body):
        assert "EXT_ID%3A123" in url
        return {"resultList": {"result": [{"pmid": "123", "title": "CFTR folding defect in F508del"}]}}

    v = EuropePmcPmidVerifier(transport=transport)
    assert v.verify("123", "CFTR folding defect in F508del") is True
    assert v.verify("123", "Venetoclax in leukemia") is False
    assert EuropePmcPmidVerifier(transport=lambda u, b: {"resultList": {"result": []}}).verify("123", "x") is False

    def broken(url, body):
        raise OSError("network down")

    assert EuropePmcPmidVerifier(transport=broken).verify("123", "x") is None


# -- 2. one evidence line counts once ---------------------------------------

def test_same_paper_through_two_providers_counts_once():
    items = [
        it("a", "clinvar", "ClinVar Variation ID 7105 (fetched 2026-11-24 via NCBI E-utilities)"),
        it("b", "pubmed-literature", "PMID 12345 via Europe PMC (fetched 2026-11-24)"),
        it("c", "pubmed-literature", "PMID 12345 (indexed copy fetched 2026-11-01)"),
    ]
    d = evaluate_gate(items, T)
    assert not d.passed and d.verified_count == 2
    assert any("collapsed" in w for w in d.warnings)


def test_same_record_fetched_on_two_days_counts_once():
    items = [
        it("a", "clinvar", "ClinVar Variation ID 7105 (fetched 2026-11-24 via NCBI E-utilities)"),
        it("b", "clinvar", "ClinVar Variation ID 7105 (fetched 2026-11-25 via NCBI E-utilities)"),
        it("c", "omim", "OMIM 602421.0001"),
    ]
    assert not evaluate_gate(items, T).passed


def test_canonical_ids():
    assert canonical_evidence_id(it("a", "x", "PMID: 00123")) == "pmid:123"
    assert canonical_evidence_id(it("a", "x", "see paper", locator="https://pubmed.ncbi.nlm.nih.gov/123/")) == "pmid:123"
    assert canonical_evidence_id(it("a", "clinvar", "VCV000007105")) == "clinvar:7105"
    assert canonical_evidence_id(it("a", "clinvar", "ClinVar Variation ID 7105")) == "clinvar:7105"
    # Distinct records sharing a target id must stay distinct.
    target = it("a", "chembl", "ChEMBL target CHEMBL1234 (fetched 2026-11-24 via the ChEMBL API)")
    mech = it("b", "chembl", "ChEMBL mechanisms for CHEMBL1234 (fetched 2026-11-25 via the ChEMBL API)")
    assert canonical_evidence_id(target) != canonical_evidence_id(mech)
    again = it("c", "chembl", "ChEMBL target CHEMBL1234 (fetched 2026-12-01 via the ChEMBL API)")
    assert canonical_evidence_id(target) == canonical_evidence_id(again)


def test_failed_check_is_sticky():
    items = [
        it("a", "clinvar", "c1", status=S.FAILED),
        it("a2", "clinvar", "c1"),
        it("b", "omim", "c2"),
        it("c", "pubmed-literature", "c3"),
    ]
    assert [i.status for i in deduplicate(items)][0] is S.FAILED
    assert not evaluate_gate(items, T).passed


# -- 3. direction from content -----------------------------------------------

@pytest.mark.parametrize("sig,expected", [
    ("Pathogenic", D.SUPPORTS),
    ("Pathogenic/Likely pathogenic", D.SUPPORTS),
    ("Benign", D.REFUTES),
    ("Likely benign", D.REFUTES),
    ("Uncertain significance", D.NEUTRAL),
    ("Conflicting classifications of pathogenicity", D.NEUTRAL),
    ("risk factor", D.NEUTRAL),
])
def test_clinvar_direction(sig, expected):
    assert clinvar_direction(sig) is expected


# -- 4. registry, refuters, topic keys -------------------------------------

def test_demo_sources_absent_from_production_registry():
    prod = default_source_registry()
    assert "demo-bundled" not in prod and "demo-bundled-2" not in prod
    items = [it("a", "demo-bundled", "d1"), it("b", "demo-bundled", "d2"), it("c", "demo-bundled-2", "d3")]
    assert not evaluate_gate(items, T).passed
    assert evaluate_gate(items, T, registry=demo_source_registry()).passed


def test_demo_registry_only_paired_with_demo_provider():
    assert "demo-bundled" in Copilot().registry

    class EmptyProvider:
        def topic_for_question(self, text):
            return None

        def items_for_topic(self, key):
            return []

    assert "demo-bundled" not in Copilot(provider=EmptyProvider()).registry


def test_refuter_sharing_a_citation_is_not_deduplicated_away():
    items = [
        it("s1", "clinvar", "VCV000007105"),
        it("r1", "clinvar", "VCV000007105", D.REFUTES),
        it("s2", "omim", "OMIM 602421.0001"),
        it("s3", "pubmed-literature", "PMID 1"),
    ]
    assert len(evaluate_gate(items, T).verified_refuting) == 1


def test_unregistered_refuters_cannot_veto():
    items = [it("a", "clinvar", "c1"), it("b", "omim", "c2"), it("c", "pubmed-literature", "c3"),
             *[it(f"x{i}", "random-blog", f"blog {i}", D.REFUTES) for i in range(3)]]
    d = evaluate_gate(items, T)
    assert d.passed and not d.verified_refuting
    assert any("refuting item(s) were excluded" in w for w in d.warnings)


def test_topic_key_case_and_whitespace_insensitive():
    items = [it("a", "clinvar", "c1", topic="CFTR:p.Phe508del"), it("b", "omim", "c2"),
             it("c", "pubmed-literature", "c3", topic=" cftr:p.phe508del ")]
    assert evaluate_gate(items, "CFTR:P.PHE508DEL").passed


# -- 5. LLM narrative -------------------------------------------------------

def test_narrative_tokens_must_be_grounded():
    claims = (Claim(statement="ClinVar records the variant as Pathogenic.",
                    citations=("PMID 12345",), evidence_item_ids=("a",)),)
    assert ungrounded_narrative_tokens("Pathogenic per PMID 12345.", claims) == []
    assert ungrounded_narrative_tokens("Seen in 87% of patients (PMID 999).", claims) == ["87%", "PMID 999"]


def test_ungrounded_narrative_is_withheld():
    class Inventive:
        def rephrase(self, answer):
            return "This variant is found in 92% of cases (PMID 31415926)."

    answer = Copilot(llm_hook=Inventive(), use_llm=True).ask("What is known about CFTR p.Phe508del?")
    assert answer.answered
    assert answer.narrative is None
    assert any("narrative withheld" in w for w in answer.warnings)


# -- 6. refuter policy (Founder's call, 2026-11-25) ---------------------------
# Verified refuters from registered sources block a clean pass whenever
# their count reaches at least half the supporting count, rounded up.
# Below that, the gate passes but the conflict must be disclosed.

def test_two_refuters_block_three_supporters():
    items = [
        it("a", "clinvar", "c1"), it("b", "omim", "c2"),
        it("c", "pubmed-literature", "c3"),
        it("r1", "clinvar", "c4", D.REFUTES),
        it("r2", "omim", "c5", D.REFUTES),
    ]
    d = evaluate_gate(items, T)
    assert not d.passed
    # The blocking reason names both sides.
    assert any("2 verified item(s) refute" in r and "3 supporting" in r
               for r in d.reasons)


def test_one_refuter_against_three_supporters_passes_with_disclosure():
    items = [
        it("a", "clinvar", "c1"), it("b", "omim", "c2"),
        it("c", "pubmed-literature", "c3"),
        it("r1", "clinvar", "c4", D.REFUTES),
    ]
    d = evaluate_gate(items, T)
    assert d.passed
    assert len(d.verified_refuting) == 1
    assert any("1 verified item(s) refute the claim" in w for w in d.warnings)
