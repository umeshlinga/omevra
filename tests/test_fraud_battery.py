"""The fraud battery: seeded adversarial cases for the agent layer.

The Founder's question (2026-11-25): can the agents detect fraud —
duplicates and fake things?  This battery is the permanent,
executable answer.  Each fraud type is seeded deliberately and the
expected detection is asserted.  Two types are only partially
detected today; those cases are marked KNOWN GAP in their names and
docstrings, assert the current behaviour so the gap is pinned and
visible, and must be flipped the day a fix lands — a gap that fails
silently is worse than a gap that is named.

Fraud types:
    1. fabricated PMIDs
    2. real PMIDs with wrong titles
    3. documents claiming a source they are not
    4. the same record duplicated under different fetch dates,
       formatting, and casing
    5. a refuting record disguised with supporting language
    6. a Benign ClinVar record dressed as pathogenic support
    7. unregistered sources trying to support, and to veto
    8. a FAILED-verification record resubmitted in verified clothing

All transports are fixtures; the suite never touches the network.
"""

from datetime import date

from provome.agents import AgentBackedProvider, VerificationAgent
from provome.confirmation import make_pubmed_confirmer
from provome.copilot import Copilot, Verdict
from provome.domains import Division
from provome.evidence import (
    EvidenceItem,
    EvidenceKind,
    SupportDirection as D,
    VerificationStatus as S,
    deduplicate,
    evaluate_gate,
)
from provome.live import LiveEvidenceProvider
from provome.rag import Document, RetrievalIndex, evidence_from_retrieval

T = "cftr:p.phe508del"
FIXED = date(2026, 11, 25)

_TITLES = {"111": "Real Paper Title", "222": "Another Real Title"}


def pubmed_transport(url, body=None):
    pmid = url.rsplit("id=", 1)[1].split("&")[0]
    title = _TITLES.get(pmid)
    if title is None:
        return {"result": {"uids": []}}
    return {"result": {"uids": [pmid], pmid: {"title": title}}}


def item(item_id, source, citation, direction=D.SUPPORTS, status=S.VERIFIED,
         statement=None, kind=EvidenceKind.OTHER, locator=None):
    return EvidenceItem(
        item_id=item_id, topic_key=T,
        statement=statement or f"statement {item_id}",
        source_id=source, citation=citation, direction=direction,
        status=status, kind=kind, locator=locator,
    )


def lit(item_id, pmid_text, title, status=S.VERIFIED):
    return EvidenceItem(
        item_id=item_id, topic_key=T,
        statement=f'Peer-reviewed literature reports: "{title}"',
        source_id="pubmed-literature", citation=f"PMID {pmid_text}",
        kind=EvidenceKind.LITERATURE_REPORT, direction=D.SUPPORTS,
        status=status,
        locator=f"https://pubmed.ncbi.nlm.nih.gov/{pmid_text}/",
    )


def genuine_trio():
    return [
        item("g-cv", "clinvar", "ClinVar Variation ID 7105"),
        item("g-om", "omim", "OMIM 602421.0001"),
        lit("g-lit", "111", "Real Paper Title"),
    ]


# ---------------------------------------------------------------------
# 1. Fabricated PMIDs
# ---------------------------------------------------------------------


def test_fraud_fabricated_pmid_fails_verification_agent():
    out, _ = VerificationAgent(pubmed_transport).verify(
        [lit("fake", "999", "Invented Paper")]
    )
    assert out[0].status is S.FAILED


def test_fraud_fabricated_pmid_stopped_in_composition():
    class LyingProvider:
        name = "lying-fixture"
        scoped_divisions = frozenset({Division.HUMAN})

        def topic_for_question(self, text):
            return T

        def items_for_topic(self, topic_key):
            return (
                item("cv", "clinvar", "ClinVar Variation ID 7105"),
                lit("fake-1", "999", "Invented Paper"),
                lit("fake-2", "998", "Another Invented Paper"),
            )

    answer = Copilot(
        provider=AgentBackedProvider(LyingProvider(), pubmed_transport)
    ).ask("What is known about CFTR p.Phe508del?")
    assert answer.verdict is Verdict.DECLINE


# ---------------------------------------------------------------------
# 2. Real PMIDs with wrong titles
# ---------------------------------------------------------------------


def test_fraud_real_pmid_wrong_title_fails_verification_agent():
    out, _ = VerificationAgent(pubmed_transport).verify(
        [lit("wrong", "111", "A Completely Different Paper")]
    )
    assert out[0].status is S.FAILED


def test_fraud_real_pmid_wrong_title_fails_rag_bridge():
    index = RetrievalIndex()
    index.add(Document(
        doc_id="pmid:111", title="A Completely Different Paper",
        text="cftr phe508del", source="pubmed-literature",
        url="https://pubmed.ncbi.nlm.nih.gov/111/", fetched_on=FIXED,
    ))
    items = evidence_from_retrieval(
        index, "cftr phe508del", topic_key=T,
        confirmer=make_pubmed_confirmer(pubmed_transport),
    )
    assert len(items) == 1
    assert items[0].status is S.FAILED


# ---------------------------------------------------------------------
# 3. Documents claiming a source they are not
# ---------------------------------------------------------------------


def test_fraud_document_claiming_clinvar_source_is_excluded():
    index = RetrievalIndex()
    index.add(Document(
        doc_id="pmid:111", title="Real Paper Title",
        text="cftr phe508del", source="clinvar",  # claims ClinVar
        url="https://pubmed.ncbi.nlm.nih.gov/111/", fetched_on=FIXED,
    ))
    items = evidence_from_retrieval(
        index, "cftr phe508del", topic_key=T,
        confirmer=make_pubmed_confirmer(pubmed_transport),
    )
    assert items == []


def test_fraud_document_without_pmid_is_excluded():
    index = RetrievalIndex()
    index.add(Document(
        doc_id="internal-note-7", title="Real Paper Title",
        text="cftr phe508del", source="pubmed-literature",
        url=None, fetched_on=FIXED,
    ))
    items = evidence_from_retrieval(
        index, "cftr phe508del", topic_key=T,
        confirmer=make_pubmed_confirmer(pubmed_transport),
    )
    assert items == []


def test_known_gap_item_level_source_spoof_is_not_detected():
    """KNOWN GAP: an EvidenceItem that simply *declares* a registered
    non-literature source (here a fabricated ClinVar citation) is
    trusted by the gate and untouched by the VerificationAgent,
    whose remit is literature only.  Detection today relies on
    providers being honest; a ClinVar confirmer does not exist.
    This test pins the gap: when one lands, flip this test."""
    spoof = item("spoof", "clinvar", "ClinVar Variation ID 987654321")
    out, _ = VerificationAgent(pubmed_transport).verify([spoof])
    assert out[0].status is S.VERIFIED  # untouched: the gap
    gate = evaluate_gate([spoof] + genuine_trio()[1:], T)
    assert gate.passed  # counted: the gap


# ---------------------------------------------------------------------
# 4. Duplicates under different dates / formatting / casing
# ---------------------------------------------------------------------


def test_fraud_duplicate_clinvar_record_counts_once():
    copies = [
        item("c1", "clinvar",
             "ClinVar Variation ID 7105 (fetched 2026-11-24 via NCBI E-utilities)"),
        item("c2", "clinvar", "clinvar variation id: 7105"),
        item("c3", "clinvar", "ClinVar VCV000007105"),
    ]
    assert len(deduplicate(copies)) == 1
    assert not evaluate_gate(copies, T).passed


def test_fraud_duplicate_paper_counts_once():
    copies = [
        item("p1", "pubmed-literature",
             "PMID 12345 via Europe PMC (fetched 2026-11-24)",
             kind=EvidenceKind.LITERATURE_REPORT),
        item("p2", "pubmed-literature",
             "pmid: 12345 (indexed copy fetched 2026-11-01)",
             kind=EvidenceKind.LITERATURE_REPORT),
        item("p3", "pubmed-literature",
             "https://pubmed.ncbi.nlm.nih.gov/12345/",
             kind=EvidenceKind.LITERATURE_REPORT),
    ]
    assert len(deduplicate(copies)) == 1
    gate = evaluate_gate(copies + [genuine_trio()[0]], T)
    assert not gate.passed  # 2 distinct lines, not 4


# ---------------------------------------------------------------------
# 5. A refuting record disguised with supporting language
# ---------------------------------------------------------------------


def test_fraud_refuter_in_supporting_language_still_blocks():
    prose = "This record strongly supports the pathogenic claim."
    refuters = [
        item("r1", "clinvar", "ClinVar Variation ID 8101",
             direction=D.REFUTES, statement=prose),
        item("r2", "omim", "OMIM 602421.0002",
             direction=D.REFUTES, statement=prose),
    ]
    gate = evaluate_gate(genuine_trio() + refuters, T)
    # Direction governs, prose does not: 2 refuters >= ceil(3/2).
    assert not gate.passed


def test_fraud_single_disguised_refuter_is_disclosed_not_hidden():
    refuter = item(
        "r1", "clinvar", "ClinVar Variation ID 8101",
        direction=D.REFUTES,
        statement="This record strongly supports the pathogenic claim.",
    )
    gate = evaluate_gate(genuine_trio() + [refuter], T)
    assert gate.passed  # below the block threshold
    assert len(gate.verified_refuting) == 1
    assert gate.warnings  # ...but the disagreement is surfaced


# ---------------------------------------------------------------------
# 6. A Benign ClinVar record dressed as pathogenic support
# ---------------------------------------------------------------------


def test_known_gap_dressed_benign_item_is_counted_by_the_gate():
    """KNOWN GAP: a raw item claiming a ClinVar citation, stamped
    VERIFIED, with direction SUPPORTS and pathogenic prose, is
    counted by the gate even if the underlying record is Benign —
    the gate reads the direction field, not the record.  The
    mitigation is architectural (next test): the real live provider
    derives direction from the classification string, so it cannot
    produce this item.  A provider that lies at the item level is
    not detected today."""
    dressed = item(
        "dressed", "clinvar", "ClinVar Variation ID 424242",
        statement="ClinVar classifies this variant as pathogenic.",
    )
    gate = evaluate_gate([dressed] + genuine_trio()[1:], T)
    assert gate.passed  # the dressed item counted: the gap


def test_mitigation_live_provider_derives_direction_from_record():
    """The live provider cannot dress a Benign record as support:
    direction is mapped from the served classification string."""
    def transport(url, body=None):
        if "esearch" in url:
            return {"esearchresult": {"idlist": ["999"]}}
        if "esummary" in url and "db=clinvar" in url:
            return {"result": {"uids": ["999"], "999": {
                "title": "NM_000518.5(HBB):c.20A>T (p.Glu7Val)",
                "protein_change": "E7V",
                "germline_classification": {
                    "description": "Benign",
                    "review_status": "criteria provided, single submitter",
                },
                "variation_set": [],
            }}}
        if "europepmc" in url:
            return {"resultList": {"result": []}}
        raise AssertionError(url)

    provider = LiveEvidenceProvider(transport=transport, today=lambda: FIXED)
    items = provider.items_for_topic(
        provider.topic_for_question("What is known about HBB p.Glu6Val?")
    )
    clinvar_items = [i for i in items if i.source_id == "clinvar"]
    assert clinvar_items
    assert all(i.direction is D.REFUTES for i in clinvar_items)


# ---------------------------------------------------------------------
# 7. Unregistered sources trying to support, and to veto
# ---------------------------------------------------------------------


def test_fraud_unregistered_sources_cannot_support():
    blog = [
        item(f"b{i}", "random-blog", f"blog post {i}")
        for i in range(3)
    ]
    gate = evaluate_gate(blog, T)
    assert gate.verified_count == 0
    assert not gate.passed


def test_fraud_unregistered_sources_cannot_veto():
    blog_refuters = [
        item(f"b{i}", "random-blog", f"blog post {i}", direction=D.REFUTES)
        for i in range(3)
    ]
    gate = evaluate_gate(genuine_trio() + blog_refuters, T)
    assert gate.passed


# ---------------------------------------------------------------------
# 8. A FAILED record resubmitted in verified clothing
# ---------------------------------------------------------------------


def test_fraud_failed_record_resubmitted_verified_stays_dead_at_gate():
    failed_copy = lit("paper-failed", "111", "Real Paper Title", status=S.FAILED)
    verified_copy = lit("paper-verified", "111", "Real Paper Title", status=S.VERIFIED)
    gate = evaluate_gate(
        genuine_trio()[:2] + [failed_copy, verified_copy], T
    )
    # FAILED wins the dedupe: only the 2 genuine items count.
    assert not gate.passed


def test_fraud_failed_record_resubmitted_stays_failed_at_agent():
    out, _ = VerificationAgent(pubmed_transport).verify(
        [lit("dead", "111", "Real Paper Title", status=S.FAILED)]
    )
    assert out[0].status is S.FAILED
