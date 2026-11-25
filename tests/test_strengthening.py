"""Tests for the strengthening pass (2026-11-25).

Three fixes, each pinned here at unit level (the fraud battery pins
them at composition level):

1. ``confirm_clinvar`` — ClinVar items are re-confirmed against the
   served ClinVar record: identity match, and direction derived
   from the served classification, never from the item's claim.
2. Retraction checking — a literature item whose served PubMed
   record carries a retraction notice fails confirmation, and the
   VerificationAgent attaches the integrity warning to the item.
3. The inverted index — ``RetrievalIndex.search`` returns exactly
   what the original linear TF-IDF scan returned (same documents,
   same order, same scores), only faster.
"""

from datetime import date

import pytest

from provome.agents import VerificationAgent
from provome.confirmation import (
    confirm_clinvar,
    confirm_literature_detail,
    record_is_retracted,
)
from provome.evidence import (
    EvidenceItem,
    EvidenceKind,
    SupportDirection as D,
    VerificationStatus as S,
)
from provome.rag import Document, LexicalEmbedder, RetrievalIndex, _cosine

FIXED = date(2026, 11, 25)
TOPIC = "cftr:p.phe508del"

_CLINVAR = {
    "7105": {
        "title": "NM_000492.4(CFTR):c.1521_1523delCTT (p.Phe508del)",
        "protein_change": "F508del",
        "germline_classification": {
            "description": "Pathogenic",
            "review_status": "criteria provided, multiple submitters",
        },
    },
    "424242": {
        "title": "NM_000492.4(CFTR):c.1521_1523delCTT (p.Phe508del)",
        "protein_change": "F508del",
        "germline_classification": {
            "description": "Benign",
            "review_status": "criteria provided, single submitter",
        },
    },
    "555": {  # a real record, but about a different variant
        "title": "NM_000492.4(CFTR):c.350G>A (p.Arg117His)",
        "protein_change": "R117H",
        "germline_classification": {
            "description": "Pathogenic",
            "review_status": "criteria provided, single submitter",
        },
    },
}


def clinvar_transport(url, body=None):
    uid = url.rsplit("id=", 1)[1].split("&")[0]
    record = _CLINVAR.get(uid)
    if record is None:
        return {"result": {"uids": []}}
    return {"result": {"uids": [uid], uid: record}}


# ---------------------------------------------------------------------
# 1. confirm_clinvar
# ---------------------------------------------------------------------


def test_confirm_clinvar_genuine_pathogenic_record():
    status, direction, _ = confirm_clinvar(clinvar_transport, "7105", TOPIC)
    assert status is S.VERIFIED
    assert direction is D.SUPPORTS


def test_confirm_clinvar_direction_comes_from_the_record_not_the_claim():
    # Variation 424242's served record is Benign.  Whatever the
    # item claimed, the confirmed direction is REFUTES.
    status, direction, note = confirm_clinvar(
        clinvar_transport, "424242", TOPIC
    )
    assert status is S.VERIFIED
    assert direction is D.REFUTES
    assert "Benign" in note


def test_confirm_clinvar_fabricated_id_fails():
    status, direction, _ = confirm_clinvar(
        clinvar_transport, "987654321", TOPIC
    )
    assert status is S.FAILED
    assert direction is None


def test_confirm_clinvar_identity_mismatch_fails():
    # Variation 555 exists but is about p.Arg117His, not Phe508del.
    status, _, note = confirm_clinvar(clinvar_transport, "555", TOPIC)
    assert status is S.FAILED
    assert "identity mismatch" in note


def test_confirm_clinvar_transport_failure_is_unverified():
    def boom(url, body=None):
        raise ConnectionError("down")

    status, direction, _ = confirm_clinvar(boom, "7105", TOPIC)
    assert status is S.UNVERIFIED
    assert direction is None


def test_confirm_clinvar_non_numeric_id_fails():
    status, _, _ = confirm_clinvar(clinvar_transport, "abc", TOPIC)
    assert status is S.FAILED


def test_verification_agent_rewrites_a_dressed_direction():
    item = EvidenceItem(
        item_id="dressed", topic_key=TOPIC,
        statement="ClinVar classifies this variant as pathogenic.",
        source_id="clinvar", citation="ClinVar Variation ID 424242",
        kind=EvidenceKind.CURATED_CLASSIFICATION,
        direction=D.SUPPORTS, status=S.VERIFIED,
    )
    out, report = VerificationAgent(clinvar_transport).verify([item])
    assert out[0].direction is D.REFUTES
    assert out[0].status is S.VERIFIED
    assert any("direction" in note for note in report.notes)


# ---------------------------------------------------------------------
# 2. Retraction checking
# ---------------------------------------------------------------------

_PUBMED = {
    "111": {"title": "Real Paper Title"},
    "777": {
        "title": "Withdrawn Paper Title",
        "pubtype": ["Journal Article", "Retracted Publication"],
    },
    "778": {
        "title": "Quietly Withdrawn Paper",
        "commentscorrections": [
            {"reftype": "RetractionIn", "pmid": "999001"}
        ],
    },
}


def pubmed_transport(url, body=None):
    pmid = url.rsplit("id=", 1)[1].split("&")[0]
    record = _PUBMED.get(pmid)
    if record is None:
        return {"result": {"uids": []}}
    return {"result": {"uids": [pmid], pmid: record}}


def test_record_is_retracted_shapes():
    assert record_is_retracted(_PUBMED["777"])
    assert record_is_retracted(_PUBMED["778"])
    assert not record_is_retracted(_PUBMED["111"])
    assert not record_is_retracted({})


def test_confirm_literature_retracted_paper_fails_with_flag():
    status, note, retracted = confirm_literature_detail(
        pubmed_transport, "777", "Withdrawn Paper Title"
    )
    assert status is S.FAILED
    assert retracted
    assert "integrity warning" in note


def test_confirm_literature_retraction_link_shape_also_fails():
    status, _, retracted = confirm_literature_detail(
        pubmed_transport, "778", "Quietly Withdrawn Paper"
    )
    assert status is S.FAILED
    assert retracted


def test_confirm_literature_clean_control_still_verifies():
    status, _, retracted = confirm_literature_detail(
        pubmed_transport, "111", "Real Paper Title"
    )
    assert status is S.VERIFIED
    assert not retracted


def _lit(item_id, pmid, title):
    return EvidenceItem(
        item_id=item_id, topic_key=TOPIC,
        statement=f'Peer-reviewed literature reports: "{title}"',
        source_id="pubmed-literature", citation=f"PMID {pmid}",
        kind=EvidenceKind.LITERATURE_REPORT, direction=D.SUPPORTS,
        status=S.VERIFIED,
        locator=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


def test_verification_agent_attaches_integrity_warning_to_item():
    out, _ = VerificationAgent(pubmed_transport).verify(
        [_lit("retracted", "777", "Withdrawn Paper Title")]
    )
    assert out[0].status is S.FAILED
    assert "integrity warning" in out[0].notes
    assert "retract" in out[0].notes.lower()


def test_verification_agent_clean_literature_control_untouched():
    item = _lit("clean", "111", "Real Paper Title")
    out, _ = VerificationAgent(pubmed_transport).verify([item])
    assert out[0].status is S.VERIFIED
    assert "integrity" not in out[0].notes


# ---------------------------------------------------------------------
# 3. Inverted index == linear scan
# ---------------------------------------------------------------------


def _brute_force_search(embedder, documents, query, k):
    """The original linear scan, recomputed independently."""
    query_vector = embedder.embed(query)
    scored = []
    for document in documents:
        score = _cosine(query_vector, embedder.embed(document.indexed_text))
        if score > 0.0:
            scored.append((document, score))
    scored.sort(key=lambda pair: (-pair[1], pair[0].doc_id))
    return scored[:k]


def test_inverted_index_matches_linear_scan():
    documents = [
        Document(
            doc_id=f"pmid:{1000 + i}",
            title=f"Study {i} of variant {i % 7} folding",
            text=(
                f"variant {i % 7} folding trafficking corrector "
                f"assay result {i} chromatography buffer"
            ),
            source="pubmed-literature", url=None, fetched_on=FIXED,
        )
        for i in range(60)
    ]
    index = RetrievalIndex()
    index.add_many(documents)
    embedder = LexicalEmbedder()
    for document in documents:
        embedder.observe(document.indexed_text)
    for query in (
        "variant 3 folding corrector",
        "chromatography buffer assay",
        "variant 0 trafficking",
        "nonexistentterm",
    ):
        expected = _brute_force_search(embedder, documents, query, k=5)
        got = index.search(query, k=5)
        assert [d.doc_id for d, _ in got] == [d.doc_id for d, _ in expected]
        for (_, got_score), (_, exp_score) in zip(got, expected):
            assert got_score == pytest.approx(exp_score, abs=1e-9)
