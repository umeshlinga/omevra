"""Tests for literature retrieval (LLM track, Stage A).

All offline: the fixture corpus is hand-written in provome.rag and
the embedder is the lexical baseline.  No network, no model files.
"""

from datetime import date

import pytest

from provome.evidence import VerificationStatus, evaluate_gate
from provome.rag import (
    Document,
    LexicalEmbedder,
    RetrievalIndex,
    _fixture_documents,
    document_pmid,
    evidence_from_retrieval,
)

STAMP = date(2026, 11, 25)


def _index() -> RetrievalIndex:
    index = RetrievalIndex()
    index.add_many(_fixture_documents())
    return index


def test_relevant_document_ranks_first():
    index = _index()
    hits = index.search("CFTR Phe508del folding corrector trafficking", k=3)
    assert hits
    assert hits[0][0].doc_id == "pmid:90000001"
    hits = index.search("venetoclax BCL2 apoptosis inhibitor", k=3)
    assert hits[0][0].doc_id == "pmid:90000003"
    hits = index.search("sickle haemoglobin polymerisation red blood cells", k=3)
    assert hits[0][0].doc_id in {"pmid:90000004", "pmid:90000005"}


def test_scores_are_in_unit_interval_and_descending():
    index = _index()
    hits = index.search("CFTR channel gating potentiator", k=6)
    assert hits
    scores = [score for _, score in hits]
    assert all(0.0 < score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_ties_break_deterministically_by_doc_id():
    text = "identical body text about kinase inhibition and dosing"
    index = RetrievalIndex()
    index.add(
        Document(
            doc_id="pmid:90000020",
            title="Identical study",
            text=text,
            source="pubmed-literature",
            fetched_on=STAMP,
        )
    )
    index.add(
        Document(
            doc_id="pmid:90000010",
            title="Identical study",
            text=text,
            source="pubmed-literature",
            fetched_on=STAMP,
        )
    )
    first = index.search("kinase inhibition dosing", k=2)
    second = index.search("kinase inhibition dosing", k=2)
    assert [doc.doc_id for doc, _ in first] == ["pmid:90000010", "pmid:90000020"]
    assert first[0][1] == first[1][1]  # a genuine tie
    assert [(d.doc_id, s) for d, s in first] == [(d.doc_id, s) for d, s in second]


def test_empty_index_returns_empty_list():
    index = RetrievalIndex()
    assert index.search("anything at all", k=5) == []
    assert (
        evidence_from_retrieval(index, "anything", topic_key="X:y") == []
    )


def test_no_overlap_query_retrieves_nothing():
    index = _index()
    assert index.search("photosynthesis chlorophyll sunlight", k=5) == []


def test_document_validation_and_duplicate_ids():
    with pytest.raises(ValueError):
        Document(doc_id=" ", title="t", text="x", source="pubmed-literature")
    with pytest.raises(ValueError):
        Document(doc_id="d", title="t", text=" ", source="pubmed-literature")
    index = _index()
    with pytest.raises(ValueError):
        index.add(_fixture_documents()[0])


def test_document_pmid_extraction():
    docs = {doc.doc_id: doc for doc in _fixture_documents()}
    assert document_pmid(docs["pmid:90000001"]) == "90000001"
    assert document_pmid(docs["lab-notebook-2026-014"]) is None
    url_only = Document(
        doc_id="internal-copy-7",
        title="A paper",
        text="body text",
        source="pubmed-literature",
        url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        fetched_on=STAMP,
    )
    assert document_pmid(url_only) == "12345678"


def test_bridge_excludes_documents_without_pmid():
    index = _index()
    # The lab notebook is lexically the closest thing to a folding
    # query in parts of the corpus, but it carries no identifier, so
    # the bridge must never turn it into evidence.
    items = evidence_from_retrieval(
        index,
        "CFTR Phe508del folding assay corrector notebook",
        topic_key="CFTR:p.Phe508del",
        k=6,
    )
    assert items
    assert all("notebook" not in item.item_id for item in items)
    assert all(item.citation.startswith("PMID ") for item in items)


def test_bridge_items_are_unverified_without_a_verifier():
    index = _index()
    items = evidence_from_retrieval(
        index,
        "CFTR Phe508del folding corrector trafficking",
        topic_key="CFTR:p.Phe508del",
        k=5,
    )
    assert items
    for item in items:
        # A PMID-shaped id is not verification: nothing was resolved.
        assert item.status is VerificationStatus.UNVERIFIED
        assert item.topic_key == "CFTR:p.Phe508del"
        assert item.source_id == "pubmed-literature"
        assert item.locator and "pubmed" in item.locator
    decision = evaluate_gate(items, "CFTR:p.Phe508del")
    assert decision.verified_count == 0
    assert not decision.passed


def test_bridge_items_verified_only_through_a_verifier():
    class AcceptAll:
        def verify(self, pmid, title):
            return True

    index = _index()
    items = evidence_from_retrieval(
        index,
        "CFTR Phe508del folding corrector trafficking",
        topic_key="CFTR:p.Phe508del",
        k=5,
        verifier=AcceptAll(),
    )
    assert items
    assert all(item.status is VerificationStatus.VERIFIED for item in items)
    # They all share one independence group (primary literature), so
    # the gate still declines on independence.
    decision = evaluate_gate(items, "CFTR:p.Phe508del")
    assert decision.verified_count == len(items)
    assert not decision.passed


def test_neural_style_embedder_plugs_in_without_observe():
    """An Embedder with no observe() hook must work unchanged."""

    class KeywordEmbedder:
        def embed(self, text):
            lowered = text.lower()
            return (
                1.0 if "cftr" in lowered else 0.0,
                1.0 if "bcl2" in lowered else 0.0,
            )

    index = RetrievalIndex(embedder=KeywordEmbedder())
    index.add_many(_fixture_documents())
    hits = index.search("a question about cftr", k=5)
    assert hits
    assert all("cftr" in doc.indexed_text.lower() for doc, _ in hits)
    assert all(score == 1.0 for _, score in hits)


def test_lexical_embedder_is_deterministic_and_normalised():
    embedder = LexicalEmbedder()
    embedder.observe("cftr folding corrector study")
    embedder.observe("bcl2 apoptosis inhibitor study")
    first = embedder.embed("cftr folding")
    second = embedder.embed("cftr folding")
    assert first == second
    assert first  # non-empty once a corpus is observed
    import math

    assert math.isclose(sum(w * w for w in first), 1.0)
