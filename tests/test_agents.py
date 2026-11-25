"""Tests for the agent layer (provome.agents).

The agents are measured, not trusted: these tests pin the real
behaviours the Phase 2 evaluation reported — retrieval completeness,
independent literature re-confirmation (fabricated PMIDs caught,
sticky failure, positive upgrades), corpus bridging with the
confirmer wired, and the composed provider running under the plain
Copilot.  All transports are fixtures; the suite never touches the
network.
"""

from datetime import date

import pytest

from provome.agents import (
    AgentBackedProvider,
    LiteratureAgent,
    RetrievalAgent,
    VerificationAgent,
)
from provome.copilot import Copilot, OfflineDemoProvider, Verdict
from provome.domains import Division
from provome.evidence import (
    EvidenceItem,
    EvidenceKind,
    SupportDirection,
    VerificationStatus as S,
)
from provome.rag import Document, RetrievalIndex

FIXED = date(2026, 11, 25)

_TITLES = {
    "111": "Real Paper Title",
    "222": "CFTR Phe508del folding study",
}


def pubmed_transport(url, body=None):
    pmid = url.rsplit("id=", 1)[1].split("&")[0]
    title = _TITLES.get(pmid)
    if title is None:
        return {"result": {"uids": []}}
    return {"result": {"uids": [pmid], pmid: {"title": title}}}


def lit(item_id, pmid_text, title, status):
    return EvidenceItem(
        item_id=item_id,
        topic_key="gene:p.variant",
        statement=f'Peer-reviewed literature reports: "{title}"',
        source_id="pubmed-literature",
        citation=f"PMID {pmid_text} via Europe PMC (fetched 2026-11-25)",
        kind=EvidenceKind.LITERATURE_REPORT,
        direction=SupportDirection.SUPPORTS,
        status=status,
        locator=f"https://pubmed.ncbi.nlm.nih.gov/{pmid_text}/",
    )


# ---------------------------------------------------------------------
# RetrievalAgent
# ---------------------------------------------------------------------


def test_retrieval_agent_returns_everything_the_provider_holds():
    provider = OfflineDemoProvider()
    agent = RetrievalAgent(provider)
    result = agent.retrieve(
        "What is CFTR p.Phe508del and does it cause cystic fibrosis?"
    )
    assert result.topic_key == "cftr:p.phe508del"
    assert result.items == tuple(provider.items_for_topic("cftr:p.phe508del"))
    assert result.report.items_out == len(result.items)


def test_retrieval_agent_reports_no_topic_honestly():
    agent = RetrievalAgent(OfflineDemoProvider())
    result = agent.retrieve("Tell me about the weather in Indianapolis")
    assert result.topic_key is None
    assert result.items == ()


# ---------------------------------------------------------------------
# VerificationAgent
# ---------------------------------------------------------------------


def test_verification_agent_catches_fabricated_pmid():
    agent = VerificationAgent(pubmed_transport)
    out, _ = agent.verify([lit("fake", "999", "Invented Paper", S.VERIFIED)])
    assert out[0].status is S.FAILED


def test_verification_agent_catches_title_mismatch():
    agent = VerificationAgent(pubmed_transport)
    out, _ = agent.verify(
        [lit("wrong", "111", "A Completely Different Paper", S.VERIFIED)]
    )
    assert out[0].status is S.FAILED


def test_verification_agent_keeps_genuine_and_upgrades_unverified():
    agent = VerificationAgent(pubmed_transport)
    out, _ = agent.verify(
        [
            lit("ok", "111", "Real Paper Title", S.VERIFIED),
            lit("up", "111", "Real Paper Title", S.UNVERIFIED),
        ]
    )
    assert [item.status for item in out] == [S.VERIFIED, S.VERIFIED]


def test_verification_agent_failure_is_sticky():
    agent = VerificationAgent(pubmed_transport)
    out, _ = agent.verify([lit("dead", "111", "Real Paper Title", S.FAILED)])
    assert out[0].status is S.FAILED


def test_verification_agent_unconfirmable_item_is_demoted_not_failed():
    agent = VerificationAgent(pubmed_transport)
    out, _ = agent.verify([lit("ghost", "abc", "Untraceable", S.VERIFIED)])
    assert out[0].status is S.UNVERIFIED


def test_verification_agent_transport_failure_means_unverified():
    def boom(url, body=None):
        raise ConnectionError("down")

    out, _ = VerificationAgent(boom).verify(
        [lit("net", "111", "Real Paper Title", S.VERIFIED)]
    )
    assert out[0].status is S.UNVERIFIED


def test_verification_agent_leaves_non_literature_alone():
    item = EvidenceItem(
        item_id="cv", topic_key="gene:p.variant",
        statement="ClinVar records this variant as pathogenic.",
        source_id="clinvar", citation="ClinVar Variation ID 1",
        kind=EvidenceKind.CURATED_CLASSIFICATION,
        direction=SupportDirection.SUPPORTS, status=S.VERIFIED,
    )
    out, _ = VerificationAgent(pubmed_transport).verify([item])
    assert out[0] is item


def test_verification_agent_requires_a_transport():
    with pytest.raises(ValueError):
        VerificationAgent(None)


# ---------------------------------------------------------------------
# LiteratureAgent
# ---------------------------------------------------------------------


def _corpus():
    index = RetrievalIndex()
    index.add_many(
        [
            Document(
                doc_id="pmid:222",
                title="CFTR Phe508del folding study",
                text="CFTR Phe508del folding trafficking corrector",
                source="pubmed-literature",
                url="https://pubmed.ncbi.nlm.nih.gov/222/",
                fetched_on=FIXED,
            ),
            Document(
                doc_id="pmid:99999991",
                title="CFTR Phe508del fabricated claim",
                text="CFTR Phe508del folding trafficking invented",
                source="pubmed-literature",
                url="https://pubmed.ncbi.nlm.nih.gov/99999991/",
                fetched_on=FIXED,
            ),
            Document(
                doc_id="pmid:99999993",
                title="CFTR Phe508del curated record",
                text="CFTR Phe508del folding trafficking",
                source="clinvar",
                url="https://pubmed.ncbi.nlm.nih.gov/99999993/",
                fetched_on=FIXED,
            ),
        ]
    )
    return index


def test_literature_agent_confirms_real_and_fails_fabricated():
    agent = LiteratureAgent(_corpus(), pubmed_transport, k=5)
    items, report = agent.gather(
        "CFTR Phe508del folding trafficking", "cftr:p.phe508del"
    )
    by_id = {item.item_id: item.status for item in items}
    assert by_id["rag-pmid-222"] is S.VERIFIED
    assert by_id["rag-pmid-99999991"] is S.FAILED
    # The document that merely claims to be ClinVar is excluded by
    # the bridge: a PMID document cannot borrow another source.
    assert "rag-pmid-99999993" not in by_id
    assert report.items_out == len(items)


def test_literature_agent_requires_a_transport():
    with pytest.raises(ValueError):
        LiteratureAgent(_corpus(), None)


# ---------------------------------------------------------------------
# AgentBackedProvider under the plain Copilot
# ---------------------------------------------------------------------


class _NaiveProvider:
    """Stamps VERIFIED without checking (the pre-hardening pattern)."""

    name = "naive-fixture"
    scoped_divisions = frozenset({Division.HUMAN})

    def topic_for_question(self, text):
        return "gene:p.variant"

    def items_for_topic(self, topic_key):
        return (
            EvidenceItem(
                item_id="cv", topic_key=topic_key,
                statement="ClinVar records this variant as pathogenic.",
                source_id="clinvar", citation="ClinVar Variation ID 1",
                kind=EvidenceKind.CURATED_CLASSIFICATION,
                direction=SupportDirection.SUPPORTS, status=S.VERIFIED,
            ),
            lit("fake-1", "999", "Invented Paper", S.VERIFIED),
            lit("fake-2", "998", "Another Invented Paper", S.VERIFIED),
        )


def test_agent_backed_provider_stops_fabricated_literature():
    plain = Copilot(provider=_NaiveProvider()).ask(
        "What is known about GENE p.Variant?"
    )
    assert plain.verdict is Verdict.ANSWER  # the exposure, documented

    backed = Copilot(
        provider=AgentBackedProvider(_NaiveProvider(), pubmed_transport)
    ).ask("What is known about GENE p.Variant?")
    assert backed.verdict is Verdict.DECLINE


class _GenuineProvider:
    """Three registered-source items, the literature one real."""

    name = "genuine-fixture"
    scoped_divisions = frozenset({Division.HUMAN})

    def topic_for_question(self, text):
        return "gene:p.variant"

    def items_for_topic(self, topic_key):
        return (
            EvidenceItem(
                item_id="cv", topic_key=topic_key,
                statement="ClinVar records this variant as pathogenic.",
                source_id="clinvar", citation="ClinVar Variation ID 1",
                kind=EvidenceKind.CURATED_CLASSIFICATION,
                direction=SupportDirection.SUPPORTS, status=S.VERIFIED,
            ),
            EvidenceItem(
                item_id="om", topic_key=topic_key,
                statement="OMIM records the gene-disease relationship.",
                source_id="omim", citation="OMIM 602421.0001",
                kind=EvidenceKind.CURATED_CLASSIFICATION,
                direction=SupportDirection.SUPPORTS, status=S.VERIFIED,
            ),
            lit("lit-1", "111", "Real Paper Title", S.VERIFIED),
        )


def test_agent_backed_provider_passes_genuine_evidence_through():
    provider = AgentBackedProvider(_GenuineProvider(), pubmed_transport)
    answer = Copilot(provider=provider).ask(
        "What is known about GENE p.Variant?"
    )
    assert answer.verdict is Verdict.ANSWER
    assert provider.last_reports  # every agent reported its work


def test_agent_backed_wrapping_the_demo_provider_uses_production_registry():
    # Registry selection in Copilot.ask keys on the provider type:
    # an agent-wrapped demo provider is no longer an
    # OfflineDemoProvider, so the production registry applies and
    # the demo-only sources do not count.  Fail-safe direction:
    # the wrapped stack declines where the bare demo answers.
    provider = AgentBackedProvider(OfflineDemoProvider(), pubmed_transport)
    answer = Copilot(provider=provider).ask(
        "What is CFTR p.Phe508del and does it cause cystic fibrosis?"
    )
    assert answer.verdict is Verdict.DECLINE
