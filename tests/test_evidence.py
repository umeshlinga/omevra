"""Tests for omevra.evidence: gate, deduplication, registry."""

import pytest

from omevra.evidence import (
    EvidenceItem,
    EvidenceKind,
    SourceRegistry,
    Source,
    SupportDirection,
    VerificationStatus,
    deduplicate,
    default_source_registry,
    evaluate_gate,
    summarise_items,
)


def item(item_id, source="clinvar", status=VerificationStatus.VERIFIED,
         topic="gene:x", statement=None, direction=SupportDirection.SUPPORTS):
    return EvidenceItem(
        item_id=item_id,
        topic_key=topic,
        statement=statement or f"statement for {item_id}",
        source_id=source,
        citation=f"citation for {item_id}",
        kind=EvidenceKind.LITERATURE_REPORT,
        direction=direction,
        status=status,
    )


def test_gate_passes_with_three_items_two_sources():
    items = [item("a"), item("b"), item("c", source="omim")]
    decision = evaluate_gate(items, "gene:x")
    assert decision.passed
    assert decision.verified_count == 3
    assert decision.independent_source_count == 2


def test_gate_fails_with_only_two_items():
    decision = evaluate_gate([item("a"), item("b", source="omim")], "gene:x")
    assert not decision.passed
    assert any("only 2 verified" in r for r in decision.reasons)


def test_gate_fails_when_all_items_from_one_source_group():
    items = [item("a"), item("b"), item("c")]
    decision = evaluate_gate(items, "gene:x")
    assert not decision.passed
    assert any("independent source group" in r for r in decision.reasons)


def test_unverified_items_do_not_count():
    items = [
        item("a"),
        item("b", source="omim"),
        item("c", source="omim", status=VerificationStatus.UNVERIFIED),
    ]
    decision = evaluate_gate(items, "gene:x")
    assert not decision.passed
    assert decision.verified_count == 2


def test_failed_items_do_not_count_and_warn():
    items = [
        item("a"),
        item("b", source="omim"),
        item("c", source="omim", status=VerificationStatus.FAILED),
    ]
    decision = evaluate_gate(items, "gene:x")
    assert not decision.passed
    assert any("failed verification" in w for w in decision.warnings)


def test_other_topics_are_ignored():
    items = [item("a"), item("b"), item("c", source="omim", topic="gene:y")]
    decision = evaluate_gate(items, "gene:x")
    assert not decision.passed
    assert decision.verified_count == 2


def test_refuting_items_are_surfaced_not_silently_counted():
    items = [
        item("a"),
        item("b", source="omim"),
        item("c", source="omim"),
        item("r", source="omim", direction=SupportDirection.REFUTES),
    ]
    decision = evaluate_gate(items, "gene:x")
    assert decision.passed
    assert len(decision.verified_refuting) == 1
    assert any("refute" in w for w in decision.warnings)


def test_deduplicate_keeps_verified_copy():
    unverified = item("a", status=VerificationStatus.UNVERIFIED)
    verified = item("a", status=VerificationStatus.VERIFIED)
    result = deduplicate([unverified, verified])
    assert len(result) == 1
    assert result[0].status is VerificationStatus.VERIFIED


def test_deduplicate_distinct_statements_survive():
    items = [item("a"), item("b"), item("c")]
    assert len(deduplicate(items)) == 3


def test_registry_rejects_duplicate_source():
    registry = SourceRegistry()
    registry.register(Source("x", "X", "group-x"))
    with pytest.raises(ValueError):
        registry.register(Source("x", "X again", "group-x"))


def test_default_registry_groups_are_sensible():
    registry = default_source_registry()
    assert registry.independence_group_of("clinvar") != registry.independence_group_of("omim")
    assert "clinvar" in registry
    assert len(registry) >= 5


def test_unregistered_source_warns_and_forms_own_group():
    items = [
        item("a", source="mystery-db"),
        item("b", source="mystery-db"),
        item("c", source="omim"),
    ]
    decision = evaluate_gate(items, "gene:x")
    assert any("unregistered source" in w for w in decision.warnings)


def test_summarise_items_counts_by_status():
    items = [
        item("a"),
        item("b", status=VerificationStatus.UNVERIFIED),
        item("c", status=VerificationStatus.FAILED),
    ]
    counts = summarise_items(items)
    assert counts["verified"] == 1
    assert counts["unverified"] == 1
    assert counts["failed"] == 1


def test_item_requires_citation_and_statement():
    with pytest.raises(ValueError):
        EvidenceItem(item_id="x", topic_key="t", statement="s",
                     source_id="clinvar", citation=" ")
    with pytest.raises(ValueError):
        EvidenceItem(item_id="x", topic_key="t", statement=" ",
                     source_id="clinvar", citation="c")
