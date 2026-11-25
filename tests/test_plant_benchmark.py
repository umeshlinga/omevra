"""Tests for the Plant division benchmark harness (v1).

Fixture mode only: the suite never touches the network. The
point of these tests is the harness itself — scoring is against
recorded ground truth, and a provider rigged to answer
everything must fail, loudly.
"""

import pytest

from provome.evidence import (
    EvidenceItem,
    SupportDirection,
    VerificationStatus,
)
from provome.plant import PlantEvidenceProvider
from provome.plant_benchmark import (
    BENCHMARK_VERSION,
    fixture_transport,
    probes,
    run_benchmark,
)

_V = VerificationStatus.VERIFIED
_S = SupportDirection.SUPPORTS


def test_fixture_benchmark_all_probes_correct():
    report = run_benchmark()
    assert report.version == BENCHMARK_VERSION
    assert report.mode == "fixture"
    assert report.total == 21
    assert report.passed
    assert report.accuracy == 1.0
    assert report.failures == ()
    assert all(r.correct for r in report.results)


def test_probe_set_shape_and_ground_truth():
    all_probes = probes()
    assert len(all_probes) >= 18
    assert {p.category for p in all_probes} == {"answer", "decline"}
    assert {p.expected_answer for p in all_probes} == {True, False}
    names = [p.name for p in all_probes]
    assert len(names) == len(set(names))  # names are unique
    for probe in all_probes:
        assert probe.ground_truth.strip()
        assert probe.source_url.startswith("https://")
        assert probe.question.strip()


def test_honest_gap_probes_are_present_and_decline():
    """The benchmark must not hide what v1 cannot do: the two
    famous genes with no Ensembl Plants symbol record (rice SD1,
    maize TB1), the below-gate real gene (FRI), and the
    literature-only trait topics are in the set as declines."""
    by_name = {p.name: p for p in probes()}
    for name in (
        "rice-sd1-no-symbol-record",
        "maize-tb1-no-symbol-record",
        "arabidopsis-fri-below-gate",
        "wheat-drought-tolerance-trait",
        "rice-salt-tolerance-trait",
    ):
        assert name in by_name
        assert by_name[name].expected_answer is False


def test_fixture_mode_is_deterministic():
    first = run_benchmark()
    second = run_benchmark()
    assert first.failures == second.failures
    assert [r.answered for r in first.results] == [
        r.answered for r in second.results
    ]
    assert [r.verified_items for r in first.results] == [
        r.verified_items for r in second.results
    ]


def test_answer_probes_genuinely_pass_the_gate():
    """Every ANSWER verdict in the report is backed by gate
    arithmetic re-checked on the provider's own items — not by
    the surface's say-so."""
    report = run_benchmark()
    for result in report.results:
        if result.expected_answer:
            assert result.gate_passed is True
            assert result.verified_items >= 3
            assert result.source_groups >= 2


class _RiggedProvider(PlantEvidenceProvider):
    """Answers everything: a topic for every question and three
    verified items from two groups for every topic, including
    fabricated genes. A benchmark that scores against ground
    truth must fail this provider on the decline probes."""

    def __init__(self):
        super().__init__(transport=fixture_transport)

    def topic_for_question(self, question_text):
        return "plant:rice:RIGGED"

    def items_for_topic(self, topic_key):
        def item(item_id, source):
            return EvidenceItem(
                item_id=item_id,
                topic_key=topic_key,
                statement=f"Rigged fixture statement {item_id}.",
                source_id=source,
                citation=f"Rigged fixture citation {item_id}.",
                direction=_S,
                status=_V,
            )

        return (
            item("r1", "ensembl-plants"),
            item("r2", "ensembl-plants"),
            item("r3", "pubmed-literature"),
        )


def test_rigged_provider_fails_the_negative_probes():
    report = run_benchmark(
        mode="fixture", provider_factory=_RiggedProvider
    )
    assert not report.passed
    assert report.accuracy < 1.0
    flipped = {
        "wheat-zzzqx9-fabricated",
        "rice-fakgene1-fabricated",
        "tomato-notreal7-fabricated",
        "rice-brca1-human-gene",
        "rice-flc-wrong-species",
        "rice-sd1-no-symbol-record",
    }
    assert flipped <= set(report.failures)


class _SilentProvider(PlantEvidenceProvider):
    """Finds a topic but never any evidence: everything declines."""

    def __init__(self):
        super().__init__(transport=fixture_transport)

    def items_for_topic(self, topic_key):
        return ()


def test_silent_provider_scores_only_the_declines():
    report = run_benchmark(
        mode="fixture", provider_factory=_SilentProvider
    )
    decline_count = sum(1 for p in probes() if not p.expected_answer)
    assert not report.passed
    assert report.correct == decline_count
    assert set(report.failures) == {
        p.name for p in probes() if p.expected_answer
    }


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        run_benchmark(mode="bogus")


def test_fixture_transport_rejects_unrecorded_urls():
    with pytest.raises(AssertionError):
        fixture_transport("https://example.com/not-recorded")
