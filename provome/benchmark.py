"""The gate benchmark: does the cite-or-decline gate behave?

Priority item #1 under the Founder's 2026-11-25 order (work one by
one, by importance and efficiency, on the basis of what the whole
industry is chasing): before Provome claims trust, the gate's
behaviour is measured, in the open, against fixed probes.

Each probe is a fixed evidence set with a known correct outcome:

* ``answer`` probes carry at least ``MIN_VERIFIED_ITEMS`` verified
  supporting items from at least ``MIN_INDEPENDENT_SOURCES``
  independent groups - the gate must pass.
* ``decline`` probes fail one or more gate requirements - too few
  verified items, one source group only, unverified items, items
  about a different topic, refuting evidence only, or nothing at
  all - the gate must decline.
* ``disclose`` probes pass but carry verified refuting items -
  the gate must pass AND surface the disagreement, never hide it.

The fixtures are synthetic items built on the real registered
sources; they test gate behaviour, not biomedical facts. The
public benchmark (roadmap Item 38) replays this harness on public
data only, after Quality Division and Founder sign-off.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import (
    EvidenceItem,
    GateDecision,
    SupportDirection,
    VerificationStatus,
    default_source_registry,
    evaluate_gate,
)

__all__ = [
    "BenchmarkReport",
    "Probe",
    "probes",
    "run_benchmark",
]

_V = VerificationStatus.VERIFIED
_U = VerificationStatus.UNVERIFIED
_S = SupportDirection.SUPPORTS
_R = SupportDirection.REFUTES


def _item(item_id, topic, source, direction=_S, status=_V):
    return EvidenceItem(
        item_id=item_id,
        topic_key=topic,
        statement=f"Benchmark fixture statement {item_id}.",
        source_id=source,
        citation=f"Benchmark fixture citation {item_id}.",
        direction=direction,
        status=status,
    )


@dataclass(frozen=True)
class Probe:
    """One fixed gate test: evidence in, expected verdict out."""

    name: str
    category: str  # "answer" | "decline" | "disclose"
    topic_key: str
    items: tuple
    expected_pass: bool


def probes() -> tuple:
    """The fixed probe set. Deterministic; no network, no data."""
    t = "benchmark:topic"
    return (
        # --- must answer -------------------------------------------------
        Probe("three-verified-two-groups", "answer", t, (
            _item("a1", t, "clinvar"), _item("a2", t, "clinvar"),
            _item("a3", t, "pubmed-literature"),
        ), True),
        Probe("four-verified-three-groups", "answer", t, (
            _item("b1", t, "clinvar"), _item("b2", t, "gnomad"),
            _item("b3", t, "pubmed-literature"),
            _item("b4", t, "uniprot"),
        ), True),
        Probe("verified-plus-unverified-noise", "answer", t, (
            _item("c1", t, "clinvar"), _item("c2", t, "gnomad"),
            _item("c3", t, "pubmed-literature"),
            _item("c4", t, "clinvar", status=_U),
            _item("c5", t, "gnomad", status=_U),
        ), True),
        # --- must decline ------------------------------------------------
        Probe("only-two-verified", "decline", t, (
            _item("d1", t, "clinvar"),
            _item("d2", t, "pubmed-literature"),
        ), False),
        Probe("three-verified-one-group", "decline", t, (
            _item("e1", t, "clinvar"), _item("e2", t, "clinvar"),
            _item("e3", t, "clinvar"),
        ), False),
        Probe("unverified-only", "decline", t, (
            _item("f1", t, "clinvar", status=_U),
            _item("f2", t, "gnomad", status=_U),
            _item("f3", t, "pubmed-literature", status=_U),
            _item("f4", t, "uniprot", status=_U),
        ), False),
        Probe("wrong-topic", "decline", t, (
            _item("g1", "benchmark:other", "clinvar"),
            _item("g2", "benchmark:other", "gnomad"),
            _item("g3", "benchmark:other", "pubmed-literature"),
        ), False),
        Probe("refuting-only", "decline", t, (
            _item("h1", t, "clinvar", direction=_R),
            _item("h2", t, "gnomad", direction=_R),
            _item("h3", t, "pubmed-literature", direction=_R),
        ), False),
        Probe("empty", "decline", t, (), False),
        # --- must pass and disclose disagreement -------------------------
        Probe("pass-with-refutation", "disclose", t, (
            _item("i1", t, "clinvar"), _item("i2", t, "gnomad"),
            _item("i3", t, "pubmed-literature"),
            _item("i4", t, "uniprot", direction=_R),
        ), True),
    )


@dataclass(frozen=True)
class BenchmarkReport:
    """The benchmark outcome, stated exactly as measured."""

    total: int
    correct: int
    failures: tuple  # probe names whose verdict was wrong
    disclosure_failures: tuple  # passed but hid refuting evidence

    @property
    def passed(self) -> bool:
        return not self.failures and not self.disclosure_failures

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_benchmark() -> BenchmarkReport:
    """Run every probe through the real gate and score it."""
    registry = default_source_registry()
    failures = []
    disclosure_failures = []
    correct = 0
    all_probes = probes()
    for probe in all_probes:
        decision: GateDecision = evaluate_gate(
            probe.items, probe.topic_key, registry
        )
        if decision.passed == probe.expected_pass:
            correct += 1
        else:
            failures.append(probe.name)
        if probe.category == "disclose":
            if not decision.verified_refuting:
                disclosure_failures.append(probe.name)
    return BenchmarkReport(
        total=len(all_probes),
        correct=correct,
        failures=tuple(failures),
        disclosure_failures=tuple(disclosure_failures),
    )


def _demo() -> None:
    report = run_benchmark()
    print("Provome gate benchmark (internal v1, fixture probes)")
    print("=" * 72)
    print(f"  probes:              {report.total}")
    print(f"  correct verdicts:    {report.correct}")
    print(f"  accuracy:            {report.accuracy:.0%}")
    print(f"  verdict failures:    {list(report.failures)}")
    print(f"  disclosure failures: {list(report.disclosure_failures)}")
    print(f"  RESULT:              {'PASS' if report.passed else 'FAIL'}")
    print(
        "Note: fixture probes test gate behaviour, not biomedical\n"
        "facts. The public benchmark replays this harness on public\n"
        "data after Quality Division and Founder sign-off."
    )


if __name__ == "__main__":
    _demo()
