"""Step 1 hardening regression tests (2026-11-25).

Each test names the worker finding it locks in:
Worker A (adversarial gate tests, 61 cases), Worker B (fuzz,
18,706 calls), Worker C (orchestra stress + 10,000-worker scale).
"""
import pytest

from provome.calibration import auroc, calibration_table, log_loss
from provome.consequence import translate_codon
from provome.copilot import Claim, Copilot, CopilotQuestion, Verdict
from provome.domains import Division
from provome.evidence import (
    EvidenceItem, VerificationStatus, SupportDirection, evaluate_gate,
)
from provome.population import DiseaseModel, Inheritance, max_credible_allele_count
from provome.variant import (
    VariantParseError, VcfRecord, parse_coding_hgvs, parse_protein_hgvs,
    parse_vcf_line, validate_dna_sequence,
)


def _item(iid, src, stmt, citation=None, direction=SupportDirection.SUPPORTS):
    return EvidenceItem(
        item_id=iid, topic_key="t:topic", statement=stmt, source_id=src,
        citation=citation or f"Citation {iid}", direction=direction,
        status=VerificationStatus.VERIFIED,
    )


class _Provider:
    def __init__(self, items): self._i = items
    def topic_for_question(self, q): return "t:topic"
    def items_for_topic(self, t): return tuple(self._i)


# Worker A Bug 1: fabricated sources must not satisfy the gate
def test_unregistered_sources_do_not_count():
    fake = [_item("f1", "fake-1", "Claim one."), _item("f2", "fake-2", "Claim two."),
            _item("f3", "fake-3", "Claim three.")]
    assert not evaluate_gate(fake, "t:topic").passed
    ans = Copilot(provider=_Provider(fake)).ask("Tell me about t:topic")
    assert ans.verdict is Verdict.DECLINE


# Worker A Bug 2: human evidence never issued in Animal/Plant
@pytest.mark.parametrize("kwargs", [dict(species="Bos taurus"), dict(division=Division.PLANT)])
def test_cross_division_declines(kwargs):
    ans = Copilot().ask(CopilotQuestion(
        text="What is CFTR p.Phe508del and does it cause cystic fibrosis?", **kwargs))
    assert ans.verdict is Verdict.DECLINE
    assert ans.claims == ()


# Worker A Bug 3: one citation restated 3x is one evidence line
def test_paraphrase_duplicates_do_not_inflate():
    items = [
        _item("a", "clinvar", "Statement one.", citation="Shared Citation X"),
        _item("b", "omim", "Statement one!", citation="Shared Citation X"),
        _item("c", "clinvar", "A statement one.", citation="Shared Citation X"),
    ]
    assert not evaluate_gate(items, "t:topic").passed


# Worker A Bug 5: refuting >= supporting blocks the answer
def test_majority_refuting_blocks():
    items = [_item(f"s{i}", "clinvar" if i % 2 else "omim", f"Support {i}.") for i in range(3)]
    items += [_item(f"r{i}", "clinvar" if i % 2 else "omim", f"Refute {i}.",
                    citation=f"RefCite {i}", direction=SupportDirection.REFUTES) for i in range(5)]
    assert not evaluate_gate(items, "t:topic").passed


# Worker A Bugs 4/6/7: validation invariants
def test_claim_rejects_empty_citation_string():
    with pytest.raises(ValueError):
        Claim(statement="A claim.", citations=("",), evidence_item_ids=("e1",))


def test_evidence_item_rejects_whitespace_id():
    with pytest.raises(ValueError):
        _item("   ", "clinvar", "A statement.")


def test_ask_rejects_non_string_input():
    with pytest.raises(TypeError):
        Copilot().ask(None)
    with pytest.raises(TypeError):
        Copilot().ask(123)


# Worker A Bug 8: alias inside a larger token must not match
def test_embedded_alias_token_does_not_match():
    ans = Copilot().ask("Tell me about xcftrx please")
    assert ans.verdict is Verdict.DECLINE


# Worker B: wrong-type inputs raise the documented clean errors
def test_variant_wrong_types():
    for fn, bad in [(parse_coding_hgvs, 123), (parse_coding_hgvs, b"c.76A>C"),
                    (parse_protein_hgvs, 42), (parse_vcf_line, 7),
                    (validate_dna_sequence, 123), (translate_codon, 123)]:
        with pytest.raises((VariantParseError, ValueError)):
            fn(bad)
    with pytest.raises(VariantParseError):
        VcfRecord("1", "100", "A", "G")


def test_population_wrong_types():
    model = DiseaseModel(prevalence=0.001, inheritance=Inheritance.AUTOSOMAL_RECESSIVE)
    with pytest.raises(ValueError):
        max_credible_allele_count(model, "x")
    with pytest.raises(ValueError):
        DiseaseModel(prevalence="0.1", inheritance=Inheritance.AUTOSOMAL_RECESSIVE)


def test_calibration_wrong_types():
    with pytest.raises(ValueError):
        auroc([1, 0], [None, 0.5])
    with pytest.raises(ValueError):
        calibration_table([1, 0], [0.9, 0.1], n_bins=2.5)
    with pytest.raises(ValueError):
        calibration_table([1, 0], [0.9, 0.1], n_bins=1_000_000)
    with pytest.raises(ValueError):
        log_loss([1, 0], [0.9, 0.1], epsilon=None)
