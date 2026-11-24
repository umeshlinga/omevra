"""Tests for omevra.copilot: answer, decline, gate sensitivity, LLM guard."""

import pytest

from omevra.copilot import (
    Copilot,
    CopilotQuestion,
    DisabledLLMHook,
    OfflineDemoProvider,
    Verdict,
)
from omevra.evidence import (
    EvidenceItem,
    EvidenceKind,
    VerificationStatus,
)

CFTR_Q = "What is CFTR p.Phe508del and does it cause cystic fibrosis?"
BRCA1_Q = "What can you tell me about the BRCA1 c.68_69delAG founder variant?"


def test_demo_provider_recognises_topics():
    provider = OfflineDemoProvider()
    assert provider.topic_for_question(CFTR_Q) == "cftr:p.phe508del"
    assert provider.topic_for_question(BRCA1_Q) == "brca1:c.68_69delag"
    assert provider.topic_for_question("What is the weather?") is None


def test_copilot_answers_cftr_with_citations():
    answer = Copilot().ask(CFTR_Q)
    assert answer.verdict is Verdict.ANSWER
    assert answer.answered
    assert answer.topic_key == "cftr:p.phe508del"
    assert answer.verified_item_count >= 3
    assert answer.independent_source_count >= 2
    assert 0.0 < answer.confidence <= 0.95
    assert answer.declined_reasons == ()
    for claim in answer.claims:
        assert claim.citations, "every claim must carry citations"
        assert claim.evidence_item_ids


def test_copilot_declines_brca1_demo_below_gate():
    # The bundled BRCA1 demo set deliberately holds only two verified
    # items: below the three-item gate, so the Copilot must decline
    # and say exactly that.
    answer = Copilot().ask(BRCA1_Q)
    assert answer.verdict is Verdict.DECLINE
    assert not answer.answered
    assert answer.claims == ()
    assert answer.confidence == 0.0
    assert any("only 2 verified" in r for r in answer.declined_reasons)


def test_copilot_declines_unknown_topic():
    answer = Copilot().ask("Tell me about the ZZZ9 gene and its variants")
    assert answer.verdict is Verdict.DECLINE
    assert answer.topic_key is None
    assert any("no recognised" in r for r in answer.declined_reasons)


class _MutableProvider:
    """Provider whose evidence can be thinned to cross the gate."""

    def __init__(self, items):
        self._items = items

    def topic_for_question(self, question_text):
        return "topic:test"

    def items_for_topic(self, topic_key):
        from dataclasses import replace
        return tuple(replace(i, topic_key=topic_key) for i in self._items)


def _demo_items():
    provider = OfflineDemoProvider()
    return list(provider.items_for_topic("cftr:p.phe508del"))


def test_gate_sensitivity_dropping_items_causes_decline():
    full = Copilot(provider=_MutableProvider(_demo_items())).ask("anything")
    assert full.answered
    thinned = Copilot(provider=_MutableProvider(_demo_items()[:2])).ask("anything")
    assert not thinned.answered
    single_source = Copilot(
        provider=_MutableProvider(
            [i for i in _demo_items() if i.source_id == "clinvar"] * 3
        )
    ).ask("anything")
    assert not single_source.answered


def test_unverified_evidence_does_not_answer():
    items = [
        EvidenceItem(
            item_id=f"u{i}", topic_key="topic:test",
            statement=f"unverified claim {i}", source_id="clinvar",
            citation="some citation", kind=EvidenceKind.OTHER,
            status=VerificationStatus.UNVERIFIED,
        )
        for i in range(5)
    ]
    answer = Copilot(provider=_MutableProvider(items)).ask("anything")
    assert not answer.answered


def test_answer_serialises_to_dict():
    data = Copilot().ask(CFTR_Q).to_dict()
    assert data["verdict"] == "answer"
    assert data["division"] == "human"
    assert isinstance(data["claims"], list) and data["claims"]
    assert data["claims"][0]["citations"]


def test_render_mentions_verdict_and_sources():
    rendered = Copilot().ask(CFTR_Q).render()
    assert "ANSWER" in rendered
    assert "Source:" in rendered
    declined = Copilot().ask(BRCA1_Q).render()
    assert "DECLINE" in declined


def test_llm_hook_is_off_by_default_and_disabled_hook_raises():
    copilot = Copilot()
    assert copilot.use_llm is False
    answer = copilot.ask(CFTR_Q)
    assert answer.narrative is None
    with pytest.raises(RuntimeError):
        DisabledLLMHook().rephrase(answer)


def test_use_llm_without_hook_is_rejected():
    with pytest.raises(ValueError):
        Copilot(use_llm=True)


class _EchoHook:
    def rephrase(self, answer):
        return f"Narrative for {answer.topic_key} with {len(answer.claims)} claims."


def test_enabled_hook_adds_narrative_but_cannot_change_claims():
    copilot = Copilot(llm_hook=_EchoHook(), use_llm=True)
    answer = copilot.ask(CFTR_Q)
    plain = Copilot().ask(CFTR_Q)
    assert answer.narrative is not None
    assert answer.claims == plain.claims
    assert answer.verdict is plain.verdict


def test_question_object_and_string_are_equivalent():
    from_string = Copilot().ask(CFTR_Q)
    from_object = Copilot().ask(CopilotQuestion(text=CFTR_Q))
    assert from_string.verdict is from_object.verdict
    assert from_string.topic_key == from_object.topic_key


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        CopilotQuestion(text="  ")
