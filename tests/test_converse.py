"""Tests for the human Copilot layer (Step 4, Items 21 + 23)."""

import pytest

from omevra.converse import Depth, Guide, tool_question_key
from omevra.copilot import Copilot
from omevra.live import LiveEvidenceProvider


def test_tool_questions_route_to_facts():
    assert tool_question_key("What is Provome?") == "what"
    assert tool_question_key("How do you work?") == "how"
    assert tool_question_key("What is the pricing?") == "price"
    assert tool_question_key("Do you cover the animal division?") == "coverage"


def test_biology_questions_never_route_to_tool_facts():
    # A real evidence question that merely mentions a source or a
    # species word must reach the gate, not the fact sheet.
    assert tool_question_key("What does ClinVar say about CFTR p.Phe508del?") is None
    assert tool_question_key("What is known about HBB p.Glu6Val?") is None
    assert tool_question_key("Is this plant variant in rice?") is None


def test_guided_answer_simple_expert_and_transparency():
    guide = Guide()
    simple = guide.ask("What is known about CFTR p.Phe508del?", Depth.SIMPLE)
    assert simple.kind == "answer"
    assert "pathogenic" in simple.body.lower()
    expert = guide.ask("What is known about CFTR p.Phe508del?", Depth.EXPERT)
    assert "Source:" in expert.body
    assert "cite-or-decline gate" in expert.how_i_got_this
    assert "not for clinical use" in expert.limits.lower()


def test_guided_decline_explains_and_asks_for_the_missing_piece():
    guide = Guide()
    guided = guide.ask("Tell me about the weather in Indianapolis today please")
    assert guided.kind == "decline"
    assert "won't guess" in guided.opening
    assert "CFTR" in guided.follow_up  # shows the shape that unblocks


def test_guide_adds_no_claims_beyond_the_gate():
    guide = Guide()
    guided = guide.ask("What is known about CFTR p.Phe508del?", Depth.EXPERT)
    gated_statements = {claim.statement for claim in guided.answer.claims}
    for statement in gated_statements:
        assert statement in guided.body


def test_guide_over_live_provider_reports_provider():
    def empty_transport(url, body=None):
        if "esearch" in url:
            return {"esearchresult": {"idlist": []}}
        return {"resultList": {"result": []}}

    guide = Guide(copilot=Copilot(provider=LiveEvidenceProvider(transport=empty_transport)))
    guided = guide.ask("What is known about XYZ1 p.Ala1Val?")
    assert guided.kind == "decline"
    assert "live" in guided.how_i_got_this


def test_guide_rejects_non_string():
    with pytest.raises(TypeError):
        Guide().ask(None)
