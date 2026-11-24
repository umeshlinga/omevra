"""Tests for omevra.orchestra: one Copilot, an organisation inside.

Covers the end-to-end orchestrated ask, the six QC stages and their
failure paths, cross-worker disagreement and escalation, exclusion
of bad and crashing workers, clean declines when the organisation
cannot staff the work, and a 1,000-worker scale test.
"""

from dataclasses import dataclass

import pytest

from omevra.copilot import OrchestratedCopilot, Verdict
from omevra.evidence import (
    EvidenceItem,
    SupportDirection,
    VerificationStatus,
    default_source_registry,
)
from omevra.orchestra import (
    ClaimDraft,
    Orchestra,
    QCInput,
    WorkerRegistry,
    default_worker_registry,
    run_qc_pipeline,
)
from omevra.teams import TEAM_RARE_DISEASE

CFTR_Q = "What is CFTR p.Phe508del and does it cause cystic fibrosis?"
BRCA1_Q = "What can you tell me about the BRCA1 c.68_69delAG founder variant?"
CDS = "ATGGCTTATGGTTAA"  # M A Y G then stop, as in test_consequence.py


# ---------------------------------------------------------------------
# End-to-end runs
# ---------------------------------------------------------------------


def test_orchestrated_cftr_answers_with_full_qc():
    run = OrchestratedCopilot().ask(CFTR_Q)
    assert run.answer.verdict is Verdict.ANSWER
    assert run.answered
    assert run.team.team_id == TEAM_RARE_DISEASE
    # The whole six-stage QC pipeline passed and is recorded.
    assert run.qc_report.passed
    assert len(run.qc_report.stages) == 6
    assert all(s.passed for s in run.qc_report.stages)
    # The answer carries the QC summary and the audit reference.
    assert run.answer.qc_summary is not None
    assert run.answer.qc_summary["passed"] is True
    assert run.answer.audit_id == run.audit_log.audit_id
    assert run.answer.to_dict()["audit_id"] == run.audit_log.audit_id
    # Workers from more than one capability were assigned.
    kinds = {a["kind"] for a in run.assignments}
    assert "evidence" in kinds and "population" in kinds
    # Every claim carries citations, and the population computation
    # is among them as a numeric, recomputed claim.
    for claim in run.answer.claims:
        assert claim.citations
    assert any(
        "maximum credible allele frequency" in c.statement
        for c in run.answer.claims
    )
    # The audit trail tells the whole story.
    event_types = {e.event_type for e in run.audit_log.events}
    assert {"received", "routed", "assignment", "result",
            "qc_stage", "signoff", "answer"} <= event_types


def test_orchestrated_brca1_still_declines_and_names_failed_stages():
    run = OrchestratedCopilot().ask(BRCA1_Q)
    assert run.answer.verdict is Verdict.DECLINE
    assert not run.answered
    assert run.answer.claims == ()
    assert "evidence_gate" in run.qc_report.failed_stages
    # The decline reasons name the failed QC stages, per stage.
    reasons = " ".join(run.answer.declined_reasons)
    assert "QC stage 1 (evidence_gate) failed" in reasons
    assert "only 2 verified" in reasons
    assert "QC stage 6 (senior_review) failed" in reasons
    assert run.answer.qc_summary["passed"] is False


def test_orchestrated_unknown_topic_declines_cleanly():
    run = OrchestratedCopilot().ask("Tell me about the ZZZ9 gene")
    assert run.answer.verdict is Verdict.DECLINE
    assert len(run.qc_report.stages) == 6
    assert not run.qc_report.passed


def test_engine_task_consequence_runs_through_orchestra():
    run = OrchestratedCopilot().ask(
        CFTR_Q,
        engine_tasks=("consequence",),
        payload={"cds": CDS, "coding_variant": "c.5C>T"},
    )
    assert run.answered
    assert any(
        "missense_variant" in c.statement for c in run.answer.claims
    )
    assert run.qc_report.passed


# ---------------------------------------------------------------------
# QC stage failure paths (unit level)
# ---------------------------------------------------------------------


def _item(item_id, source_id, statement, direction=SupportDirection.SUPPORTS):
    return EvidenceItem(
        item_id=item_id,
        topic_key="gene:x",
        statement=statement,
        source_id=source_id,
        citation=f"citation for {item_id}",
        direction=direction,
        status=VerificationStatus.VERIFIED,
    )


def _qc_input(**overrides):
    base = dict(
        items=(
            _item("i1", "clinvar", "claim one"),
            _item("i2", "omim", "claim two"),
            _item("i3", "pubmed-literature", "claim three"),
        ),
        topic_key="gene:x",
        source_registry=default_source_registry(),
        claims=(
            ClaimDraft(
                statement="claim one",
                citations=("citation for i1",),
                evidence_item_ids=("i1",),
                worker_id="w1, w2",
            ),
        ),
        agreement={"claim one": ("w1", "w2")},
    )
    base.update(overrides)
    return QCInput(**base)


def test_qc_stage1_fails_on_single_source_evidence():
    items = (
        _item("i1", "clinvar", "claim one"),
        _item("i2", "clinvar", "claim two"),
        _item("i3", "clinvar", "claim three"),
    )
    report = run_qc_pipeline(_qc_input(items=items))
    stage = report.stage("evidence_gate")
    assert not stage.passed
    assert "independent source group" in " ".join(stage.details)
    assert not report.passed


def test_qc_stage2_fails_on_claim_without_citation():
    claims = (
        ClaimDraft(
            statement="an uncited claim",
            citations=(),
            evidence_item_ids=("i1",),
            worker_id="w1",
        ),
    )
    report = run_qc_pipeline(
        _qc_input(claims=claims, agreement={"an uncited claim": ("w1", "w2")})
    )
    stage = report.stage("citation_check")
    assert not stage.passed
    assert "no citation" in " ".join(stage.details)


def test_qc_stage3_fails_on_uncheckable_number():
    claims = (
        ClaimDraft(
            statement="the number is 5",
            citations=("some citation",),
            numeric_label="unregistered_metric",
            numeric_value=5.0,
            worker_id="w1",
        ),
    )
    report = run_qc_pipeline(
        _qc_input(claims=claims, agreement={"the number is 5": ("w1", "w2")})
    )
    stage = report.stage("calculation_check")
    assert not stage.passed
    assert "no checker registered" in " ".join(stage.details)


def test_qc_stage3_fails_on_wrong_number_and_passes_on_right_one():
    claims = (
        ClaimDraft(
            statement="the recomputed number",
            citations=("some citation",),
            numeric_label="double",
            numeric_value=4.0,
            worker_id="w1",
        ),
    )
    agreement = {"the recomputed number": ("w1", "w2")}
    wrong = run_qc_pipeline(
        _qc_input(
            claims=claims, agreement=agreement,
            numeric_checkers={"double": lambda ctx: 2.0},
        )
    )
    assert not wrong.stage("calculation_check").passed
    right = run_qc_pipeline(
        _qc_input(
            claims=claims, agreement=agreement,
            numeric_checkers={"double": lambda ctx: 4.0},
        )
    )
    assert right.stage("calculation_check").passed
    assert right.passed


def test_qc_stage4_fails_on_contradictory_claims():
    claims = (
        ClaimDraft(
            statement="the variant is pathogenic",
            citations=("citation a",),
            direction=SupportDirection.SUPPORTS,
            worker_id="w1",
        ),
        ClaimDraft(
            statement="the variant is pathogenic",
            citations=("citation b",),
            direction=SupportDirection.REFUTES,
            worker_id="w2",
        ),
    )
    report = run_qc_pipeline(
        _qc_input(
            claims=claims,
            agreement={"the variant is pathogenic": ("w1", "w2")},
        )
    )
    stage = report.stage("consistency_check")
    assert not stage.passed
    assert "contradict" in " ".join(stage.details)


def test_qc_stage5_fails_with_no_claims_at_all():
    report = run_qc_pipeline(_qc_input(claims=(), agreement={}))
    assert not report.stage("cross_worker_agreement").passed


def test_qc_stage6_fails_when_a_prior_stage_failed():
    report = run_qc_pipeline(_qc_input(claims=(), agreement={}))
    stage = report.stage("senior_review")
    assert not stage.passed
    assert "prior QC stage(s) failed" in " ".join(stage.details)


def test_qc_report_always_has_all_six_stages_in_order():
    report = run_qc_pipeline(_qc_input())
    assert [s.stage for s in report.stages] == [1, 2, 3, 4, 5, 6]
    assert [s.name for s in report.stages] == [
        "evidence_gate", "citation_check", "calculation_check",
        "consistency_check", "cross_worker_agreement", "senior_review",
    ]
    assert report.passed


# ---------------------------------------------------------------------
# Test workers: small protocol implementations for adversarial cases
# ---------------------------------------------------------------------


@dataclass
class StubWorker:
    """A worker that returns one fixed claim, or misbehaves on cue."""

    worker_id: str
    statement: str = "Stub claim S."
    direction: SupportDirection = SupportDirection.SUPPORTS
    citations: tuple = ("Stub citation, test fixture",)
    raises: bool = False
    role_id: str = "senior-researcher"
    team_id: str = "evidence-verification"
    capabilities: tuple = ("evidence",)

    def run(self, task, context):
        from omevra.orchestra import WorkResult

        if self.raises:
            raise RuntimeError("stub worker exploded")
        return WorkResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            role_id=self.role_id,
            team_id=self.team_id,
            claims=(
                ClaimDraft(
                    statement=self.statement,
                    citations=self.citations,
                    evidence_item_ids=("demo-cftr-1",),
                    direction=self.direction,
                    worker_id=self.worker_id,
                ),
            ),
        )


def _orchestra_with(workers):
    return Orchestra(workers=WorkerRegistry(workers))


def test_disagreement_escalates_tiebreaks_and_is_audited():
    workers = [
        StubWorker("evidence-a", direction=SupportDirection.SUPPORTS),
        StubWorker("evidence-b", direction=SupportDirection.REFUTES),
        StubWorker("evidence-c", direction=SupportDirection.SUPPORTS),
    ]
    run = _orchestra_with(workers).ask(CFTR_Q)
    # The tiebreak worker sided with the supporting claim, so the
    # claim survives cross-checking and the answer is issued.
    assert run.answered
    assert any(c.statement == "Stub claim S." for c in run.answer.claims)
    event_types = [e.event_type for e in run.audit_log.events]
    assert "disagreement" in event_types
    assert "escalation" in event_types
    assert "tiebreak" in event_types
    assert run.escalations


def test_bad_worker_result_is_excluded_and_reported():
    workers = [
        StubWorker("evidence-bad", citations=()),  # uncited claim
        StubWorker("evidence-good-a"),
        StubWorker("evidence-good-b"),
    ]
    run = _orchestra_with(workers).ask(CFTR_Q)
    assert run.answered
    exclusions = [
        e for e in run.audit_log.events_of_type("exclusion")
        if e.context.get("worker_id") == "evidence-bad"
    ]
    assert exclusions, "the bad worker must be reported in the audit trail"
    assert "citation" in exclusions[0].message


def test_crashing_worker_is_caught_audited_and_excluded():
    workers = [
        StubWorker("evidence-crash", raises=True),
        StubWorker("evidence-good-a"),
        StubWorker("evidence-good-b"),
    ]
    # Must not raise.
    run = _orchestra_with(workers).ask(CFTR_Q)
    assert run.answered
    errors = [
        e for e in run.audit_log.events_of_type("worker_error")
        if e.context.get("worker_id") == "evidence-crash"
    ]
    assert errors, "the crash must be in the audit trail"
    exclusions = [
        e for e in run.audit_log.events_of_type("exclusion")
        if e.context.get("worker_id") == "evidence-crash"
    ]
    assert exclusions, "the crashed worker's result must be excluded"


def test_reported_failure_worker_is_excluded_and_tiebreak_covers():
    # A worker that returns ok=False (a reported failure, distinct
    # from raising) must be excluded with its error named in the
    # audit trail, and the tiebreak worker covers the gap.  The
    # failing worker's id sorts first so it is definitely assigned.
    @dataclass
    class FailingWorker:
        worker_id: str
        role_id: str = "senior-researcher"
        team_id: str = "evidence-verification"
        capabilities: tuple = ("evidence",)

        def run(self, task, context):
            from omevra.orchestra import WorkResult

            return WorkResult(
                task_id=task.task_id,
                worker_id=self.worker_id,
                role_id=self.role_id,
                team_id=self.team_id,
                ok=False,
                error="source unavailable in fixture",
            )

    workers = [
        FailingWorker("evidence-aaa-fail"),
        StubWorker("evidence-b"),
        StubWorker("evidence-c"),
    ]
    run = _orchestra_with(workers).ask(CFTR_Q)
    assert run.answered
    exclusions = [
        e for e in run.audit_log.events_of_type("exclusion")
        if e.context.get("worker_id") == "evidence-aaa-fail"
    ]
    assert exclusions
    assert "source unavailable in fixture" in exclusions[0].message


def test_empty_worker_registry_declines_cleanly():
    run = Orchestra(workers=WorkerRegistry()).ask(CFTR_Q)
    assert run.answer.verdict is Verdict.DECLINE
    assert not run.answered
    reasons = " ".join(run.answer.declined_reasons)
    assert "QC stage" in reasons
    assert "cross_worker_agreement" in reasons or "senior_review" in reasons


def test_team_with_no_available_workers_declines_cleanly():
    # Workers exist, but none can do evidence work: the routed team
    # cannot staff the question, and the Copilot must decline with
    # the failed stages named rather than crash or improvise.
    workers = [
        StubWorker("population-only", capabilities=("population",)),
    ]
    run = _orchestra_with(workers).ask(CFTR_Q)
    assert run.answer.verdict is Verdict.DECLINE
    assert run.answer.claims == ()
    assert not run.qc_report.passed
    reasons = " ".join(run.answer.declined_reasons)
    assert "QC stage 5 (cross_worker_agreement) failed" in reasons


def test_lone_worker_claim_never_reaches_an_answer():
    # One worker alone, however good its claim, is below the
    # two-worker agreement threshold, and there is no tiebreak pool.
    run = _orchestra_with([StubWorker("evidence-solo")]).ask(CFTR_Q)
    assert run.answer.verdict is Verdict.DECLINE
    assert not any(
        c.statement == "Stub claim S." for c in run.answer.claims
    )


# ---------------------------------------------------------------------
# Scale: 1,000 registered workers
# ---------------------------------------------------------------------


@dataclass
class SyntheticWorker:
    worker_id: str
    role_id: str = "senior-researcher"
    team_id: str = "evidence-verification"
    capabilities: tuple = ("synthetic",)

    def run(self, task, context):
        from omevra.orchestra import WorkResult

        return WorkResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            role_id=self.role_id,
            team_id=self.team_id,
        )


def test_registry_scales_to_a_thousand_workers():
    registry = default_worker_registry()
    base = len(registry)
    for i in range(1000):
        registry.register(SyntheticWorker(worker_id=f"synthetic-{i:04d}"))
    assert len(registry) == base + 1000
    assert len(registry.workers_for("synthetic")) == 1000
    # Routing and assignment still work correctly with the large
    # registry: the orchestrated CFTR run still answers, staffed by
    # the real evidence workers.
    run = Orchestra(workers=registry).ask(CFTR_Q)
    assert run.answered
    assigned = {a["worker_id"] for a in run.assignments}
    assert "evidence-primary" in assigned
    assert not any(w.startswith("synthetic-") for w in assigned)


def test_worker_registry_rejects_duplicate_ids():
    registry = WorkerRegistry([SyntheticWorker("dup")])
    with pytest.raises(ValueError):
        registry.register(SyntheticWorker("dup"))
