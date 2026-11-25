"""The orchestra: one Copilot outside, an organisation inside.

From the outside, Provome has exactly one Copilot.  This module is
what is inside it: a working organisation of senior specialist
workers, arranged like a company and run like a production system.

The flow of every question:

1. **Intake and routing.**  The Chief Copilot (the orchestrator in
   this module) classifies the question by division and routes it to
   the owning senior team (:mod:`provome.teams`).
2. **Assignment.**  The Chief assigns sub-tasks to senior workers
   (:mod:`provome.roles`) drawn from a worker registry.  Workers are
   lightweight records plus a callable, held in dictionaries and
   capability indexes, so the registry is designed to scale to
   thousands of registered specialist workers without changing the
   orchestration logic.  The built-in workers are deterministic and
   offline: they wrap the real engines of this package (evidence,
   population, consequence, ACMG, calibration, phenotype) and compute
   real results - no language model, no network.
3. **Cross-checking.**  Every key claim must be produced
   independently by at least two workers.  A claim produced by only
   one worker, or produced in conflicting directions by two workers,
   is a disagreement: it is recorded, escalated to the team lead, and
   the lead orders a further worker (a tiebreak) or a re-run.  A
   worker whose result fails screening - a claim without a citation,
   a crashed run, a failed result - is reported in the audit trail by
   worker id with the failure named, and its result is excluded.  A
   bad worker therefore cannot poison an answer quietly, and cannot
   crash the run either: worker exceptions are caught, audited, and
   excluded.
4. **The QC pipeline.**  Collected work passes six explicit QC
   stages, in order, each a real check function, each recorded:
   (1) evidence gate, (2) citation check, (3) calculation check,
   (4) consistency check, (5) cross-worker agreement check,
   (6) senior review (team lead sign-off).  The final answer is
   produced only if every stage passes; otherwise the Copilot
   declines, with every failed stage named, on top of the existing
   cite-or-decline rule.
5. **Audit.**  Every assignment, result, check, disagreement,
   escalation, exclusion, and sign-off is appended to an audit log
   with a timestamp and ids.  The issued answer carries the QC
   report summary and the audit id, so any answer can be traced back
   through the organisation that produced it.

Honest scale note: the registry and orchestration here are designed
to scale to thousands of registered specialist workers; the built-in
offline workers wrap the real engines in this package, and live
model-backed workers attach through the same worker protocol.  This
module does not claim that thousands of workers are running today -
it claims the organisation is built so they can be registered,
assigned, cross-checked, and audited when they are.

Standard library only.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from .acmg import (
    classify,
    classify_by_points,
    classify_by_richards_rules,
    total_points,
)
from .calibration import auroc, calibration_table
from .calibration import evaluate as calibration_evaluate
from .consequence import CODON_TABLE, apply_coding_variant, call_consequence
from .copilot import (
    Claim,
    CopilotAnswer,
    CopilotQuestion,
    EvidenceProvider,
    OfflineDemoProvider,
    Verdict,
    _confidence,
)
from .domains import Division
from .evidence import (
    EvidenceItem,
    SourceRegistry,
    SupportDirection,
    default_source_registry,
    demo_source_registry,
    deduplicate,
    evaluate_gate,
)
from .phenotype import best_match_average, compare_term_sets, example_ontology
from .population import DiseaseModel, Inheritance, evaluate_frequency, max_credible_af
from .roles import ROLE_SENIOR_QC_LEAD, RoleRegistry, default_role_registry
from .teams import Team, TeamRegistry, default_team_registry
from .variant import parse_coding_hgvs

__all__ = [
    "AuditEvent",
    "AuditLog",
    "ClaimDraft",
    "WorkTask",
    "WorkResult",
    "WorkContext",
    "Worker",
    "WorkerRegistry",
    "QCStageResult",
    "QCReport",
    "QCInput",
    "qc_evidence_gate",
    "qc_citation_check",
    "qc_calculation_check",
    "qc_consistency_check",
    "qc_agreement_check",
    "qc_senior_review",
    "run_qc_pipeline",
    "QC_STAGE_NAMES",
    "default_worker_registry",
    "Orchestra",
    "OrchestratedAnswer",
    "main",
]


# ---------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEvent:
    """One entry in the audit log.

    Attributes:
        seq: Monotonic sequence number within the log (1-based).
        timestamp: ISO 8601 UTC timestamp of the event.
        event_type: One of ``received``, ``routed``, ``assignment``,
            ``result``, ``worker_error``, ``exclusion``,
            ``disagreement``, ``escalation``, ``tiebreak``,
            ``qc_stage``, ``signoff``, ``answer``, ``decline``.
        actor: Who acted: a worker id, a role id, or ``"chief-copilot"``.
        message: What happened, in plain language.
        context: Ids and details (task id, team id, claim key, stage
            name) as plain JSON-compatible data.
    """

    seq: int
    timestamp: str
    event_type: str
    actor: str
    message: str
    context: dict = field(default_factory=dict)


class AuditLog:
    """Append-only audit trail for one orchestrated question.

    Every assignment, result, check, disagreement, escalation, and
    exclusion in a run is appended here.  The log is never rewritten:
    a corrected mistake appears as a later event, which is what makes
    the trail worth trusting.
    """

    def __init__(self, audit_id: Optional[str] = None) -> None:
        self.audit_id = audit_id or f"audit-{uuid.uuid4().hex[:12]}"
        self._events: list[AuditEvent] = []

    def record(
        self,
        event_type: str,
        actor: str,
        message: str,
        **context: object,
    ) -> AuditEvent:
        """Append one event and return it."""
        event = AuditEvent(
            seq=len(self._events) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            actor=actor,
            message=message,
            context=dict(context),
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    def events_of_type(self, event_type: str) -> tuple[AuditEvent, ...]:
        return tuple(e for e in self._events if e.event_type == event_type)

    def __len__(self) -> int:
        return len(self._events)

    def to_dict(self) -> dict:
        return {
            "audit_id": self.audit_id,
            "events": [
                {
                    "seq": e.seq,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "actor": e.actor,
                    "message": e.message,
                    "context": e.context,
                }
                for e in self._events
            ],
        }


# ---------------------------------------------------------------------
# Work: tasks, claims, results, context
# ---------------------------------------------------------------------


def _claim_key(statement: str) -> str:
    """Normalised identity of a claim, for agreement grouping."""
    return " ".join(statement.lower().split())


@dataclass(frozen=True)
class ClaimDraft:
    """One claim produced by a worker, before QC.

    Unlike :class:`provome.copilot.Claim`, a draft is allowed to be
    defective - it may lack citations, or carry an unverified number.
    Defects are exactly what screening and the QC pipeline exist to
    catch; a draft that survived screening and QC is promoted to a
    ``Claim`` in the final answer, and a defective one is excluded
    with the failure audited against its worker.

    Attributes:
        statement: The claim, in plain language.
        citations: Citations backing the claim.  May be empty in a
            draft; screening excludes such claims' results.
        evidence_item_ids: Evidence items (or computation provenance
            ids of the form ``"computed:<label>"``) behind the claim.
        direction: Whether the claim supports or refutes the topic's
            main claim.
        numeric_label: When the claim carries a number that QC must
            recompute, the label of the registered checker for it.
        numeric_value: The claimed number, when ``numeric_label``
            is set.
        worker_id: The worker that produced this claim.
    """

    statement: str
    citations: tuple[str, ...] = ()
    evidence_item_ids: tuple[str, ...] = ()
    direction: SupportDirection = SupportDirection.SUPPORTS
    numeric_label: Optional[str] = None
    numeric_value: Optional[float] = None
    worker_id: str = ""

    @property
    def claim_key(self) -> str:
        return _claim_key(self.statement)


@dataclass(frozen=True)
class WorkTask:
    """One sub-task assigned by the Chief Copilot to a worker.

    Attributes:
        task_id: Stable id within the run, for example ``"task-3"``.
        kind: The capability the task needs: ``"evidence"``,
            ``"population"``, ``"consequence"``, ``"acmg"``,
            ``"calibration"``, or ``"phenotype"`` for the built-in
            workers; host applications may define their own kinds.
        topic_key: The topic the task is about, when one is known.
        team_id: The owning team the task was assigned under.
        purpose: Why the task was assigned: ``"primary"``,
            ``"crosscheck"``, or ``"tiebreak"``.
    """

    task_id: str
    kind: str
    topic_key: Optional[str]
    team_id: str
    purpose: str = "primary"


@dataclass(frozen=True)
class WorkResult:
    """Structured result returned by one worker for one task.

    Attributes:
        task_id: The task this result answers.
        worker_id: The worker that produced it.
        role_id: The senior role the worker acted in.
        team_id: The worker's home team.
        claims: Claims the worker produced.
        evidence_items: Evidence items the worker used or gathered.
        notes: Plain language notes from the worker.
        ok: False when the worker itself reports failure.  A worker
            that cannot do its task must say so here (or raise, which
            the orchestra converts into a failed result); it must
            never return a plausible-looking empty success.
        error: The failure, when ``ok`` is False.
    """

    task_id: str
    worker_id: str
    role_id: str
    team_id: str
    claims: tuple[ClaimDraft, ...] = ()
    evidence_items: tuple[EvidenceItem, ...] = ()
    notes: tuple[str, ...] = ()
    ok: bool = True
    error: Optional[str] = None


@dataclass(frozen=True)
class WorkContext:
    """Everything a worker may need for one question.

    Attributes:
        question: The question being answered.
        topic_key: The recognised topic, or ``None``.
        items: Evidence items gathered for the topic.
        source_registry: Registry for source independence groups.
        team: The owning team the question was routed to.
        payload: Host-supplied computation inputs, keyed by name:
            ``cds`` and ``coding_variant`` for consequence work,
            ``acmg_assessments`` for classification work, ``labels``
            and ``scores`` for calibration work, ``term_set_a`` and
            ``term_set_b`` for phenotype work, ``disease_model`` for
            population work.
    """

    question: CopilotQuestion
    topic_key: Optional[str]
    items: tuple[EvidenceItem, ...]
    source_registry: SourceRegistry
    team: Team
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# Workers and the worker registry
# ---------------------------------------------------------------------


@runtime_checkable
class Worker(Protocol):
    """Protocol for a specialist worker in the organisation.

    A worker is a lightweight record - an id, the senior role it acts
    in, its home team, and the capabilities it can be assigned - plus
    one callable.  That shape is deliberate: a host application can
    register thousands of workers (engine wrappers, curated-data
    workers, or live model-backed workers) without the orchestra
    needing to know anything about them beyond this protocol.
    """

    worker_id: str
    role_id: str
    team_id: str
    capabilities: tuple[str, ...]

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        """Do one assigned task and return a structured result."""
        ...


class WorkerRegistry:
    """Registry of workers, indexed by capability.

    Workers are held in a dictionary by id and in a capability index
    (capability -> sorted worker ids), so assignment stays fast and
    deterministic when the registry holds thousands of workers:
    looking up the workers for a capability never scans the whole
    registry, and ties are always broken the same way, by worker id.
    """

    def __init__(self, workers: Sequence[Worker] = ()) -> None:
        self._workers: dict[str, Worker] = {}
        self._by_capability: dict[str, list[str]] = {}
        for worker in workers:
            self.register(worker)

    def register(self, worker: Worker) -> None:
        """Add a worker; raises ``ValueError`` on a duplicate id."""
        if worker.worker_id in self._workers:
            raise ValueError(f"duplicate worker_id: {worker.worker_id!r}")
        self._workers[worker.worker_id] = worker
        for capability in worker.capabilities:
            ids = self._by_capability.setdefault(capability, [])
            ids.append(worker.worker_id)
            ids.sort()

    def get(self, worker_id: str) -> Optional[Worker]:
        return self._workers.get(worker_id)

    def __contains__(self, worker_id: object) -> bool:
        return worker_id in self._workers

    def __len__(self) -> int:
        return len(self._workers)

    def all(self) -> tuple[Worker, ...]:
        return tuple(self._workers.values())

    def workers_for(self, capability: str) -> tuple[Worker, ...]:
        """Workers offering a capability, in deterministic id order."""
        return tuple(
            self._workers[worker_id]
            for worker_id in self._by_capability.get(capability, [])
        )

    def capabilities(self) -> tuple[str, ...]:
        """Every capability offered by at least one registered worker."""
        return tuple(sorted(self._by_capability))


# ---------------------------------------------------------------------
# The QC pipeline: six explicit stages, in order
# ---------------------------------------------------------------------

#: The six QC stage names, in pipeline order.
QC_STAGE_NAMES = (
    "evidence_gate",
    "citation_check",
    "calculation_check",
    "consistency_check",
    "cross_worker_agreement",
    "senior_review",
)


@dataclass(frozen=True)
class QCStageResult:
    """Outcome of one QC stage.

    Attributes:
        stage: Stage number, 1 to 6.
        name: Stage name (see :data:`QC_STAGE_NAMES`).
        passed: True when the stage's check passed.
        details: What the check found, including the exact failure
            when it failed.  A stage never fails silently.
    """

    stage: int
    name: str
    passed: bool
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class QCReport:
    """The full QC record for one orchestrated answer.

    Attributes:
        stages: Every stage result, in pipeline order.  Always all
            six stages: a stage that could not run because an earlier
            stage failed is recorded as failed with the reason, never
            omitted, so a report can never look cleaner than the run
            was.
        audit_id: Id of the audit log this report belongs to.
    """

    stages: tuple[QCStageResult, ...]
    audit_id: Optional[str] = None

    @property
    def passed(self) -> bool:
        return all(stage.passed for stage in self.stages)

    @property
    def failed_stages(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if not stage.passed)

    def stage(self, name: str) -> QCStageResult:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"no QC stage named {name!r}")

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "audit_id": self.audit_id,
            "failed_stages": list(self.failed_stages),
            "stages": [
                {
                    "stage": s.stage,
                    "name": s.name,
                    "passed": s.passed,
                    "details": list(s.details),
                }
                for s in self.stages
            ],
        }


@dataclass(frozen=True)
class QCInput:
    """Everything the QC stages check, prepared by the orchestra.

    Attributes:
        items: All evidence items gathered for the topic.
        topic_key: The topic the evidence is about.
        source_registry: Registry for independence groups.
        claims: The accepted claims after screening and cross-worker
            resolution (merged claim drafts).
        agreement: Claim key -> ids of the distinct workers that
            independently produced that claim.
        numeric_checkers: Numeric label -> recompute function.  A
            checker receives ``context`` and returns the independently
            recomputed value, or ``None`` when it cannot recompute.
        context: Inputs the checkers recompute from (disease model,
            labels and scores, and similar).
        min_agreement: Distinct workers that must agree on each key
            claim.  Two by default: no claim in an Provome answer rests
            on one worker alone.
        unresolved_disagreements: Disagreements the team lead could
            not resolve.  Any nonzero count fails senior review.
    """

    items: tuple[EvidenceItem, ...]
    topic_key: Optional[str]
    source_registry: SourceRegistry
    claims: tuple[ClaimDraft, ...] = ()
    agreement: dict = field(default_factory=dict)
    numeric_checkers: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    min_agreement: int = 2
    unresolved_disagreements: int = 0


def qc_evidence_gate(qc: QCInput) -> QCStageResult:
    """Stage 1: the cite-or-decline evidence gate.

    Reuses :func:`provome.evidence.evaluate_gate` unchanged: at least
    three verified supporting items from at least two independent
    source groups, or this stage fails with the gate's own reasons.
    """
    if qc.topic_key is None:
        return QCStageResult(
            stage=1,
            name="evidence_gate",
            passed=False,
            details=(
                "no recognised topic, so no evidence could be "
                "gathered and the gate cannot be satisfied",
            ),
        )
    gate = evaluate_gate(list(qc.items), qc.topic_key, registry=qc.source_registry)
    return QCStageResult(
        stage=1,
        name="evidence_gate",
        passed=gate.passed,
        details=gate.reasons,
    )


def qc_citation_check(qc: QCInput) -> QCStageResult:
    """Stage 2: every accepted claim carries at least one citation."""
    offenders = [c for c in qc.claims if not c.citations]
    if offenders:
        return QCStageResult(
            stage=2,
            name="citation_check",
            passed=False,
            details=tuple(
                f"claim by worker {c.worker_id} has no citation: "
                f"{c.statement[:80]}"
                for c in offenders
            ),
        )
    return QCStageResult(
        stage=2,
        name="citation_check",
        passed=True,
        details=(
            f"all {len(qc.claims)} accepted claim(s) carry at least "
            "one citation",
        ),
    )


def qc_calculation_check(qc: QCInput) -> QCStageResult:
    """Stage 3: numeric claims are recomputed or fail.

    A claim carrying a ``numeric_label`` must have a registered
    checker, and the checker's independently recomputed value must
    match the claimed value.  A numeric claim with no registered
    checker fails this stage: an uncheckable number is not allowed
    to ride along unchecked.
    """
    numeric = [c for c in qc.claims if c.numeric_label is not None]
    if not numeric:
        return QCStageResult(
            stage=3,
            name="calculation_check",
            passed=True,
            details=("no numeric claims required recomputation",),
        )
    failures: list[str] = []
    for claim in numeric:
        checker = qc.numeric_checkers.get(claim.numeric_label or "")
        if checker is None:
            failures.append(
                f"no checker registered for numeric claim "
                f"{claim.numeric_label!r} by worker {claim.worker_id}"
            )
            continue
        expected = checker(qc.context)
        if expected is None:
            failures.append(
                f"checker for {claim.numeric_label!r} could not "
                "recompute the claimed value from the stated inputs"
            )
            continue
        if claim.numeric_value is None or not math.isclose(
            claim.numeric_value, expected, rel_tol=1e-9, abs_tol=1e-12
        ):
            failures.append(
                f"numeric claim {claim.numeric_label!r} by worker "
                f"{claim.worker_id} states {claim.numeric_value!r}; "
                f"independent recomputation gives {expected!r}"
            )
    if failures:
        return QCStageResult(
            stage=3, name="calculation_check", passed=False,
            details=tuple(failures),
        )
    return QCStageResult(
        stage=3,
        name="calculation_check",
        passed=True,
        details=(
            f"{len(numeric)} numeric claim(s) recomputed and matched",
        ),
    )


def qc_consistency_check(qc: QCInput) -> QCStageResult:
    """Stage 4: no two accepted claims contradict on direction.

    Claims are grouped by their normalised statement; a group holding
    both a supporting and a refuting version of the same statement is
    a contradiction, and contradictions are never issued as answers.
    """
    directions: dict[str, set] = {}
    for claim in qc.claims:
        directions.setdefault(claim.claim_key, set()).add(claim.direction)
    conflicts = sorted(
        key for key, dirs in directions.items() if len(dirs) > 1
    )
    if conflicts:
        return QCStageResult(
            stage=4,
            name="consistency_check",
            passed=False,
            details=tuple(
                f"accepted claims contradict on direction for: {key[:80]}"
                for key in conflicts
            ),
        )
    return QCStageResult(
        stage=4,
        name="consistency_check",
        passed=True,
        details=("no accepted claims contradict on direction",),
    )


def qc_agreement_check(qc: QCInput) -> QCStageResult:
    """Stage 5: every key claim has enough independent workers.

    Also fails when there are no accepted claims at all: an answer
    with no claims is not an answer, whatever the other stages say.
    """
    if not qc.claims:
        return QCStageResult(
            stage=5,
            name="cross_worker_agreement",
            passed=False,
            details=(
                "no accepted claims were produced by any worker; "
                "there is nothing for workers to agree on",
            ),
        )
    failures: list[str] = []
    for claim in qc.claims:
        workers = set(qc.agreement.get(claim.claim_key, ()))
        if len(workers) < qc.min_agreement:
            failures.append(
                f"claim agreed by {len(workers)} worker(s), needs "
                f"{qc.min_agreement}: {claim.statement[:80]}"
            )
    if failures:
        return QCStageResult(
            stage=5, name="cross_worker_agreement", passed=False,
            details=tuple(failures),
        )
    return QCStageResult(
        stage=5,
        name="cross_worker_agreement",
        passed=True,
        details=(
            f"every key claim was independently produced by at "
            f"least {qc.min_agreement} workers",
        ),
    )


def qc_senior_review(
    qc: QCInput, prior: Sequence[QCStageResult]
) -> QCStageResult:
    """Stage 6: senior review - the team lead sign-off rule.

    Sign-off is a rule, not a mood: the lead signs only when every
    prior stage passed, the agreement threshold is met, and no
    disagreement is left unresolved.  Anything else is a refusal
    with the blockers named.
    """
    blockers: list[str] = []
    failed_prior = [s.name for s in prior if not s.passed]
    if failed_prior:
        blockers.append(
            "prior QC stage(s) failed: " + ", ".join(failed_prior)
        )
    if qc.unresolved_disagreements:
        blockers.append(
            f"{qc.unresolved_disagreements} disagreement(s) remain "
            "unresolved after escalation"
        )
    if blockers:
        return QCStageResult(
            stage=6, name="senior_review", passed=False,
            details=tuple(blockers),
        )
    return QCStageResult(
        stage=6,
        name="senior_review",
        passed=True,
        details=(
            "senior review sign-off granted: all prior QC stages "
            "passed, the cross-worker agreement threshold is met, "
            "and no disagreement is unresolved",
        ),
    )


def run_qc_pipeline(qc: QCInput) -> QCReport:
    """Run all six QC stages in order and return the full report.

    Every stage runs and is recorded even when an earlier stage
    failed: later stages check what they can (a citation defect is
    worth recording even when the evidence gate already failed), and
    the report must show the whole picture, not stop at the first
    red light.
    """
    stages: list[QCStageResult] = [
        qc_evidence_gate(qc),
        qc_citation_check(qc),
        qc_calculation_check(qc),
        qc_consistency_check(qc),
        qc_agreement_check(qc),
    ]
    stages.append(qc_senior_review(qc, stages))
    return QCReport(stages=tuple(stages))


def _not_run_report(reason: str, audit_id: Optional[str] = None) -> QCReport:
    """A full six-stage report for a run that stopped before QC.

    Used when a question cannot even be gathered (no recognised
    topic): stage 1 fails with the reason, and stages 2 to 6 are
    recorded as failed because they never ran - a report must never
    imply checks happened that did not.
    """
    stages = [
        QCStageResult(
            stage=1, name="evidence_gate", passed=False, details=(reason,)
        )
    ]
    for number, name in enumerate(QC_STAGE_NAMES[1:], start=2):
        stages.append(
            QCStageResult(
                stage=number,
                name=name,
                passed=False,
                details=(f"not run: {reason}",),
            )
        )
    return QCReport(stages=tuple(stages), audit_id=audit_id)


# ---------------------------------------------------------------------
# Built-in workers: deterministic, offline, wrapping the real engines
# ---------------------------------------------------------------------

from .roles import (  # noqa: E402 - grouped with the workers they staff
    ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
    ROLE_SENIOR_DATA_ANALYST,
    ROLE_SENIOR_EVIDENCE_REVIEWER,
    ROLE_SENIOR_RESEARCHER,
    ROLE_SENIOR_SCIENTIST,
)
from .teams import (  # noqa: E402
    TEAM_CALIBRATION_BENCHMARKS,
    TEAM_DNA_VARIANT,
    TEAM_EVIDENCE_VERIFICATION,
    TEAM_RARE_DISEASE,
)


@dataclass
class BuiltinWorker:
    """Base record for the built-in workers.

    Concrete workers override :meth:`run`.  The record fields are
    exactly the worker protocol: id, senior role, home team,
    capabilities.
    """

    worker_id: str
    role_id: str
    team_id: str
    capabilities: tuple[str, ...]

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        raise NotImplementedError

    def _result(
        self,
        task: WorkTask,
        claims: Sequence[ClaimDraft] = (),
        *,
        items: Sequence[EvidenceItem] = (),
        notes: Sequence[str] = (),
        ok: bool = True,
        error: Optional[str] = None,
    ) -> WorkResult:
        return WorkResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            role_id=self.role_id,
            team_id=self.team_id,
            claims=tuple(
                ClaimDraft(
                    statement=c.statement,
                    citations=c.citations,
                    evidence_item_ids=c.evidence_item_ids,
                    direction=c.direction,
                    numeric_label=c.numeric_label,
                    numeric_value=c.numeric_value,
                    worker_id=self.worker_id,
                )
                for c in claims
            ),
            evidence_items=tuple(items),
            notes=tuple(notes),
            ok=ok,
            error=error,
        )

    def _failed(self, task: WorkTask, error: str) -> WorkResult:
        return self._result(task, ok=False, error=error)


@dataclass
class EvidenceWorker(BuiltinWorker):
    """Builds claims from the verified supporting evidence items.

    Three strategies exist so three evidence workers can work the
    same topic by genuinely different routes and still be expected
    to agree claim for claim:

    * ``"gate"``: claims from the gate decision's verified
      supporting items (the route the plain Copilot takes).
    * ``"manual"``: an independent filter - deduplicate, keep
      verified supporting items for the topic, sort by item id.
    * ``"recount"``: a second independent filter in original order,
      used as the tiebreak worker.
    """

    strategy: str = "gate"

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        if context.topic_key is None:
            return self._failed(task, "no topic recognised; nothing to gather")
        items = list(context.items)
        if self.strategy == "gate":
            gate = evaluate_gate(
                items, context.topic_key, registry=context.source_registry
            )
            supporting = list(gate.verified_supporting)
        elif self.strategy == "manual":
            supporting = sorted(
                (
                    item
                    for item in deduplicate(items)
                    if item.topic_key == context.topic_key
                    and item.is_verified
                    and item.direction is SupportDirection.SUPPORTS
                ),
                key=lambda item: item.item_id,
            )
        elif self.strategy == "recount":
            seen: set = set()
            supporting = []
            for item in items:
                if item.item_id in seen:
                    continue
                seen.add(item.item_id)
                if (
                    item.topic_key == context.topic_key
                    and item.is_verified
                    and item.direction is SupportDirection.SUPPORTS
                ):
                    supporting.append(item)
        else:  # pragma: no cover - guarded by registry construction
            return self._failed(task, f"unknown strategy {self.strategy!r}")
        claims = tuple(
            ClaimDraft(
                statement=item.statement,
                citations=(item.citation_label(),),
                evidence_item_ids=(item.item_id,),
                direction=item.direction,
            )
            for item in supporting
        )
        return self._result(
            task,
            claims,
            items=items,
            notes=(
                f"strategy {self.strategy}: {len(claims)} claim(s) "
                "from verified supporting items",
            ),
        )


def _population_statement(model: DiseaseModel, value: float) -> str:
    return (
        f"Under the illustrative {model.name or 'disease'} model "
        f"(prevalence {model.prevalence:.6g}, {model.inheritance.value}), "
        f"the maximum credible allele frequency is {value:.6g}."
    )


_POPULATION_CITATION = (
    "Whiffin et al. 2017 maximum credible allele frequency "
    "framework, computed with provome.population"
)


@dataclass
class PopulationWorker(BuiltinWorker):
    """Computes the Whiffin maximum credible allele frequency.

    Two strategies, one expected number: ``"direct"`` calls
    :func:`provome.population.max_credible_af`; ``"evaluation"``
    reaches the same ceiling through
    :func:`provome.population.evaluate_frequency`.  The claim carries
    the number as a registered numeric claim, so QC stage 3
    recomputes it a third time before it can be issued.
    """

    strategy: str = "direct"

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        model = context.payload.get("disease_model")
        if model is None:
            return self._failed(task, "no disease model supplied in payload")
        if self.strategy == "direct":
            value = max_credible_af(model)
        else:
            value = evaluate_frequency(0.0, model=model).max_credible_af
        claim = ClaimDraft(
            statement=_population_statement(model, value),
            citations=(_POPULATION_CITATION,),
            evidence_item_ids=("computed:max_credible_af",),
            numeric_label="max_credible_af",
            numeric_value=value,
        )
        return self._result(task, (claim,), notes=(f"strategy {self.strategy}",))


def _translate_tolerant(sequence: str) -> str:
    """Translate complete codons, ignoring a trailing partial codon."""
    seq = sequence.upper()
    return "".join(
        CODON_TABLE[seq[i : i + 3]] for i in range(0, len(seq) - 2, 3)
    )


def _first_diff(a: str, b: str) -> Optional[int]:
    """1-based index of the first position where two strings differ."""
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index + 1
    if len(a) != len(b):
        return min(len(a), len(b)) + 1
    return None


def _consequence_statement(variant: str, term: str, residue: Optional[int]) -> str:
    return (
        f"Molecular consequence of {variant} on the supplied CDS: "
        f"{term} (first affected residue: {residue})."
    )


_CONSEQUENCE_CITATION = (
    "provome.consequence: standard genetic code consequence calling "
    "on the supplied coding sequence"
)


@dataclass
class ConsequenceWorker(BuiltinWorker):
    """Calls molecular consequence on a supplied CDS and variant.

    Two strategies: ``"engine"`` uses
    :func:`provome.consequence.call_consequence`; ``"verify"`` is an
    independent derivation - it edits the sequence with
    :func:`provome.consequence.apply_coding_variant`, translates both
    proteins codon by codon, and derives the term from the protein
    comparison itself.  If the two derivations ever disagree, the
    claims differ, the orchestra records a disagreement, and the
    escalation path - not silent averaging - decides what happens.
    """

    strategy: str = "engine"

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        cds = context.payload.get("cds")
        variant_text = context.payload.get("coding_variant")
        if not cds or not variant_text:
            return self._failed(
                task, "consequence work needs 'cds' and 'coding_variant' in payload"
            )
        if self.strategy == "engine":
            result = call_consequence(cds, variant_text)
            term = result.term.value
            residue = result.first_affected_residue
        else:
            parsed = parse_coding_hgvs(variant_text)
            alt_seq = apply_coding_variant(cds, parsed)
            ref_protein = _translate_tolerant(cds).rstrip("*")
            alt_protein = _translate_tolerant(alt_seq).rstrip("*")
            residue = _first_diff(ref_protein, alt_protein)
            if (len(alt_seq) - len(cds)) % 3 != 0:
                term = "frameshift_variant"
            elif alt_seq == cds:
                term = "no_sequence_effect"
            elif alt_protein == ref_protein:
                term = "synonymous_variant"
            elif "*" in alt_protein and alt_protein.index("*") < len(ref_protein):
                term = "stop_gained"
            elif ref_protein and alt_protein[:1] != ref_protein[:1]:
                term = "start_lost"
            elif len(alt_protein) < len(ref_protein):
                term = "inframe_deletion"
            elif len(alt_protein) > len(ref_protein):
                term = "inframe_insertion"
            else:
                term = "missense_variant"
        claim = ClaimDraft(
            statement=_consequence_statement(variant_text, term, residue),
            citations=(_CONSEQUENCE_CITATION,),
            evidence_item_ids=("computed:consequence",),
        )
        return self._result(task, (claim,), notes=(f"strategy {self.strategy}",))


def _acmg_statement(total: int, by_points: str, by_rules: str) -> str:
    return (
        f"ACMG classification of the supplied assessments (illustrative "
        f"curation): {by_points} by Tavtigian points (total {total}); "
        f"Richards combining rules give {by_rules}."
    )


_ACMG_CITATION = (
    "Richards et al. 2015 combining rules and Tavtigian et al. 2018 "
    "points, computed with provome.acmg"
)


@dataclass
class AcmgWorker(BuiltinWorker):
    """Classifies supplied ACMG assessments under both frameworks.

    ``"report"`` uses :func:`provome.acmg.classify`; ``"recompute"``
    rebuilds the same outcome from ``total_points``,
    ``classify_by_points``, and ``classify_by_richards_rules``
    separately.  Human division, clinical context only - the
    orchestra only assigns this work when a caller supplies
    assessments in the payload.
    """

    strategy: str = "report"

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        assessments = context.payload.get("acmg_assessments")
        if not assessments:
            return self._failed(task, "no 'acmg_assessments' supplied in payload")
        if self.strategy == "report":
            report = classify(assessments)
            total = report.points_total
            by_points = report.points_classification.value
            by_rules = report.richards_classification.value
        else:
            total = total_points(assessments)
            by_points = classify_by_points(total).value
            by_rules = classify_by_richards_rules(assessments).value
        claim = ClaimDraft(
            statement=_acmg_statement(total, by_points, by_rules),
            citations=(_ACMG_CITATION,),
            evidence_item_ids=("computed:acmg",),
        )
        return self._result(task, (claim,), notes=(f"strategy {self.strategy}",))


def _calibration_statement(auroc_value: float, ece_value: float, n: int) -> str:
    return (
        f"Calibration check on the supplied labelled set: AUROC "
        f"{auroc_value:.6g}, ECE {ece_value:.6g} over {n} predictions."
    )


_CALIBRATION_CITATION = (
    "provome.calibration: rank-method AUROC and binned ECE, computed "
    "from the supplied labels and scores"
)


@dataclass
class CalibrationWorker(BuiltinWorker):
    """Evaluates supplied labels and scores.

    ``"report"`` uses :func:`provome.calibration.evaluate`;
    ``"recompute"`` rebuilds AUROC and ECE from ``auroc`` and
    ``calibration_table`` directly.  The AUROC is carried as a
    registered numeric claim for QC stage 3.
    """

    strategy: str = "report"

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        labels = context.payload.get("labels")
        scores = context.payload.get("scores")
        if labels is None or scores is None:
            return self._failed(task, "calibration work needs 'labels' and 'scores'")
        if self.strategy == "report":
            report = calibration_evaluate(labels, scores)
            auroc_value, ece_value, n = report.auroc, report.ece, report.n
        else:
            table = calibration_table(labels, scores)
            auroc_value = auroc(labels, scores)
            ece_value, n = table.ece, len(labels)
        claim = ClaimDraft(
            statement=_calibration_statement(auroc_value, ece_value, n),
            citations=(_CALIBRATION_CITATION,),
            evidence_item_ids=("computed:calibration",),
            numeric_label="auroc",
            numeric_value=auroc_value,
        )
        return self._result(task, (claim,), notes=(f"strategy {self.strategy}",))


def _phenotype_statement(similarity: float) -> str:
    return (
        f"Resnik best-match average similarity between the two "
        f"supplied phenotype term sets: {similarity:.6g} (illustrative "
        f"example ontology)."
    )


_PHENOTYPE_CITATION = (
    "provome.phenotype: Resnik semantic similarity over the bundled "
    "illustrative ontology subset"
)


@dataclass
class PhenotypeWorker(BuiltinWorker):
    """Compares two phenotype term sets on the example ontology.

    ``"compare"`` uses :func:`provome.phenotype.compare_term_sets`;
    ``"average"`` uses :func:`provome.phenotype.best_match_average`.
    Both are the same published mathematics reached by two routes.
    """

    strategy: str = "compare"

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        set_a = context.payload.get("term_set_a")
        set_b = context.payload.get("term_set_b")
        if not set_a or not set_b:
            return self._failed(
                task, "phenotype work needs 'term_set_a' and 'term_set_b'"
            )
        ontology = example_ontology()
        if self.strategy == "compare":
            similarity = compare_term_sets(ontology, set_a, set_b).similarity
        else:
            similarity = best_match_average(ontology, set_a, set_b)
        claim = ClaimDraft(
            statement=_phenotype_statement(similarity),
            citations=(_PHENOTYPE_CITATION,),
            evidence_item_ids=("computed:phenotype",),
        )
        return self._result(task, (claim,), notes=(f"strategy {self.strategy}",))


def default_worker_registry() -> WorkerRegistry:
    """The built-in worker registry: senior workers for every engine.

    Each capability has at least two workers that reach the same
    result by different routes (that is what makes cross-checking
    real), and evidence has a third, the tiebreak worker.  Worker
    ids sort so that assignment is deterministic: the first two
    workers of a capability are the primary pair, the rest are the
    tiebreak pool.
    """
    return WorkerRegistry(
        [
            EvidenceWorker(
                worker_id="evidence-crosscheck",
                role_id=ROLE_SENIOR_RESEARCHER,
                team_id=TEAM_EVIDENCE_VERIFICATION,
                capabilities=("evidence",),
                strategy="manual",
            ),
            EvidenceWorker(
                worker_id="evidence-primary",
                role_id=ROLE_SENIOR_EVIDENCE_REVIEWER,
                team_id=TEAM_EVIDENCE_VERIFICATION,
                capabilities=("evidence",),
                strategy="gate",
            ),
            EvidenceWorker(
                worker_id="evidence-tiebreak",
                role_id=ROLE_SENIOR_SCIENTIST,
                team_id=TEAM_EVIDENCE_VERIFICATION,
                capabilities=("evidence",),
                strategy="recount",
            ),
            PopulationWorker(
                worker_id="population-crosscheck",
                role_id=ROLE_SENIOR_DATA_ANALYST,
                team_id=TEAM_CALIBRATION_BENCHMARKS,
                capabilities=("population",),
                strategy="evaluation",
            ),
            PopulationWorker(
                worker_id="population-primary",
                role_id=ROLE_SENIOR_DATA_ANALYST,
                team_id=TEAM_CALIBRATION_BENCHMARKS,
                capabilities=("population",),
                strategy="direct",
            ),
            ConsequenceWorker(
                worker_id="consequence-primary",
                role_id=ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
                team_id=TEAM_DNA_VARIANT,
                capabilities=("consequence",),
                strategy="engine",
            ),
            ConsequenceWorker(
                worker_id="consequence-verification",
                role_id=ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
                team_id=TEAM_DNA_VARIANT,
                capabilities=("consequence",),
                strategy="verify",
            ),
            AcmgWorker(
                worker_id="acmg-crosscheck",
                role_id=ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
                team_id=TEAM_DNA_VARIANT,
                capabilities=("acmg",),
                strategy="recompute",
            ),
            AcmgWorker(
                worker_id="acmg-primary",
                role_id=ROLE_SENIOR_SCIENTIST,
                team_id=TEAM_DNA_VARIANT,
                capabilities=("acmg",),
                strategy="report",
            ),
            CalibrationWorker(
                worker_id="calibration-crosscheck",
                role_id=ROLE_SENIOR_DATA_ANALYST,
                team_id=TEAM_CALIBRATION_BENCHMARKS,
                capabilities=("calibration",),
                strategy="recompute",
            ),
            CalibrationWorker(
                worker_id="calibration-primary",
                role_id=ROLE_SENIOR_DATA_ANALYST,
                team_id=TEAM_CALIBRATION_BENCHMARKS,
                capabilities=("calibration",),
                strategy="report",
            ),
            PhenotypeWorker(
                worker_id="phenotype-crosscheck",
                role_id=ROLE_SENIOR_RESEARCHER,
                team_id=TEAM_RARE_DISEASE,
                capabilities=("phenotype",),
                strategy="average",
            ),
            PhenotypeWorker(
                worker_id="phenotype-primary",
                role_id=ROLE_SENIOR_SCIENTIST,
                team_id=TEAM_RARE_DISEASE,
                capabilities=("phenotype",),
                strategy="compare",
            ),
        ]
    )


# ---------------------------------------------------------------------
# The Chief Copilot: orchestration
# ---------------------------------------------------------------------

from .domains import get_division  # noqa: E402 - grouped with the engine

#: Illustrative disease model for the bundled CFTR demo topic.  The
#: inputs are the textbook shape of cystic fibrosis (autosomal
#: recessive, birth prevalence of the order of 1 in 2500 in the
#: populations where it is most studied); they are model inputs for
#: demonstrating the Whiffin computation, labelled illustrative in
#: every claim built from them, not a clinical statement.
_CF_DEMO_MODEL = DiseaseModel(
    prevalence=0.0004,
    inheritance=Inheritance.AUTOSOMAL_RECESSIVE,
    name="cystic fibrosis",
)

#: Topics in the bundled demo data that carry a population model.
_TOPIC_POPULATION_MODELS = {
    "cftr:p.phe508del": _CF_DEMO_MODEL,
}


@dataclass(frozen=True)
class OrchestratedAnswer:
    """Full record of one orchestrated run.

    Attributes:
        answer: The issued :class:`provome.copilot.CopilotAnswer`
            (ANSWER with QC summary and audit id attached, or
            DECLINE with every failed QC stage named).
        team: The team the question was routed to.
        qc_report: The six-stage QC report for the run.
        audit_log: The complete audit trail of the run.
        assignments: Assignment records (task id, kind, worker id,
            purpose) for every worker assigned, including tiebreaks.
        escalations: Plain language record of every escalation the
            team lead ordered during the run.
    """

    answer: CopilotAnswer
    team: Team
    qc_report: QCReport
    audit_log: AuditLog
    assignments: tuple[dict, ...] = ()
    escalations: tuple[str, ...] = ()

    @property
    def answered(self) -> bool:
        return self.answer.answered

    def to_dict(self) -> dict:
        return {
            "answer": self.answer.to_dict(),
            "team": {
                "team_id": self.team.team_id,
                "name": self.team.name,
                "lead_role_id": self.team.lead_role_id,
            },
            "qc_report": self.qc_report.to_dict(),
            "audit_id": self.audit_log.audit_id,
            "assignments": [dict(a) for a in self.assignments],
            "escalations": list(self.escalations),
        }


class Orchestra:
    """The Chief Copilot: routes, assigns, cross-checks, and signs off.

    Args:
        provider: Evidence provider for topic recognition and
            evidence items.  Defaults to the offline demo provider.
        source_registry: Source registry for independence groups.
        workers: Worker registry.  Defaults to the built-in senior
            workers.  An empty or partial registry is legal: work
            that cannot be staffed produces a clean decline, never
            a crash, because an unstaffed organisation must say so
            rather than improvise.
        teams: Team registry for routing.
        roles: Role registry (used to name leads in the audit).
        min_agreement: Distinct workers that must independently
            produce each key claim.  Defaults to 2.
    """

    def __init__(
        self,
        provider: Optional[EvidenceProvider] = None,
        source_registry: Optional[SourceRegistry] = None,
        workers: Optional[WorkerRegistry] = None,
        teams: Optional[TeamRegistry] = None,
        roles: Optional[RoleRegistry] = None,
        min_agreement: int = 2,
    ) -> None:
        self.provider = provider if provider is not None else OfflineDemoProvider()
        if source_registry is None:
            # Illustrative sources are registered only when the
            # bundled demo provider is the one supplying evidence.
            source_registry = (
                demo_source_registry()
                if isinstance(self.provider, OfflineDemoProvider)
                else default_source_registry()
            )
        self.source_registry = source_registry
        self.workers = workers if workers is not None else default_worker_registry()
        self.teams = teams if teams is not None else default_team_registry()
        self.roles = roles if roles is not None else default_role_registry()
        if min_agreement < 1:
            raise ValueError("min_agreement must be at least 1")
        self.min_agreement = min_agreement

    # -- main entry point -------------------------------------------
    def ask(
        self,
        question: CopilotQuestion | str,
        *,
        engine_tasks: Sequence[str] = (),
        payload: Optional[dict] = None,
    ) -> OrchestratedAnswer:
        """Run one question through the whole organisation.

        Args:
            question: The question, as text or a
                :class:`CopilotQuestion` (which may carry division
                or species context).
            engine_tasks: Extra engine capabilities to assign for
                this question (``"consequence"``, ``"acmg"``,
                ``"calibration"``, ``"phenotype"``), with their
                inputs in ``payload``.
            payload: Computation inputs for engine workers; see
                :class:`WorkContext`.
        """
        audit = AuditLog()
        if isinstance(question, str):
            question = CopilotQuestion(text=question)
        audit.record(
            "received", "chief-copilot",
            f"question received: {question.text}",
            division=question.division.value,
        )
        team = self.teams.route(question.division, text=question.text)
        audit.record(
            "routed", "chief-copilot",
            f"routed to {team.name} (lead role: {team.lead_role_id})",
            team_id=team.team_id, lead_role_id=team.lead_role_id,
        )
        division_info = get_division(question.division)
        warnings: list[str] = []
        if question.division is not Division.HUMAN:
            warnings.append(
                f"this is a {division_info.name} division question; "
                "the cite-or-decline gate applies unchanged, and "
                "human-clinical frameworks such as ACMG/AMP are not "
                "applied outside the Human division"
            )

        topic_key = self.provider.topic_for_question(question.text)
        if topic_key is None:
            reason = (
                "no recognised variant, gene, or disease topic "
                "matched the question in the evidence provider, so "
                "no evidence could be gathered; the Copilot declines "
                "rather than guessing a topic"
            )
            report = _not_run_report(reason, audit_id=audit.audit_id)
            audit.record("decline", "chief-copilot", f"declined: {reason}")
            answer = CopilotAnswer(
                verdict=Verdict.DECLINE,
                question=question.text,
                topic_key=None,
                division=question.division,
                division_name=division_info.name,
                species=question.species,
                declined_reasons=(reason, "QC stage 1 (evidence_gate) failed: " + reason),
                warnings=tuple(warnings),
                qc_summary=report.to_dict(),
                audit_id=audit.audit_id,
            )
            return OrchestratedAnswer(
                answer=answer, team=team, qc_report=report, audit_log=audit
            )

        items = tuple(self.provider.items_for_topic(topic_key))
        run_payload: dict = dict(payload or {})
        kinds: list[str] = ["evidence"]
        model = _TOPIC_POPULATION_MODELS.get(topic_key)
        if model is not None:
            run_payload.setdefault("disease_model", model)
            kinds.append("population")
        for kind in engine_tasks:
            if kind not in kinds:
                kinds.append(kind)

        context = WorkContext(
            question=question,
            topic_key=topic_key,
            items=items,
            source_registry=self.source_registry,
            team=team,
            payload=run_payload,
        )

        assignments: list[dict] = []
        results: list[WorkResult] = []
        ran: dict[str, set] = {}
        task_kind: dict[str, str] = {}
        task_counter = [0]

        def run_worker(worker: Worker, kind: str, purpose: str) -> None:
            task_counter[0] += 1
            task = WorkTask(
                task_id=f"task-{task_counter[0]}",
                kind=kind,
                topic_key=topic_key,
                team_id=team.team_id,
                purpose=purpose,
            )
            task_kind[task.task_id] = kind
            ran.setdefault(kind, set()).add(worker.worker_id)
            assignments.append(
                {
                    "task_id": task.task_id,
                    "kind": kind,
                    "worker_id": worker.worker_id,
                    "role_id": worker.role_id,
                    "purpose": purpose,
                }
            )
            audit.record(
                "assignment", "chief-copilot",
                f"{task.task_id} ({kind}, {purpose}) assigned to "
                f"worker {worker.worker_id} ({worker.role_id})",
                task_id=task.task_id, worker_id=worker.worker_id, kind=kind,
            )
            try:
                result = worker.run(task, context)
            except Exception as exc:  # a worker must never crash the run
                audit.record(
                    "worker_error", worker.worker_id,
                    f"worker {worker.worker_id} raised "
                    f"{type(exc).__name__}: {exc}; its result is "
                    "treated as failed and excluded",
                    task_id=task.task_id, worker_id=worker.worker_id,
                )
                result = WorkResult(
                    task_id=task.task_id,
                    worker_id=worker.worker_id,
                    role_id=worker.role_id,
                    team_id=worker.team_id,
                    ok=False,
                    error=f"worker raised {type(exc).__name__}: {exc}",
                )
            results.append(result)
            audit.record(
                "result", worker.worker_id,
                f"worker {worker.worker_id} returned "
                f"{len(result.claims)} claim(s), ok={result.ok}",
                task_id=task.task_id, worker_id=worker.worker_id,
            )

        for kind in kinds:
            pool = self.workers.workers_for(kind)
            if not pool:
                audit.record(
                    "assignment", "chief-copilot",
                    f"no workers registered for capability {kind!r}; "
                    "the task is unassigned and its absence will be "
                    "handled by the QC pipeline, not improvised",
                    kind=kind,
                )
                continue
            for index, worker in enumerate(pool[:2]):
                run_worker(
                    worker, kind, "primary" if index == 0 else "crosscheck"
                )

        accepted, unresolved, escalations = self._resolve(
            results, audit, team, run_worker, task_kind, ran
        )

        agreement: dict[str, tuple[str, ...]] = {}
        merged: list[ClaimDraft] = []
        for (key, _direction), group in accepted.items():
            workers_in_group = tuple(sorted(group["workers"]))
            agreement[key] = workers_in_group
            first = group["claims"][0]
            citations: list[str] = []
            evidence_ids: list[str] = []
            numeric_label = None
            numeric_value = None
            for claim in group["claims"]:
                for citation in claim.citations:
                    if citation not in citations:
                        citations.append(citation)
                for item_id in claim.evidence_item_ids:
                    if item_id not in evidence_ids:
                        evidence_ids.append(item_id)
                if claim.numeric_label is not None:
                    numeric_label = claim.numeric_label
                    numeric_value = claim.numeric_value
            merged.append(
                ClaimDraft(
                    statement=first.statement,
                    citations=tuple(citations),
                    evidence_item_ids=tuple(evidence_ids),
                    direction=first.direction,
                    numeric_label=numeric_label,
                    numeric_value=numeric_value,
                    worker_id=", ".join(workers_in_group),
                )
            )

        checkers: dict = {}
        qc_context: dict = {}
        if "disease_model" in run_payload:
            checkers["max_credible_af"] = (
                lambda ctx: max_credible_af(ctx["disease_model"])
            )
            qc_context["disease_model"] = run_payload["disease_model"]
        if "labels" in run_payload and "scores" in run_payload:
            checkers["auroc"] = lambda ctx: auroc(ctx["labels"], ctx["scores"])
            qc_context["labels"] = run_payload["labels"]
            qc_context["scores"] = run_payload["scores"]

        qc_input = QCInput(
            items=items,
            topic_key=topic_key,
            source_registry=self.source_registry,
            claims=tuple(merged),
            agreement=agreement,
            numeric_checkers=checkers,
            context=qc_context,
            min_agreement=self.min_agreement,
            unresolved_disagreements=unresolved,
        )
        report = run_qc_pipeline(qc_input)
        report = QCReport(stages=report.stages, audit_id=audit.audit_id)
        for stage in report.stages:
            audit.record(
                "qc_stage", ROLE_SENIOR_QC_LEAD,
                f"QC stage {stage.stage} ({stage.name}): "
                f"{'passed' if stage.passed else 'FAILED'} - "
                f"{'; '.join(stage.details)}",
                stage=stage.name, passed=stage.passed,
            )

        gate = evaluate_gate(list(items), topic_key, registry=self.source_registry)
        warnings.extend(gate.warnings)

        if report.passed:
            audit.record(
                "signoff", team.lead_role_id,
                f"senior review sign-off by {team.name} lead "
                f"({team.lead_role_id}): all QC stages passed",
                team_id=team.team_id,
            )
            audit.record(
                "answer", "chief-copilot",
                "final answer issued by the Chief Copilot",
                topic_key=topic_key,
            )
            answer = CopilotAnswer(
                verdict=Verdict.ANSWER,
                question=question.text,
                topic_key=topic_key,
                division=question.division,
                division_name=division_info.name,
                species=question.species,
                claims=tuple(
                    Claim(
                        statement=c.statement,
                        citations=c.citations,
                        evidence_item_ids=(
                            c.evidence_item_ids or ("computed:orchestra",)
                        ),
                    )
                    for c in merged
                ),
                confidence=_confidence(gate),
                verified_item_count=gate.verified_count,
                independent_source_count=gate.independent_source_count,
                warnings=tuple(warnings),
                qc_summary=report.to_dict(),
                audit_id=audit.audit_id,
            )
        else:
            reasons = tuple(
                f"QC stage {s.stage} ({s.name}) failed: "
                f"{s.details[0] if s.details else 'no details recorded'}"
                for s in report.stages
                if not s.passed
            )
            audit.record(
                "decline", "chief-copilot",
                "declined; failed QC stage(s): "
                + ", ".join(report.failed_stages),
                failed_stages=list(report.failed_stages),
            )
            answer = CopilotAnswer(
                verdict=Verdict.DECLINE,
                question=question.text,
                topic_key=topic_key,
                division=question.division,
                division_name=division_info.name,
                species=question.species,
                declined_reasons=reasons,
                verified_item_count=gate.verified_count,
                independent_source_count=gate.independent_source_count,
                warnings=tuple(warnings),
                qc_summary=report.to_dict(),
                audit_id=audit.audit_id,
            )
        return OrchestratedAnswer(
            answer=answer,
            team=team,
            qc_report=report,
            audit_log=audit,
            assignments=tuple(assignments),
            escalations=tuple(escalations),
        )

    # -- screening and cross-worker resolution -----------------------
    def _resolve(
        self,
        results: list[WorkResult],
        audit: AuditLog,
        team: Team,
        run_worker: Callable[[Worker, str, str], None],
        task_kind: dict[str, str],
        ran: dict[str, set],
    ) -> tuple[dict, int, list[str]]:
        """Screen results, resolve agreement, escalate, tiebreak.

        Returns ``(accepted_groups, unresolved_disagreements,
        escalations)`` where accepted groups are keyed by
        ``(claim_key, direction)`` with ``workers`` and ``claims``.
        Mutates ``results`` by appending tiebreak results.
        """
        escalations: list[str] = []

        def screen(result_list: list[WorkResult]) -> list[WorkResult]:
            kept: list[WorkResult] = []
            for result in result_list:
                if not result.ok:
                    audit.record(
                        "exclusion", ROLE_SENIOR_QC_LEAD,
                        f"result from worker {result.worker_id} "
                        f"excluded: {result.error or 'worker reported failure'}",
                        worker_id=result.worker_id, task_id=result.task_id,
                    )
                    continue
                bad = [c for c in result.claims if not c.citations]
                if bad:
                    audit.record(
                        "exclusion", ROLE_SENIOR_QC_LEAD,
                        f"result from worker {result.worker_id} "
                        f"excluded: {len(bad)} claim(s) without a "
                        "citation (citation check); a worker's "
                        "uncited claim is never issued and never "
                        "quietly repaired",
                        worker_id=result.worker_id, task_id=result.task_id,
                    )
                    continue
                kept.append(result)
            return kept

        def group_claims(result_list: list[WorkResult]) -> dict:
            groups: dict = {}
            for result in result_list:
                kind = task_kind.get(result.task_id, "")
                for claim in result.claims:
                    entry = groups.setdefault(
                        (claim.claim_key, claim.direction),
                        {"workers": set(), "claims": [], "kinds": set()},
                    )
                    entry["workers"].add(result.worker_id)
                    entry["claims"].append(claim)
                    entry["kinds"].add(kind)
            return groups

        kept = screen(results)
        groups = group_claims(kept)

        def problem_keys(current: dict) -> tuple[set, set]:
            by_key: dict[str, set] = {}
            for (key, direction) in current:
                by_key.setdefault(key, set()).add(direction)
            disputed = {k for k, dirs in by_key.items() if len(dirs) > 1}
            insufficient = {
                key
                for (key, _direction), entry in current.items()
                if len(entry["workers"]) < self.min_agreement
            }
            return disputed, insufficient

        disputed, insufficient = problem_keys(groups)
        if disputed or insufficient:
            for key in sorted(disputed):
                audit.record(
                    "disagreement", team.lead_role_id,
                    f"workers disagree on direction for claim: {key[:80]}",
                    claim_key=key,
                )
            kinds_involved: set = set()
            for (key, _direction), entry in groups.items():
                if key in disputed or key in insufficient:
                    kinds_involved.update(entry["kinds"])
            message = (
                f"team lead ({team.lead_role_id}) escalated: "
                f"{len(disputed)} disputed claim(s), "
                f"{len(insufficient)} claim(s) below the "
                f"{self.min_agreement}-worker agreement threshold; "
                "ordering tiebreak workers for: "
                + ", ".join(sorted(kinds_involved))
            )
            escalations.append(message)
            audit.record(
                "escalation", team.lead_role_id, message,
                team_id=team.team_id,
            )
            new_results: list[WorkResult] = []
            before = len(results)
            for kind in sorted(kinds_involved):
                pool = self.workers.workers_for(kind)
                spare = [
                    w for w in pool if w.worker_id not in ran.get(kind, set())
                ]
                if spare:
                    audit.record(
                        "tiebreak", team.lead_role_id,
                        f"tiebreak ordered for capability {kind!r}: "
                        f"worker {spare[0].worker_id}",
                        kind=kind, worker_id=spare[0].worker_id,
                    )
                    run_worker(spare[0], kind, "tiebreak")
            new_results = results[before:]
            if new_results:
                kept = kept + screen(new_results)
                groups = group_claims(kept)

        unresolved = 0
        by_key: dict[str, list] = {}
        for (key, direction), entry in groups.items():
            by_key.setdefault(key, []).append((direction, entry))
        accepted: dict = {}
        for key, variants in by_key.items():
            if len(variants) > 1:
                variants_sorted = sorted(
                    variants, key=lambda v: len(v[1]["workers"]), reverse=True
                )
                winner_direction, winner = variants_sorted[0]
                runner_up_workers = variants_sorted[1][1]["workers"]
                if (
                    len(winner["workers"]) == len(runner_up_workers)
                    or len(winner["workers"]) < self.min_agreement
                ):
                    unresolved += 1
                    audit.record(
                        "exclusion", ROLE_SENIOR_QC_LEAD,
                        f"claim dropped: disagreement on {key[:80]} "
                        "could not be resolved by tiebreak; an "
                        "unresolved claim is never issued",
                        claim_key=key,
                    )
                    continue
                for direction, entry in variants:
                    if direction is not winner_direction:
                        audit.record(
                            "exclusion", ROLE_SENIOR_QC_LEAD,
                            f"claim direction {direction.value} from "
                            f"worker(s) {sorted(entry['workers'])} "
                            f"excluded for {key[:60]}: cross-check "
                            f"favoured {winner_direction.value} by "
                            f"{len(winner['workers'])} worker(s)",
                            claim_key=key,
                        )
                accepted[(key, winner_direction)] = winner
            else:
                direction, entry = variants[0]
                if len(entry["workers"]) < self.min_agreement:
                    audit.record(
                        "exclusion", ROLE_SENIOR_QC_LEAD,
                        f"claim dropped: produced by only "
                        f"{len(entry['workers'])} worker(s) after "
                        f"escalation, below the {self.min_agreement}"
                        f"-worker agreement threshold: {key[:80]}",
                        claim_key=key,
                    )
                    continue
                accepted[(key, direction)] = entry
        return accepted, unresolved, escalations


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Demonstration: the CFTR case through the full organisation.

    Prints the routed team, every assigned worker, every QC stage,
    and the verdict for one ANSWER case, then the BRCA1 decline with
    its failed stages named.
    """
    orchestra = Orchestra()
    for question_text in (
        "What is CFTR p.Phe508del and does it cause cystic fibrosis?",
        "What can you tell me about the BRCA1 c.68_69delAG founder variant?",
    ):
        run = orchestra.ask(CopilotQuestion(text=question_text))
        print(f"Q: {question_text}")
        print(f"Team: {run.team.name} (lead role: {run.team.lead_role_id})")
        print("Workers assigned:")
        for assignment in run.assignments:
            print(
                f"  - {assignment['worker_id']} [{assignment['kind']}, "
                f"{assignment['purpose']}] as {assignment['role_id']}"
            )
        print("QC pipeline:")
        for stage in run.qc_report.stages:
            print(
                f"  stage {stage.stage} {stage.name}: "
                f"{'PASS' if stage.passed else 'FAIL'}"
            )
        print(f"Verdict: {run.answer.verdict.value.upper()}")
        if run.answer.answered:
            for claim in run.answer.claims:
                print(f"  - {claim.statement}")
        else:
            for reason in run.answer.declined_reasons:
                print(f"  - {reason}")
        print(f"Audit: {run.audit_log.audit_id} ({len(run.audit_log)} events)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
