"""The Omevra Copilot: one copilot for variant, gene, and disease questions.

There is exactly one copilot in Omevra.  It is not a chatbot bolted
onto the side of the package, and there is no separate "bot": the
Copilot is the question-answering surface of the evidence engine
itself, and every design decision in it exists to keep its answers
honest:

* It answers a question about a variant, gene, or disease only from
  :class:`~omevra.evidence.EvidenceItem` objects supplied by a
  pluggable evidence provider.
* It enforces the cite-or-decline gate from
  :mod:`omevra.evidence`: at least three verified evidence items
  from at least two independent sources, or the output is a
  structured DECLINE that names exactly what is missing.
* Every claim in an answer carries the citations of the items behind
  it.  A claim without citations cannot be constructed by the
  answer builder, because claims are built *from* items, never
  written freely and decorated afterwards.
* The optional language model hook is an interface only, guarded off
  by default.  When enabled by a host application it may rephrase an
  already-gated answer; it can never add claims, because the claims
  list is fixed before the hook is called.  This package ships no
  model access and makes no network calls.

Divisions: a question may carry a division or species context (see
:mod:`omevra.domains`).  The default is Human.  The division is
carried through to the structured answer, and the Copilot states
plainly when a framework does not apply to a division - most
importantly, ACMG/AMP classification is human-clinical specific and
is never silently applied to animal or plant questions.

Running this module directly demonstrates both outcomes on bundled
offline data:

    python -m omevra.copilot

prints one ANSWER case (CFTR p.Phe508del, enough verified evidence)
and one DECLINE case (BRCA1 c.68_69delAG in the demo set, where the
bundled evidence is deliberately below the gate).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, Sequence, runtime_checkable

from .domains import Division, division_for_species, get_division
from .evidence import (
    EvidenceItem,
    EvidenceKind,
    GateDecision,
    SourceRegistry,
    SupportDirection,
    VerificationStatus,
    default_source_registry,
    evaluate_gate,
)

__all__ = [
    "Verdict",
    "CopilotQuestion",
    "Claim",
    "CopilotAnswer",
    "EvidenceProvider",
    "OfflineDemoProvider",
    "LLMHook",
    "DisabledLLMHook",
    "Copilot",
    "OmevraCopilot",
    "OrchestratedCopilot",
    "main",
]


class Verdict(str, Enum):
    """Whether the Copilot answered or declined."""

    ANSWER = "answer"
    DECLINE = "decline"


@dataclass(frozen=True)
class CopilotQuestion:
    """A question put to the Copilot, with division context.

    Attributes:
        text: The question, in natural language.
        division: Division context.  Defaults to Human for backward
            compatibility: callers who never think about divisions
            get the human division, explicitly.
        species: Optional species the question is about, as a
            scientific or common name.  When the species is in the
            division registry, the species determines the division
            (a question about *Bos taurus* is an Animal division
            question even if the caller forgot to say so).
    """

    text: str
    division: Division = Division.HUMAN
    species: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("question text must be non-empty")
        division = self.division
        if isinstance(division, str):
            division = Division(division.strip().lower())
            object.__setattr__(self, "division", division)
        if self.species:
            info = division_for_species(self.species)
            if info is not None and info.division is not division:
                object.__setattr__(self, "division", info.division)


@dataclass(frozen=True)
class Claim:
    """One claim in an answer, with the citations that support it.

    A claim is always built from evidence items; ``evidence_item_ids``
    and ``citations`` are therefore non-empty by construction and by
    validation.
    """

    statement: str
    citations: tuple[str, ...]
    evidence_item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.citations or any(
            not isinstance(c, str) or not c.strip() for c in self.citations
        ):
            raise ValueError("a claim without citations may not be constructed")
        if not self.evidence_item_ids or any(
            not isinstance(i, str) or not i.strip()
            for i in self.evidence_item_ids
        ):
            raise ValueError("a claim must name its evidence items")


@dataclass(frozen=True)
class CopilotAnswer:
    """Structured output of the Copilot.

    Attributes:
        verdict: ANSWER or DECLINE.
        question: The question text as asked.
        topic_key: The normalised topic the evidence was gathered
            for, or ``None`` when no known topic matched.
        division: The division the answer belongs to.
        division_name: Display name of the division.
        species: Species context, when supplied.
        claims: Claims with citations.  Empty on a decline.
        confidence: Heuristic confidence in an answer, in [0, 1]:
            a transparent function of how far past the gate the
            evidence goes (see :func:`_confidence`).  It is a
            reporting heuristic, not a calibrated probability, and
            is 0.0 on a decline.
        declined_reasons: On a decline, exactly which gate
            requirements failed.  Empty on an answer.
        verified_item_count: Verified supporting items counted.
        independent_source_count: Independent source groups counted.
        warnings: Non-blocking caveats, including division notes.
        narrative: Optional rephrasing produced by an enabled LLM
            hook.  ``None`` unless a host application enabled one.
        qc_summary: When this answer came from the orchestrated
            Copilot (:mod:`omevra.orchestra`), the QC report summary:
            the six stages and which passed.  ``None`` for answers
            from the plain Copilot, which only applies the gate.
        audit_id: Id of the audit log for an orchestrated run, or
            ``None`` for a plain Copilot answer.
    """

    verdict: Verdict
    question: str
    topic_key: Optional[str]
    division: Division
    division_name: str
    species: Optional[str] = None
    claims: tuple[Claim, ...] = ()
    confidence: float = 0.0
    declined_reasons: tuple[str, ...] = ()
    verified_item_count: int = 0
    independent_source_count: int = 0
    warnings: tuple[str, ...] = ()
    narrative: Optional[str] = None
    qc_summary: Optional[dict] = None
    audit_id: Optional[str] = None

    @property
    def answered(self) -> bool:
        return self.verdict is Verdict.ANSWER

    def to_dict(self) -> dict:
        """Serialise to plain JSON-compatible data."""
        return {
            "verdict": self.verdict.value,
            "question": self.question,
            "topic_key": self.topic_key,
            "division": self.division.value,
            "division_name": self.division_name,
            "species": self.species,
            "claims": [
                {
                    "statement": claim.statement,
                    "citations": list(claim.citations),
                    "evidence_item_ids": list(claim.evidence_item_ids),
                }
                for claim in self.claims
            ],
            "confidence": self.confidence,
            "declined_reasons": list(self.declined_reasons),
            "verified_item_count": self.verified_item_count,
            "independent_source_count": self.independent_source_count,
            "warnings": list(self.warnings),
            "narrative": self.narrative,
            "qc_summary": self.qc_summary,
            "audit_id": self.audit_id,
        }

    def render(self) -> str:
        """Render a compact human readable form of the answer."""
        lines: list[str] = []
        lines.append(f"[{self.verdict.value.upper()}] division: {self.division_name}")
        lines.append(f"Q: {self.question}")
        if self.topic_key:
            lines.append(f"Topic: {self.topic_key}")
        if self.answered:
            lines.append(
                f"Evidence: {self.verified_item_count} verified item(s) "
                f"from {self.independent_source_count} independent "
                f"source group(s); confidence (heuristic): "
                f"{self.confidence:.2f}"
            )
            for claim in self.claims:
                lines.append(f"- {claim.statement}")
                for citation in claim.citations:
                    lines.append(f"  Source: {citation}")
        else:
            lines.append("Declined. Reasons:")
            for reason in self.declined_reasons:
                lines.append(f"- {reason}")
        for warning in self.warnings:
            lines.append(f"Note: {warning}")
        if self.qc_summary is not None:
            failed = self.qc_summary.get("failed_stages", [])
            if self.qc_summary.get("passed"):
                lines.append("QC: all six stages passed")
            else:
                lines.append(
                    "QC: failed stage(s): " + ", ".join(failed)
                )
            if self.audit_id:
                lines.append(f"Audit: {self.audit_id}")
        return "\n".join(lines)


@runtime_checkable
class EvidenceProvider(Protocol):
    """Protocol for evidence providers the Copilot can use.

    A provider does two things: recognise which topic a question is
    about, and supply the evidence items it holds for a topic.  A
    production provider might query curated databases; the bundled
    :class:`OfflineDemoProvider` reads packaged example data and
    makes no network calls.
    """

    def topic_for_question(self, question_text: str) -> Optional[str]:
        """Return the topic key a question is about, or ``None``."""
        ...

    def items_for_topic(self, topic_key: str) -> Sequence[EvidenceItem]:
        """Return all evidence items held for a topic."""
        ...


def _load_demo_data() -> dict:
    path = Path(__file__).parent / "data" / "demo_evidence.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class OfflineDemoProvider:
    """Evidence provider backed by the bundled illustrative data set.

    The data lives in ``omevra/data/demo_evidence.json`` and is
    labelled illustrative in that file and in the README.  Topic
    recognition is simple alias matching, deliberately transparent:
    the longest alias found in the question wins, and no alias means
    no topic, which means the Copilot declines rather than guessing.
    """

    name = "offline-demo"

    def __init__(self, data: Optional[dict] = None) -> None:
        self._data = data if data is not None else _load_demo_data()
        self._topics: dict[str, dict] = self._data.get("topics", {})

    def topic_for_question(self, question_text: str) -> Optional[str]:
        lowered = question_text.lower()
        best_topic: Optional[str] = None
        best_length = -1
        for topic_key, topic in self._topics.items():
            for alias in topic.get("aliases", []):
                alias_lower = alias.lower()
                # Word-boundary match: an alias buried inside a
                # larger token (e.g. "xcftrx") must not match
                # (hardening, Worker A Bug 8).
                pattern = r"(?<![a-z0-9])" + re.escape(alias_lower) + r"(?![a-z0-9])"
                if re.search(pattern, lowered) and len(alias_lower) > best_length:
                    best_topic = topic_key
                    best_length = len(alias_lower)
        return best_topic

    def items_for_topic(self, topic_key: str) -> Sequence[EvidenceItem]:
        topic = self._topics.get(topic_key)
        if topic is None:
            return ()
        items: list[EvidenceItem] = []
        for raw in topic.get("items", []):
            items.append(
                EvidenceItem(
                    item_id=raw["item_id"],
                    topic_key=topic_key,
                    statement=raw["statement"],
                    source_id=raw["source_id"],
                    citation=raw["citation"],
                    kind=EvidenceKind(raw.get("kind", "other")),
                    direction=SupportDirection(raw.get("direction", "supports")),
                    status=VerificationStatus(raw.get("status", "unverified")),
                    locator=raw.get("locator"),
                )
            )
        return tuple(items)

    def known_topics(self) -> tuple[str, ...]:
        return tuple(self._topics.keys())


@runtime_checkable
class LLMHook(Protocol):
    """Interface for an optional language model rephrasing hook.

    Implementations receive a finished, gated :class:`CopilotAnswer`
    and return a narrative rephrasing.  They cannot change the
    claims: the answer object is immutable and the Copilot only
    attaches the returned text as ``narrative``.
    """

    def rephrase(self, answer: CopilotAnswer) -> str:
        """Return a narrative rephrasing of a gated answer."""
        ...


class DisabledLLMHook:
    """The default hook: present, explicit, and off.

    Calling it raises, so an accidental enablement fails loudly
    instead of silently producing unreviewed text.
    """

    def rephrase(self, answer: CopilotAnswer) -> str:
        raise RuntimeError(
            "the LLM hook is disabled; Omevra answers are issued "
            "from gated evidence only"
        )


def _confidence(gate: GateDecision) -> float:
    """Heuristic confidence for a passed gate, in [0, 1].

    Transparent by design: 0.60 at the gate threshold (3 items, 2
    groups), plus 0.08 for each additional verified item (capped at
    three extra), plus 0.07 for each additional independent group
    (capped at two extra), for a maximum of 0.95.  This is a
    reporting heuristic about evidence depth, not a calibrated
    probability; calibrated probabilities are the business of
    :mod:`omevra.calibration` applied to real evaluation data.
    """
    extra_items = min(max(gate.verified_count - 3, 0), 3)
    extra_groups = min(max(gate.independent_source_count - 2, 0), 2)
    return round(min(0.60 + 0.08 * extra_items + 0.07 * extra_groups, 0.95), 4)


class Copilot:
    """The Omevra Copilot.

    Args:
        provider: Evidence provider.  Defaults to
            :class:`OfflineDemoProvider`.
        registry: Source registry for independence groups.
        llm_hook: Optional rephrasing hook.  Off by default; passing
            a hook does not enable it unless ``use_llm`` is also set.
        use_llm: Explicit opt-in to the hook.  Defaults to False.
    """

    def __init__(
        self,
        provider: Optional[EvidenceProvider] = None,
        registry: Optional[SourceRegistry] = None,
        llm_hook: Optional[LLMHook] = None,
        use_llm: bool = False,
    ) -> None:
        self.provider = provider if provider is not None else OfflineDemoProvider()
        self.registry = registry if registry is not None else default_source_registry()
        self.llm_hook = llm_hook
        self.use_llm = bool(use_llm)
        if self.use_llm and self.llm_hook is None:
            raise ValueError("use_llm=True requires an llm_hook implementation")

    # -- main entry point ------------------------------------------
    def ask(self, question: CopilotQuestion | str) -> CopilotAnswer:
        """Answer or decline a question, under the evidence gate."""
        if isinstance(question, str):
            question = CopilotQuestion(text=question)
        if not isinstance(question, CopilotQuestion):
            raise TypeError(
                "question must be a CopilotQuestion or a string, "
                f"got {type(question).__name__}"
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

        # Hardening (Worker A Bug 2): the bundled evidence set is
        # human-division evidence. It must never be issued as an
        # answer in the Animal or Plant divisions, whose evidence
        # is not yet division-scoped or benchmarked.
        if question.division is not Division.HUMAN:
            return self._finish(
                CopilotAnswer(
                    verdict=Verdict.DECLINE,
                    question=question.text,
                    topic_key=None,
                    division=question.division,
                    division_name=division_info.name,
                    species=question.species,
                    declined_reasons=(
                        f"the {division_info.name} division is an "
                        "expansion division: its evidence is not yet "
                        "division-scoped or benchmarked, and human "
                        "clinical evidence is never issued outside "
                        "the Human division; the Copilot declines "
                        "rather than leaking human evidence into a "
                        f"{division_info.name} answer",
                    ),
                    warnings=tuple(warnings),
                )
            )

        topic_key = self.provider.topic_for_question(question.text)
        if topic_key is None:
            return self._finish(
                CopilotAnswer(
                    verdict=Verdict.DECLINE,
                    question=question.text,
                    topic_key=None,
                    division=question.division,
                    division_name=division_info.name,
                    species=question.species,
                    declined_reasons=(
                        "no recognised variant, gene, or disease "
                        "topic matched the question in the evidence "
                        "provider, so no evidence could be gathered; "
                        "the Copilot declines rather than guessing a "
                        "topic",
                    ),
                    warnings=tuple(warnings),
                )
            )

        items = list(self.provider.items_for_topic(topic_key))
        gate = evaluate_gate(items, topic_key, registry=self.registry)
        warnings.extend(gate.warnings)

        if not gate.passed:
            return self._finish(
                CopilotAnswer(
                    verdict=Verdict.DECLINE,
                    question=question.text,
                    topic_key=topic_key,
                    division=question.division,
                    division_name=division_info.name,
                    species=question.species,
                    declined_reasons=gate.reasons,
                    verified_item_count=gate.verified_count,
                    independent_source_count=gate.independent_source_count,
                    warnings=tuple(warnings),
                )
            )

        claims = tuple(
            Claim(
                statement=item.statement,
                citations=(item.citation_label(),),
                evidence_item_ids=(item.item_id,),
            )
            for item in gate.verified_supporting
        )
        return self._finish(
            CopilotAnswer(
                verdict=Verdict.ANSWER,
                question=question.text,
                topic_key=topic_key,
                division=question.division,
                division_name=division_info.name,
                species=question.species,
                claims=claims,
                confidence=_confidence(gate),
                verified_item_count=gate.verified_count,
                independent_source_count=gate.independent_source_count,
                warnings=tuple(warnings),
            )
        )

    def _finish(self, answer: CopilotAnswer) -> CopilotAnswer:
        """Attach an LLM narrative only when explicitly enabled."""
        if not self.use_llm:
            return answer
        if not answer.answered:
            return answer
        assert self.llm_hook is not None
        narrative = self.llm_hook.rephrase(answer)
        return CopilotAnswer(
            verdict=answer.verdict,
            question=answer.question,
            topic_key=answer.topic_key,
            division=answer.division,
            division_name=answer.division_name,
            species=answer.species,
            claims=answer.claims,
            confidence=answer.confidence,
            declined_reasons=answer.declined_reasons,
            verified_item_count=answer.verified_item_count,
            independent_source_count=answer.independent_source_count,
            warnings=answer.warnings,
            narrative=narrative,
            qc_summary=answer.qc_summary,
            audit_id=answer.audit_id,
        )


#: Descriptive alias: there is one copilot, and this is its full name.
OmevraCopilot = Copilot


class OrchestratedCopilot:
    """The same one Copilot, running through the whole organisation.

    This is the user-facing entry point for the multi-agent
    organisation in :mod:`omevra.orchestra`: one Copilot on the
    outside, with a Chief Copilot orchestrating senior teams and
    workers inside, six QC stages, and a full audit trail.  The
    plain :class:`Copilot` still exists for callers who want only
    the evidence gate; this class is the production surface.

    The import of :mod:`omevra.orchestra` is deferred to
    construction so the two modules do not import each other at
    load time.
    """

    def __init__(
        self,
        provider: Optional[EvidenceProvider] = None,
        registry: Optional[SourceRegistry] = None,
        workers=None,
        teams=None,
        roles=None,
    ) -> None:
        from .orchestra import Orchestra

        self._orchestra = Orchestra(
            provider=provider,
            source_registry=registry,
            workers=workers,
            teams=teams,
            roles=roles,
        )

    def ask(self, question, *, engine_tasks=(), payload=None):
        """Answer or decline a question through the organisation."""
        return self._orchestra.ask(
            question, engine_tasks=engine_tasks, payload=payload
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Demonstration entry point: one ANSWER and one DECLINE case."""
    copilot = Copilot()
    answer_case = copilot.ask(
        CopilotQuestion(
            text=(
                "What is CFTR p.Phe508del and does it cause cystic "
                "fibrosis?"
            )
        )
    )
    decline_case = copilot.ask(
        CopilotQuestion(
            text=(
                "What can you tell me about the BRCA1 c.68_69delAG "
                "founder variant?"
            )
        )
    )
    animal_case = copilot.ask(
        CopilotQuestion(
            text="What is CFTR p.Phe508del and does it cause cystic fibrosis?",
            species="Bos taurus",
        )
    )
    print(answer_case.render())
    print()
    print(decline_case.render())
    print()
    print(
        "Division context check: the same question asked with "
        "species Bos taurus is carried as division "
        f"'{animal_case.division.value}' and "
        f"{animal_case.verdict.value}s, because human evidence is "
        "never issued outside the Human division."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
