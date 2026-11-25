"""The agent layer: real labour underneath the one Copilot.

The Founder's decision (2026-11-25): the Copilot stays the face of
Provome, and agents are added underneath it as the labour layer.
This module is that layer, deliberately the smallest one that does
real work.  Three agents, each with one job, each measurable:

* :class:`RetrievalAgent` — wraps an evidence provider and fetches
  the evidence for a question: topic recognition plus the item set.
  It adds bookkeeping (what was asked, what came back) so retrieval
  completeness can be measured instead of assumed.
* :class:`VerificationAgent` — re-checks literature items against
  PubMed, independently of whatever the fetching provider claimed.
  A literature item that arrives stamped VERIFIED is not believed:
  its PMID is re-confirmed (record exists, title matches) through
  :mod:`provome.confirmation`, the same check the live providers
  are held to at fetch time.  Fabricated PMIDs come back FAILED,
  unresolvable ones UNVERIFIED, retracted papers FAILED with an
  integrity warning attached, and a FAILED item is never upgraded
  by a later pass — failure is sticky, exactly as in the gate's
  deduplication.  Since the strengthening pass (2026-11-25) it also
  re-confirms ClinVar items against the served ClinVar record:
  identity must match the item's topic, and the item's direction
  is re-derived from the served classification, never taken from
  the item's own claim — the fix for the fraud battery's two
  KNOWN GAP misses.  Items from other sources, and ClinVar items
  whose topic names no concrete variant (nothing to match a
  record against), pass through untouched; that scope is stated
  plainly rather than implied.
* :class:`LiteratureAgent` — searches a local literature index
  (:mod:`provome.rag`) for a topic and bridges what it finds into
  evidence items, with the PubMed confirmer wired in.  The live
  providers cap literature per topic; a literature-scale corpus
  finds papers a capped live feed misses.  Nothing it emits is
  VERIFIED unless the confirmer actually confirmed it.

Composition: :class:`AgentBackedProvider` is an ordinary evidence
provider built from the three agents, so the plain
:class:`~provome.copilot.Copilot` runs the full stack unchanged —
retrieval, corpus literature, independent re-verification — and
every item still has to pass the cite-or-decline gate before the
Copilot may say anything.  No agent in this module can assert
anything: agents return evidence items and audit reports, never
claims.  The gate remains the only place an answer is allowed to
come from.

No stubs: a unit that does nothing is not shipped.  Each agent's
behaviour is pinned by tests with fixtures, including the
adversarial cases from the gate red-team probes (fabricated PMIDs,
self-declared sources, unanchored identifiers).

Standard library only.  Research software.  Not for clinical use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

from .confirmation import (
    clinvar_variation_id,
    confirm_clinvar,
    confirm_literature_detail,
    make_pubmed_confirmer,
    variant_short_from_topic,
)
from .copilot import EvidenceProvider
from .evidence import EvidenceItem, VerificationStatus
from .rag import RetrievalIndex, evidence_from_retrieval

__all__ = [
    "AgentReport",
    "RetrievalResult",
    "RetrievalAgent",
    "VerificationAgent",
    "LiteratureAgent",
    "AgentBackedProvider",
    "LITERATURE_SOURCE_ID",
    "CLINVAR_SOURCE_ID",
]

#: The source id whose items the verification agent re-checks.
LITERATURE_SOURCE_ID = "pubmed-literature"

#: The source id whose items are re-confirmed against the served
#: ClinVar record (identity match + direction from the record).
CLINVAR_SOURCE_ID = "clinvar"

_PMID_IN_CITATION_RE = re.compile(r"\bPMID[\s:#]*0*(\d{1,9})\b", re.IGNORECASE)
_PMID_IN_URL_RE = re.compile(
    r"pubmed\.ncbi\.nlm\.nih\.gov/0*(\d{1,9})\b", re.IGNORECASE
)
_QUOTED_TITLE_RE = re.compile(r'"([^"]+)"')


@dataclass(frozen=True)
class AgentReport:
    """What one agent did, in auditable numbers.

    Attributes:
        agent: The agent's name.
        items_in: Items handed to the agent (0 for retrieval, which
            starts from a question).
        items_out: Items the agent returned.
        notes: Plain language record of consequential actions:
            demotions, exclusions, confirmations.
    """

    agent: str
    items_in: int
    items_out: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    """The retrieval agent's product for one question."""

    topic_key: Optional[str]
    items: tuple[EvidenceItem, ...]
    report: AgentReport


class RetrievalAgent:
    """Fetches evidence for questions through a wrapped provider.

    The agent owns the two-step provider protocol (recognise the
    topic, then fetch its items) and records what happened, so a
    caller can see whether a decline came from "no topic" or from
    "topic recognised, evidence thin" — the difference between a
    retrieval miss and an honest evidence gap.
    """

    name = "retrieval-agent"

    def __init__(self, provider: EvidenceProvider) -> None:
        self.provider = provider

    def topic_for_question(self, question_text: str) -> Optional[str]:
        return self.provider.topic_for_question(question_text)

    def retrieve(self, question_text: str) -> RetrievalResult:
        """Recognise the topic and fetch its evidence items."""
        topic_key = self.provider.topic_for_question(question_text)
        if topic_key is None:
            return RetrievalResult(
                topic_key=None,
                items=(),
                report=AgentReport(
                    agent=self.name,
                    items_in=0,
                    items_out=0,
                    notes=(
                        "no topic recognised in the question; "
                        "nothing retrieved",
                    ),
                ),
            )
        items = tuple(self.provider.items_for_topic(topic_key))
        return RetrievalResult(
            topic_key=topic_key,
            items=items,
            report=AgentReport(
                agent=self.name,
                items_in=0,
                items_out=len(items),
                notes=(
                    f"topic {topic_key}: {len(items)} item(s) "
                    f"fetched from provider "
                    f"{getattr(self.provider, 'name', type(self.provider).__name__)}",
                ),
            ),
        )

    def items_for_topic(self, topic_key: str) -> tuple[EvidenceItem, ...]:
        return tuple(self.provider.items_for_topic(topic_key))


def _literature_pmid(item: EvidenceItem) -> Optional[str]:
    """The PMID a literature item claims, from citation or locator."""
    for text in (item.citation, item.locator or ""):
        match = _PMID_IN_CITATION_RE.search(text) or _PMID_IN_URL_RE.search(text)
        if match:
            return match.group(1)
    return None


def _claimed_title(item: EvidenceItem) -> Optional[str]:
    """The paper title a literature item's statement quotes, if any.

    Both literature producers in this package quote the title in
    the statement (the live provider: ``... : "Title" (year).``;
    the RAG bridge: ``Peer-reviewed literature reports: "Title"``).
    An item whose title cannot be recovered cannot be re-confirmed
    and is treated as unconfirmable, never as confirmed.
    """
    match = _QUOTED_TITLE_RE.search(item.statement)
    if match:
        return match.group(1).rstrip(".")
    return None


class VerificationAgent:
    """Independently re-confirms literature and ClinVar items.

    Args:
        transport: The injectable HTTP transport used for the
            re-confirmations, called exactly as the live providers
            call theirs.  Required: an agent that cannot check must
            not pretend to, so there is no default that silently
            skips the check.  (Tests inject fixtures; a deployment
            injects the real transport.)

    Behaviour per literature item (``source_id ==
    "pubmed-literature"``):

    * FAILED stays FAILED.  Failure is sticky; this agent never
      resurrects an item a check already killed.
    * Otherwise the claimed PMID and title are extracted and
      re-confirmed via
      :func:`provome.confirmation.confirm_literature_detail`.
      The item leaves with the confirmation's status — VERIFIED
      only when PubMed holds the PMID and the titles match.
    * A served record carrying a retraction notice fails the
      item, and the integrity warning is attached to the item's
      own notes so it travels with the item.  The flag records
      the withdrawal; it accuses nobody.
    * An item with no recoverable PMID or title is demoted to
      UNVERIFIED: it cannot be re-confirmed, so it cannot count.

    Behaviour per ClinVar item (``source_id == "clinvar"``):

    * FAILED stays FAILED, as above.
    * When the item claims a Variation ID and its topic names a
      concrete variant, the item is re-confirmed via
      :func:`provome.confirmation.confirm_clinvar`: the served
      record must exist and be about that variant, and the item
      leaves with the served record's status and a direction
      *derived from the served classification* — a Benign record
      can no longer be dressed as pathogenic support by editing
      the item.
    * A ClinVar item with no recoverable Variation ID, or whose
      topic names no concrete variant, cannot be identity-matched
      and passes through unchanged; that scope limit is stated
      here rather than implied.

    Items from any other source pass through unchanged.
    """

    name = "verification-agent"

    def __init__(self, transport: Callable) -> None:
        if transport is None:
            raise ValueError(
                "VerificationAgent requires a transport: an agent "
                "that cannot check must not pretend to"
            )
        self.transport = transport

    def verify(
        self, items: Sequence[EvidenceItem]
    ) -> tuple[tuple[EvidenceItem, ...], AgentReport]:
        """Re-confirm literature and ClinVar items; return + report."""
        out: list[EvidenceItem] = []
        notes: list[str] = []
        for item in items:
            if item.source_id == CLINVAR_SOURCE_ID:
                out.append(self._verify_clinvar(item, notes))
                continue
            if item.source_id != LITERATURE_SOURCE_ID:
                out.append(item)
                continue
            if item.is_failed:
                notes.append(
                    f"{item.item_id}: already FAILED; failure is "
                    "sticky, not re-checked"
                )
                out.append(item)
                continue
            pmid = _literature_pmid(item)
            title = _claimed_title(item)
            if pmid is None or title is None:
                if item.status is not VerificationStatus.UNVERIFIED:
                    notes.append(
                        f"{item.item_id}: demoted to UNVERIFIED; "
                        "no recoverable PMID/title pair to "
                        "re-confirm against PubMed"
                    )
                out.append(item.with_status(VerificationStatus.UNVERIFIED))
                continue
            status, note, retracted = confirm_literature_detail(
                self.transport, pmid, title
            )
            if status is not item.status:
                notes.append(
                    f"{item.item_id}: {item.status.value} -> "
                    f"{status.value} on independent re-confirmation "
                    f"(PMID {pmid})"
                )
            new_item = (
                item.with_status(status)
                if status is not item.status
                else item
            )
            if retracted:
                notes.append(
                    f"{item.item_id}: integrity warning attached "
                    f"(PMID {pmid} is retracted)"
                )
                new_item = replace(
                    new_item,
                    notes=(
                        f"{new_item.notes} {note}".strip()
                        if new_item.notes
                        else note
                    ),
                )
            out.append(new_item)
        return tuple(out), AgentReport(
            agent=self.name,
            items_in=len(items),
            items_out=len(out),
            notes=tuple(notes),
        )

    def _verify_clinvar(
        self, item: EvidenceItem, notes: list[str]
    ) -> EvidenceItem:
        """Re-confirm one ClinVar item against the served record."""
        if item.is_failed:
            notes.append(
                f"{item.item_id}: already FAILED; failure is "
                "sticky, not re-checked"
            )
            return item
        variation_id = clinvar_variation_id(
            item.citation, item.locator or ""
        )
        if variation_id is None or (
            variant_short_from_topic(item.topic_key) is None
        ):
            # Nothing to identity-match against: outside the
            # confirmer's remit, stated in the class docstring.
            return item
        status, direction, note = confirm_clinvar(
            self.transport, variation_id, item.topic_key
        )
        new_direction = (
            direction if direction is not None else item.direction
        )
        if status is item.status and new_direction is item.direction:
            return item
        notes.append(
            f"{item.item_id}: ClinVar re-confirmation "
            f"(Variation ID {variation_id}): status "
            f"{item.status.value} -> {status.value}, direction "
            f"{item.direction.value} -> {new_direction.value}; "
            "direction derived from the served record"
        )
        return replace(
            item,
            status=status,
            direction=new_direction,
            notes=f"{item.notes} {note}".strip() if item.notes else note,
        )


class LiteratureAgent:
    """Gathers literature evidence for a topic from a local index.

    Args:
        index: The retrieval index over the literature corpus.
        transport: Transport for the PubMed confirmer wired into
            the bridge.  Required, for the same reason as the
            verification agent's: unconfirmed literature must never
            be emitted as verified, and wiring the confirmer is
            what makes confirmation the default rather than an
            option a caller can forget.
        k: Maximum documents to bridge per topic.
    """

    name = "literature-agent"

    def __init__(
        self,
        index: RetrievalIndex,
        transport: Callable,
        k: int = 5,
    ) -> None:
        if transport is None:
            raise ValueError(
                "LiteratureAgent requires a transport for the "
                "PubMed confirmer: unconfirmed literature is never "
                "emitted as verified"
            )
        self.index = index
        self.confirmer = make_pubmed_confirmer(transport)
        self.k = k

    def gather(
        self, query_text: str, topic_key: str
    ) -> tuple[tuple[EvidenceItem, ...], AgentReport]:
        """Bridge the index's best documents for a topic to items."""
        items = evidence_from_retrieval(
            self.index,
            query_text,
            topic_key=topic_key,
            k=self.k,
            confirmer=self.confirmer,
        )
        verified = sum(1 for item in items if item.is_verified)
        notes = (
            f"topic {topic_key}: {len(items)} item(s) bridged "
            f"from the corpus, {verified} confirmed VERIFIED, "
            f"{len(items) - verified} not counted (unconfirmed or "
            "failed confirmation)",
        )
        return tuple(items), AgentReport(
            agent=self.name,
            items_in=len(self.index),
            items_out=len(items),
            notes=notes,
        )


class AgentBackedProvider:
    """An evidence provider composed from the three agents.

    Retrieval fetches the wrapped provider's items for the topic;
    the literature agent adds whatever the corpus holds beyond the
    wrapped provider's own literature; the verification agent then
    re-confirms every literature item in the combined set.  The
    Copilot sees an ordinary provider; the gate sees ordinary
    evidence items; the agents' reports are kept on
    :attr:`last_reports` for audit.
    """

    name = "agent-backed"

    def __init__(
        self,
        provider: EvidenceProvider,
        transport: Callable,
        index: Optional[RetrievalIndex] = None,
    ) -> None:
        self.retrieval = RetrievalAgent(provider)
        self.verification = VerificationAgent(transport)
        self.literature = (
            LiteratureAgent(index, transport) if index is not None else None
        )
        scoped = getattr(provider, "scoped_divisions", None)
        if scoped is not None:
            self.scoped_divisions = scoped
        self._cache: dict[str, tuple[EvidenceItem, ...]] = {}
        self.last_reports: tuple[AgentReport, ...] = ()

    def topic_for_question(self, question_text: str) -> Optional[str]:
        return self.retrieval.topic_for_question(question_text)

    def items_for_topic(self, topic_key: str) -> Sequence[EvidenceItem]:
        if topic_key in self._cache:
            return self._cache[topic_key]
        reports: list[AgentReport] = []
        items = list(self.retrieval.items_for_topic(topic_key))
        reports.append(
            AgentReport(
                agent=self.retrieval.name,
                items_in=0,
                items_out=len(items),
                notes=(
                    f"topic {topic_key}: {len(items)} item(s) from "
                    "the wrapped provider",
                ),
            )
        )
        if self.literature is not None:
            lit_items, lit_report = self.literature.gather(
                topic_key.replace(":", " "), topic_key
            )
            items.extend(lit_items)
            reports.append(lit_report)
        items_verified, verify_report = self.verification.verify(items)
        reports.append(verify_report)
        self.last_reports = tuple(reports)
        self._cache[topic_key] = items_verified
        return items_verified
