"""Evidence models, source registry, verification, and the cite-or-decline gate.

This module is the trust layer of Provome.  Every statement the Provome
Copilot is allowed to make has to be backed by :class:`EvidenceItem`
objects, and those objects have to survive three independent checks:

1. **Provenance.**  Each item names a registered source.  A source is a
   publication, a curated database, or a primary data set.  Sources are
   grouped into *independence groups*: two records drawn from the same
   underlying curation pipeline are not independent, even if they are
   displayed by two different web sites.

2. **Verification.**  An item is only *verified* when its citation has
   been checked against the source it claims to come from.  Unverified
   items are carried through the system so a human can see them, but
   they never count toward the gate.

3. **The gate.**  :func:`evaluate_gate` implements the cite-or-decline
   rule used across Provome: a question is answerable only when at least
   ``MIN_VERIFIED_ITEMS`` verified items from at least
   ``MIN_INDEPENDENT_SOURCES`` independent source groups support it.
   Anything less produces a structured decline with explicit reasons,
   rather than a weaker, vaguer answer.

The design follows the principle, common to clinical variant
interpretation practice (Richards et al. 2015, *Genetics in Medicine*
17:405-424), that an assertion is only as strong as the evidence lines
behind it, and that independent lines of evidence matter more than
repeated copies of one line.

Standard library only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "MIN_VERIFIED_ITEMS",
    "MIN_INDEPENDENT_SOURCES",
    "EvidenceKind",
    "SupportDirection",
    "VerificationStatus",
    "Source",
    "SourceRegistry",
    "default_source_registry",
    "EvidenceItem",
    "GateDecision",
    "evaluate_gate",
    "deduplicate",
    "fingerprint",
]


#: Minimum number of verified evidence items required to answer.
MIN_VERIFIED_ITEMS = 3

#: Minimum number of independent source groups required to answer.
MIN_INDEPENDENT_SOURCES = 2


class EvidenceKind(str, Enum):
    """The methodological kind of an evidence item.

    The kind matters because three items of the same kind, drawn from
    the same kind of experiment, are weaker than three items of
    different kinds that converge on the same conclusion.
    """

    POPULATION_FREQUENCY = "population_frequency"
    FUNCTIONAL_STUDY = "functional_study"
    CLINICAL_OBSERVATION = "clinical_observation"
    SEGREGATION = "segregation"
    COMPUTATIONAL_PREDICTION = "computational_prediction"
    CURATED_CLASSIFICATION = "curated_classification"
    LITERATURE_REPORT = "literature_report"
    EXPERIMENTAL_MODEL = "experimental_model"
    OTHER = "other"


class SupportDirection(str, Enum):
    """Whether an item supports, refutes, or is neutral to a claim."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


class VerificationStatus(str, Enum):
    """Verification state of an evidence item.

    ``VERIFIED`` means the citation was checked and the source does
    say what the item claims it says.  ``UNVERIFIED`` is the default
    for newly ingested material.  ``FAILED`` means a check was made
    and the source did not support the item; failed items are kept for
    audit but can never be cited.
    """

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"


@dataclass(frozen=True)
class Source:
    """A registered evidence source.

    Attributes:
        source_id: Short stable identifier, for example ``"clinvar"``.
        name: Human readable name.
        independence_group: Sources sharing a group are treated as one
            independent line of evidence by the gate.  For example, a
            mirror of a database shares the database's group.
        description: What the source is and how it curates material.
        base_url: Optional canonical URL of the source.
        peer_reviewed: Whether the source's content is peer reviewed
            or expert curated under a published process.
    """

    source_id: str
    name: str
    independence_group: str
    description: str = ""
    base_url: Optional[str] = None
    peer_reviewed: bool = False

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not self.name or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not self.independence_group or not self.independence_group.strip():
            raise ValueError("independence_group must be a non-empty string")


class SourceRegistry:
    """A registry of known sources, keyed by ``source_id``.

    The registry exists so that an evidence item cannot quietly invent
    a source.  Items referencing an unregistered source can be built,
    but :meth:`validate_item` will flag them, and the Copilot treats
    unknown sources as their own singleton independence group with a
    warning attached to the gate decision.
    """

    def __init__(self, sources: Iterable[Source] = ()) -> None:
        self._sources: dict[str, Source] = {}
        for source in sources:
            self.register(source)

    def register(self, source: Source) -> None:
        """Add a source; raises ``ValueError`` on a duplicate id."""
        if source.source_id in self._sources:
            raise ValueError(f"duplicate source_id: {source.source_id!r}")
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> Optional[Source]:
        """Return the source with this id, or ``None``."""
        return self._sources.get(source_id)

    def require(self, source_id: str) -> Source:
        """Return the source with this id or raise ``KeyError``."""
        source = self._sources.get(source_id)
        if source is None:
            raise KeyError(f"unknown source_id: {source_id!r}")
        return source

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._sources

    def __len__(self) -> int:
        return len(self._sources)

    def all(self) -> tuple[Source, ...]:
        """All registered sources, in registration order."""
        return tuple(self._sources.values())

    def independence_group_of(self, source_id: str) -> str:
        """Independence group for a source id.

        Unknown ids fall back to a singleton group named after the id,
        prefixed so it can never collide with a registered group.
        Note: the gate never counts unregistered sources (see
        :func:`evaluate_gate`); this label exists for diagnostics.
        """
        source = self._sources.get(source_id)
        if source is None:
            return f"unregistered:{source_id}"
        return source.independence_group


def default_source_registry() -> SourceRegistry:
    """Build the registry of sources Provome knows about out of the box.

    These are real, well known resources in human genetics and drug
    discovery.  Listing a source here does not by itself mean a
    provider queries it; the registry describes provenance for
    evidence items that a provider supplies, and the gate counts
    only items from registered sources.
    """
    return SourceRegistry(
        [
            Source(
                source_id="clinvar",
                name="ClinVar",
                independence_group="ncbi-clinvar",
                description=(
                    "NCBI archive of human variant interpretations "
                    "submitted by clinical and research laboratories."
                ),
                base_url="https://www.ncbi.nlm.nih.gov/clinvar/",
                peer_reviewed=False,
            ),
            Source(
                source_id="gnomad",
                name="gnomAD",
                independence_group="gnomad",
                description=(
                    "Genome Aggregation Database: allele frequencies "
                    "from large scale exome and genome sequencing."
                ),
                base_url="https://gnomad.broadinstitute.org/",
            ),
            Source(
                source_id="pubmed-literature",
                name="Peer-reviewed literature (PubMed indexed)",
                independence_group="primary-literature",
                description=(
                    "Primary peer-reviewed publications reporting "
                    "functional, clinical, or segregation evidence."
                ),
                base_url="https://pubmed.ncbi.nlm.nih.gov/",
                peer_reviewed=True,
            ),
            Source(
                source_id="omim",
                name="OMIM",
                independence_group="omim",
                description=(
                    "Online Mendelian Inheritance in Man: curated "
                    "gene and phenotype relationships."
                ),
                base_url="https://www.omim.org/",
            ),
            Source(
                source_id="uniprot",
                name="UniProt",
                independence_group="uniprot",
                description=(
                    "Curated protein sequence and functional annotation."
                ),
                base_url="https://www.uniprot.org/",
            ),
            Source(
                source_id="chembl",
                name="ChEMBL",
                independence_group="ebi-chembl",
                description=(
                    "EMBL-EBI database of bioactive molecules, drug "
                    "targets, mechanisms of action, and clinical "
                    "development phases."
                ),
                base_url="https://www.ebi.ac.uk/chembl/",
            ),
            Source(
                source_id="ensembl",
                name="Ensembl",
                independence_group="ensembl",
                description=(
                    "Ensembl genome browser: gene, variation, and "
                    "phenotype records across animal and other "
                    "non-human species, via the Ensembl REST API."
                ),
                base_url="https://rest.ensembl.org/",
            ),
            Source(
                source_id="ensembl-plants",
                name="Ensembl Plants",
                independence_group="ensembl",
                description=(
                    "Ensembl Plants genome browser: gene and trait "
                    "records across plant species, via the Ensembl "
                    "REST API. Same organisation as Ensembl, so it "
                    "shares the 'ensembl' independence group and "
                    "never counts as a second independent source "
                    "alongside it."
                ),
                base_url="https://rest.ensembl.org/",
            ),
            Source(
                source_id="demo-bundled",
                name="Provome bundled illustrative examples",
                independence_group="provome-demo",
                description=(
                    "Small hand-checked example set bundled with this "
                    "package for offline demonstration and testing. "
                    "Illustrative only; not a clinical data source."
                ),
            ),
            Source(
                source_id="demo-bundled-2",
                name="Provome bundled illustrative examples, second set",
                independence_group="provome-demo-2",
                description=(
                    "Second illustrative example set, curated "
                    "separately from the first set so the offline "
                    "demo can exercise the independence requirement."
                ),
            ),
        ]
    )


def fingerprint(
    source_id: str,
    citation: str,
    statement: str,
) -> str:
    """Stable content fingerprint used for deduplication.

    Two items with the same source, the same citation, and the same
    normalised statement are the same evidence, however they were
    phrased by an ingest pipeline.  Normalisation is deliberately
    conservative: case folding and whitespace collapsing only.
    """
    normalised = " ".join(statement.lower().split())
    payload = f"{source_id}\n{citation.strip()}\n{normalised}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceItem:
    """One citable unit of evidence.

    Attributes:
        item_id: Stable identifier for this item.
        topic_key: Normalised key of what the item is about, for
            example ``"cftr:p.phe508del"``.  The Copilot only combines
            items that share a topic key.
        statement: The claim the item supports, in plain language.
        source_id: Id of a source in a :class:`SourceRegistry`.
        citation: Full citation text a reader can follow up, for
            example a paper title with journal and year, or a database
            accession.
        kind: Methodological kind of evidence.
        direction: Whether the item supports or refutes the topic's
            main claim.
        status: Verification status.  Only verified items count
            toward the gate.
        locator: Optional pinpoint inside the source, such as a table
            or figure number, a variant accession, or a URL.
        notes: Free text notes, for example why verification failed.
    """

    item_id: str
    topic_key: str
    statement: str
    source_id: str
    citation: str
    kind: EvidenceKind = EvidenceKind.OTHER
    direction: SupportDirection = SupportDirection.SUPPORTS
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    locator: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.item_id or not self.item_id.strip():
            raise ValueError("item_id must be non-empty")
        if not self.topic_key or not self.topic_key.strip():
            raise ValueError("topic_key must be non-empty")
        if not self.statement.strip():
            raise ValueError("statement must be non-empty")
        if not self.citation.strip():
            raise ValueError("citation must be non-empty")

    @property
    def is_verified(self) -> bool:
        """True when this item may be cited in an answer."""
        return self.status is VerificationStatus.VERIFIED

    @property
    def is_failed(self) -> bool:
        return self.status is VerificationStatus.FAILED

    def content_fingerprint(self) -> str:
        """Fingerprint used by :func:`deduplicate`."""
        return fingerprint(self.source_id, self.citation, self.statement)

    def with_status(self, status: VerificationStatus) -> "EvidenceItem":
        """Return a copy of this item with a new verification status."""
        return replace(self, status=status)

    def citation_label(self) -> str:
        """Short label used when rendering citations in an answer."""
        if self.locator:
            return f"{self.citation} ({self.locator})"
        return self.citation


def deduplicate(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    """Remove duplicate evidence items, keeping the strongest copy.

    Duplicates are detected by content fingerprint.  When duplicates
    disagree about verification, the verified copy wins, because a
    later successful check supersedes an unchecked ingest.  A failed
    copy never displaces a verified one.  Order of first appearance is
    preserved for the survivors.
    """
    rank = {
        VerificationStatus.VERIFIED: 2,
        VerificationStatus.UNVERIFIED: 1,
        VerificationStatus.FAILED: 0,
    }
    best: dict[str, EvidenceItem] = {}
    order: list[str] = []
    for item in items:
        # A citation restated in other words is the SAME evidence line:
        # collapse on normalised citation identity per topic as well as
        # on the exact content fingerprint (hardening, Worker A Bug 3).
        cite_key = " ".join(item.citation.lower().split())
        key = "cite:" + hashlib.sha256(
            f"{item.topic_key}\n{cite_key}".encode("utf-8")
        ).hexdigest()
        if key not in best:
            best[key] = item
            order.append(key)
        elif rank[item.status] > rank[best[key].status]:
            best[key] = item
    return [best[key] for key in order]


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the cite-or-decline gate for one topic.

    Attributes:
        passed: True when the gate is satisfied and an answer may be
            issued.
        topic_key: The topic the decision is about.
        verified_supporting: Verified supporting items that counted.
        verified_refuting: Verified refuting items found.  These do
            not block an answer by themselves, but they are surfaced
            so the answer can disclose disagreement.
        independent_groups: Independence groups among the verified
            supporting items.
        reasons: Human readable reasons, explaining either why the
            gate passed or exactly which requirement failed.
        warnings: Non-blocking problems, for example items that
            referenced an unregistered source.
    """

    passed: bool
    topic_key: str
    verified_supporting: tuple[EvidenceItem, ...] = ()
    verified_refuting: tuple[EvidenceItem, ...] = ()
    independent_groups: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def verified_count(self) -> int:
        return len(self.verified_supporting)

    @property
    def independent_source_count(self) -> int:
        return len(self.independent_groups)


def evaluate_gate(
    items: Sequence[EvidenceItem],
    topic_key: str,
    registry: Optional[SourceRegistry] = None,
    min_verified: int = MIN_VERIFIED_ITEMS,
    min_independent: int = MIN_INDEPENDENT_SOURCES,
) -> GateDecision:
    """Apply the cite-or-decline gate to the evidence for one topic.

    The gate passes only when, after deduplication:

    * at least ``min_verified`` items about ``topic_key`` are
      verified and support the claim, and
    * those items come from at least ``min_independent`` distinct
      independence groups in the registry.

    Refuting verified items are reported but do not silently cancel
    supporting ones; surfacing disagreement is the caller's job.

    Args:
        items: Candidate evidence items, any topic, any status.
        topic_key: Only items with this topic key are considered.
        registry: Source registry used to resolve independence groups.
            Defaults to :func:`default_source_registry`.
        min_verified: Override for the minimum verified item count.
        min_independent: Override for the minimum independence count.

    Returns:
        A :class:`GateDecision`.  When it does not pass, ``reasons``
        names each failed requirement precisely, so a decline can say
        what is missing instead of merely saying "not enough data".
    """
    if registry is None:
        registry = default_source_registry()
    if min_verified < 1:
        raise ValueError("min_verified must be at least 1")
    if min_independent < 1:
        raise ValueError("min_independent must be at least 1")

    relevant = [item for item in deduplicate(list(items)) if item.topic_key == topic_key]
    supporting_all = tuple(
        item
        for item in relevant
        if item.is_verified and item.direction is SupportDirection.SUPPORTS
    )
    # Hardening (Worker A Bug 1): an unregistered source cannot vouch
    # for evidence. Only items whose source_id is in the registry
    # count toward the gate; unregistered items are warned about and
    # excluded from the count and from any issued answer.
    supporting = tuple(
        item for item in supporting_all if item.source_id in registry
    )
    unregistered_count = len(supporting_all) - len(supporting)
    refuting = tuple(
        item
        for item in relevant
        if item.is_verified and item.direction is SupportDirection.REFUTES
    )
    groups = tuple(
        sorted({registry.independence_group_of(item.source_id) for item in supporting})
    )

    warnings: list[str] = []
    for item in relevant:
        if item.source_id not in registry:
            warnings.append(
                f"item {item.item_id} references unregistered source "
                f"{item.source_id!r}; unregistered sources do not "
                "count toward the gate"
            )
    if unregistered_count:
        warnings.append(
            f"{unregistered_count} verified supporting item(s) were "
            "excluded because their source is not registered"
        )
    failed_count = sum(1 for item in relevant if item.is_failed)
    if failed_count:
        warnings.append(
            f"{failed_count} item(s) failed verification and were excluded"
        )
    if refuting:
        warnings.append(
            f"{len(refuting)} verified item(s) refute the claim; "
            "disagreement must be disclosed in any answer"
        )

    reasons: list[str] = []
    passed = True
    if len(supporting) < min_verified:
        passed = False
        reasons.append(
            f"only {len(supporting)} verified supporting item(s); "
            f"the gate requires at least {min_verified}"
        )
    if len(groups) < min_independent:
        passed = False
        reasons.append(
            f"verified supporting items come from {len(groups)} "
            f"independent source group(s); the gate requires at "
            f"least {min_independent}"
        )
    # Hardening (Worker A Bug 5): verified refuting evidence that
    # matches or outweighs the supporting evidence blocks an answer;
    # the disagreement must be resolved, not merely disclosed.
    if refuting and len(refuting) >= len(supporting):
        passed = False
        reasons.append(
            f"{len(refuting)} verified item(s) refute the claim, "
            f"matching or outweighing the {len(supporting)} "
            "supporting item(s); disagreement is unresolved"
        )
    if passed:
        reasons.append(
            f"gate satisfied: {len(supporting)} verified supporting "
            f"item(s) from {len(groups)} independent source group(s)"
        )

    return GateDecision(
        passed=passed,
        topic_key=topic_key,
        verified_supporting=supporting,
        verified_refuting=refuting,
        independent_groups=groups,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def summarise_items(items: Sequence[EvidenceItem]) -> Mapping[str, int]:
    """Count items by verification status, for diagnostics and tests."""
    counts = {status.value: 0 for status in VerificationStatus}
    for item in items:
        counts[item.status.value] += 1
    return counts
