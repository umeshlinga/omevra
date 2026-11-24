"""ACMG/AMP germline variant classification.

Two published frameworks are implemented, side by side, because they
answer the same question in complementary ways:

1. **Richards et al. 2015** combining rules.  Richards S, et al.
   "Standards and guidelines for the interpretation of sequence
   variants: a joint consensus recommendation of the American College
   of Medical Genetics and Genomics and the Association for Molecular
   Pathology." *Genetics in Medicine* 17:405-424.  Criteria are
   combined by explicit rules (for example, one very strong plus one
   strong criterion is Pathogenic).

2. **Tavtigian et al. 2018 Bayesian points.**  Tavtigian SV, et al.
   "Modeling the ACMG/AMP variant classification guidelines as a
   Bayesian classification framework." *Genetics in Medicine*
   20:1054-1060.  Each criterion contributes points on an exponential
   scale: pathogenic very strong = 8, strong = 4, moderate = 2,
   supporting = 1; benign stand-alone = -8, strong = -4,
   supporting = -1.  Totals map to categories: at least 10
   Pathogenic, 6 to 9 Likely Pathogenic, 0 to 5 VUS, -1 to -6 Likely
   Benign, at most -7 Benign.

This module scores criteria that a curator supplies.  It does not
decide whether a criterion applies to a particular variant: that
judgement, and any strength modification (for example applying PS3
at moderate strength following ClinGen Sequence Variant
Interpretation guidance), belongs to the curator, who can state the
strength explicitly on each :class:`CriterionAssessment`.

Nothing here is a clinical classification service.  Real
classifications require current ClinGen specifications, gene specific
rules, and expert review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

__all__ = [
    "Direction",
    "Strength",
    "POINTS",
    "Criterion",
    "CriterionAssessment",
    "Classification",
    "points_for",
    "total_points",
    "classify_by_points",
    "classify_by_richards_rules",
    "classify",
    "ClassificationReport",
]


class Direction(str, Enum):
    """Whether a criterion argues toward pathogenic or benign."""

    PATHOGENIC = "pathogenic"
    BENIGN = "benign"


class Strength(str, Enum):
    """Criterion strengths, weakest to strongest within a direction."""

    SUPPORTING = "supporting"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    STAND_ALONE = "stand_alone"  # benign only (BA)


#: Tavtigian et al. point values, keyed by (direction, strength).
POINTS: dict[tuple[Direction, Strength], int] = {
    (Direction.PATHOGENIC, Strength.VERY_STRONG): 8,
    (Direction.PATHOGENIC, Strength.STRONG): 4,
    (Direction.PATHOGENIC, Strength.MODERATE): 2,
    (Direction.PATHOGENIC, Strength.SUPPORTING): 1,
    (Direction.BENIGN, Strength.STAND_ALONE): -8,
    (Direction.BENIGN, Strength.STRONG): -4,
    (Direction.BENIGN, Strength.SUPPORTING): -1,
    # Benign moderate is not a standard ACMG level; it is included at
    # -2 for symmetry when a curator applies a modified strength, and
    # is labelled as a modification wherever it is used.
    (Direction.BENIGN, Strength.MODERATE): -2,
    # Pathogenic stand-alone is not a standard level either; a single
    # criterion never classifies a variant alone on the pathogenic
    # side.  It is deliberately absent so misuse raises instead of
    # silently scoring.
}


def points_for(direction: Direction, strength: Strength) -> int:
    """Return Tavtigian points for a direction and strength.

    Raises:
        ValueError: for combinations with no published point value,
            notably pathogenic stand-alone.
    """
    try:
        return POINTS[(direction, strength)]
    except KeyError as exc:
        raise ValueError(
            f"no Tavtigian point value for {direction.value} "
            f"{strength.value}"
        ) from exc


class Criterion(str, Enum):
    """The ACMG/AMP criteria, with their default direction/strength.

    Each member's value is its code.  Default strength is the one in
    Richards et al. 2015; curators may override it per assessment.
    """

    # Pathogenic
    PVS1 = "PVS1"
    PS1 = "PS1"
    PS2 = "PS2"
    PS3 = "PS3"
    PS4 = "PS4"
    PM1 = "PM1"
    PM2 = "PM2"
    PM3 = "PM3"
    PM4 = "PM4"
    PM5 = "PM5"
    PM6 = "PM6"
    PP1 = "PP1"
    PP2 = "PP2"
    PP3 = "PP3"
    PP4 = "PP4"
    PP5 = "PP5"
    # Benign
    BA1 = "BA1"
    BS1 = "BS1"
    BS2 = "BS2"
    BS3 = "BS3"
    BS4 = "BS4"
    BP1 = "BP1"
    BP2 = "BP2"
    BP3 = "BP3"
    BP4 = "BP4"
    BP5 = "BP5"
    BP6 = "BP6"
    BP7 = "BP7"

    @property
    def direction(self) -> Direction:
        if self.value.startswith("B"):
            return Direction.BENIGN
        return Direction.PATHOGENIC

    @property
    def default_strength(self) -> Strength:
        code = self.value
        if code == "PVS1":
            return Strength.VERY_STRONG
        if code.startswith("PS"):
            return Strength.STRONG
        if code.startswith("PM"):
            return Strength.MODERATE
        if code.startswith("PP"):
            return Strength.SUPPORTING
        if code == "BA1":
            return Strength.STAND_ALONE
        if code.startswith("BS"):
            return Strength.STRONG
        if code.startswith("BP"):
            return Strength.SUPPORTING
        raise ValueError(f"unhandled criterion code: {code}")  # pragma: no cover

    @property
    def description(self) -> str:
        return _CRITERION_DESCRIPTIONS[self]


_CRITERION_DESCRIPTIONS: dict[Criterion, str] = {
    Criterion.PVS1: (
        "Null variant (nonsense, frameshift, canonical splice, "
        "initiation codon, exon deletion) in a gene where loss of "
        "function is a known mechanism of disease."
    ),
    Criterion.PS1: (
        "Same amino acid change as a previously established "
        "pathogenic variant, regardless of nucleotide change."
    ),
    Criterion.PS2: (
        "De novo in a patient with the disease and no family "
        "history, with maternity and paternity confirmed."
    ),
    Criterion.PS3: (
        "Well established functional studies supportive of a "
        "damaging effect on the gene or gene product."
    ),
    Criterion.PS4: (
        "Prevalence of the variant in affected individuals is "
        "significantly increased compared with controls."
    ),
    Criterion.PM1: (
        "Located in a mutational hot spot or critical functional "
        "domain without benign variation."
    ),
    Criterion.PM2: (
        "Absent from controls, or at extremely low frequency, in "
        "population databases for a recessive condition."
    ),
    Criterion.PM3: (
        "For recessive disorders, detected in trans with a "
        "pathogenic variant."
    ),
    Criterion.PM4: (
        "Protein length change from an in-frame indel in a "
        "non-repeat region, or a stop-loss variant."
    ),
    Criterion.PM5: (
        "Novel missense at a residue where a different pathogenic "
        "missense has been established."
    ),
    Criterion.PM6: (
        "Assumed de novo, without confirmation of maternity and "
        "paternity."
    ),
    Criterion.PP1: "Cosegregation with disease in multiple affected family members.",
    Criterion.PP2: (
        "Missense in a gene with a low rate of benign missense "
        "variation, where missense is a common disease mechanism."
    ),
    Criterion.PP3: (
        "Multiple lines of computational evidence support a "
        "deleterious effect on the gene or product."
    ),
    Criterion.PP4: (
        "Patient phenotype or family history is highly specific for "
        "a disease with a single genetic aetiology."
    ),
    Criterion.PP5: (
        "Reputable source reports the variant as pathogenic, with "
        "evidence not available for independent evaluation. "
        "(Deprecated by later ClinGen guidance; kept for parity "
        "with the 2015 framework.)"
    ),
    Criterion.BA1: (
        "Allele frequency above 5 percent in a major population "
        "database: stand-alone benign."
    ),
    Criterion.BS1: (
        "Allele frequency greater than expected for the disorder."
    ),
    Criterion.BS2: (
        "Observed in a healthy adult for a fully penetrant disorder "
        "expected at an early age."
    ),
    Criterion.BS3: (
        "Well established functional studies show no damaging "
        "effect on protein function or splicing."
    ),
    Criterion.BS4: "Lack of segregation in affected family members.",
    Criterion.BP1: (
        "Missense in a gene for which primarily truncating variants "
        "cause disease."
    ),
    Criterion.BP2: (
        "Observed in trans with a pathogenic variant for a fully "
        "penetrant dominant disorder, or in cis in any inheritance "
        "pattern."
    ),
    Criterion.BP3: (
        "In-frame indel in a repetitive region without known "
        "function."
    ),
    Criterion.BP4: (
        "Multiple lines of computational evidence suggest no impact "
        "on the gene or product."
    ),
    Criterion.BP5: (
        "Variant found in a case with an alternate molecular basis "
        "for disease."
    ),
    Criterion.BP6: (
        "Reputable source reports the variant as benign, with "
        "evidence not available for independent evaluation. "
        "(Deprecated by later ClinGen guidance; kept for parity.)"
    ),
    Criterion.BP7: (
        "Synonymous variant for which splicing prediction tools "
        "predict no impact and the nucleotide is not highly "
        "conserved."
    ),
}


@dataclass(frozen=True)
class CriterionAssessment:
    """One criterion applied to a variant, at a stated strength.

    Attributes:
        criterion: Which criterion.
        strength: Strength applied.  Defaults to the criterion's
            published default strength.  Any other value is a
            *strength modification* and is flagged as such.
        rationale: Why the curator applied it.  Optional in code,
            expected in practice; a classification without rationales
            is an opinion, not an audit trail.
    """

    criterion: Criterion
    strength: Optional[Strength] = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.strength is not None:
            # Validate the combination has a point value at all.
            points_for(self.criterion.direction, self.applied_strength)

    @property
    def applied_strength(self) -> Strength:
        return self.strength if self.strength is not None else self.criterion.default_strength

    @property
    def is_modified(self) -> bool:
        """True when the applied strength differs from the default."""
        return self.applied_strength is not self.criterion.default_strength

    @property
    def points(self) -> int:
        return points_for(self.criterion.direction, self.applied_strength)


class Classification(str, Enum):
    """The five ACMG/AMP classification categories."""

    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    VUS = "Variant of Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"


def total_points(assessments: Iterable[CriterionAssessment]) -> int:
    """Sum Tavtigian points over a set of assessments.

    Duplicate criteria are rejected: applying the same criterion
    twice double counts one line of evidence, which both published
    frameworks forbid.
    """
    seen: set[Criterion] = set()
    total = 0
    for assessment in assessments:
        if assessment.criterion in seen:
            raise ValueError(
                f"criterion {assessment.criterion.value} applied twice"
            )
        seen.add(assessment.criterion)
        total += assessment.points
    return total


def classify_by_points(total: int) -> Classification:
    """Map a Tavtigian point total to a category.

    Thresholds (Tavtigian et al. 2018): at least 10 Pathogenic;
    6 to 9 Likely Pathogenic; 0 to 5 VUS; -1 to -6 Likely Benign;
    at most -7 Benign.
    """
    if total >= 10:
        return Classification.PATHOGENIC
    if total >= 6:
        return Classification.LIKELY_PATHOGENIC
    if total >= 0:
        return Classification.VUS
    if total >= -6:
        return Classification.LIKELY_BENIGN
    return Classification.BENIGN


def _counts(
    assessments: list[CriterionAssessment],
) -> dict[tuple[Direction, Strength], int]:
    counts: dict[tuple[Direction, Strength], int] = {}
    for assessment in assessments:
        key = (assessment.criterion.direction, assessment.applied_strength)
        counts[key] = counts.get(key, 0) + 1
    return counts


def classify_by_richards_rules(
    assessments: Iterable[CriterionAssessment],
) -> Classification:
    """Apply the Richards et al. 2015 combining rules.

    The rules are evaluated in the published order: Pathogenic
    combinations first, then Likely Pathogenic, then Benign, then
    Likely Benign.  When criteria argue in both directions and no
    benign or pathogenic combination is met, the result is VUS,
    which is how the 2015 guidelines treat conflicting evidence.
    Strength modifications are honoured: a criterion applied at
    moderate strength counts as a moderate criterion here.
    """
    items = list(assessments)
    total_points(items)  # reuses duplicate validation
    counts = _counts(items)

    def n(direction: Direction, strength: Strength) -> int:
        return counts.get((direction, strength), 0)

    pvs = n(Direction.PATHOGENIC, Strength.VERY_STRONG)
    ps = n(Direction.PATHOGENIC, Strength.STRONG)
    pm = n(Direction.PATHOGENIC, Strength.MODERATE)
    pp = n(Direction.PATHOGENIC, Strength.SUPPORTING)
    ba = n(Direction.BENIGN, Strength.STAND_ALONE)
    bs = n(Direction.BENIGN, Strength.STRONG)
    bp = n(Direction.BENIGN, Strength.SUPPORTING)

    # Pathogenic combinations (Richards Table 5).
    if pvs >= 1 and ps >= 1:
        return Classification.PATHOGENIC
    if pvs >= 1 and pm >= 2:
        return Classification.PATHOGENIC
    if pvs >= 1 and pm >= 1 and pp >= 1:
        return Classification.PATHOGENIC
    if pvs >= 1 and pp >= 2:
        return Classification.PATHOGENIC
    if ps >= 2:
        return Classification.PATHOGENIC
    if ps >= 1 and pm >= 3:
        return Classification.PATHOGENIC
    if ps >= 1 and pm >= 2 and pp >= 2:
        return Classification.PATHOGENIC
    if ps >= 1 and pm >= 1 and pp >= 4:
        return Classification.PATHOGENIC

    # Benign combinations are checked before Likely Pathogenic so a
    # stand-alone benign criterion is never outvoted by accumulation
    # of weak pathogenic criteria; the 2015 text treats BA1 as
    # stand-alone.
    if ba >= 1:
        return Classification.BENIGN
    if bs >= 2:
        return Classification.BENIGN

    # Likely Pathogenic combinations.
    if pvs >= 1 and pm >= 1:
        return Classification.LIKELY_PATHOGENIC
    if ps >= 1 and 1 <= pm <= 2:
        return Classification.LIKELY_PATHOGENIC
    if ps >= 1 and pp >= 2:
        return Classification.LIKELY_PATHOGENIC
    if pm >= 3:
        return Classification.LIKELY_PATHOGENIC
    if pm >= 2 and pp >= 2:
        return Classification.LIKELY_PATHOGENIC
    if pm >= 1 and pp >= 4:
        return Classification.LIKELY_PATHOGENIC

    # Likely Benign combinations.
    if bs >= 1 and bp >= 1:
        return Classification.LIKELY_BENIGN
    if bp >= 2:
        return Classification.LIKELY_BENIGN

    return Classification.VUS


@dataclass(frozen=True)
class ClassificationReport:
    """Both framework outcomes for one set of assessments.

    Attributes:
        points_total: Tavtigian point total.
        points_classification: Category from the points system.
        richards_classification: Category from the combining rules.
        agree: True when both frameworks give the same category.
            Disagreement is informative, not an error: the points
            system is continuous where the rules are categorical,
            and boundary cases legitimately differ.
        assessments: The assessments scored, in input order.
        notes: Explanations, including any strength modifications.
    """

    points_total: int
    points_classification: Classification
    richards_classification: Classification
    agree: bool
    assessments: tuple[CriterionAssessment, ...]
    notes: tuple[str, ...] = ()


def classify(assessments: Iterable[CriterionAssessment]) -> ClassificationReport:
    """Score a set of assessments under both published frameworks."""
    items = list(assessments)
    total = total_points(items)
    by_points = classify_by_points(total)
    by_rules = classify_by_richards_rules(items)
    notes: list[str] = []
    for assessment in items:
        if assessment.is_modified:
            notes.append(
                f"{assessment.criterion.value} applied at modified "
                f"strength {assessment.applied_strength.value} "
                f"(default {assessment.criterion.default_strength.value})"
            )
    if by_points is not by_rules:
        notes.append(
            f"frameworks disagree: points give {by_points.value}, "
            f"combining rules give {by_rules.value}; boundary cases "
            "should be resolved by expert review with current "
            "ClinGen specifications"
        )
    has_path = any(a.criterion.direction is Direction.PATHOGENIC for a in items)
    has_benign = any(a.criterion.direction is Direction.BENIGN for a in items)
    if has_path and has_benign:
        notes.append(
            "criteria argue in both directions; conflicting evidence "
            "requires explicit resolution before reporting"
        )
    return ClassificationReport(
        points_total=total,
        points_classification=by_points,
        richards_classification=by_rules,
        agree=by_points is by_rules,
        assessments=tuple(items),
        notes=tuple(notes),
    )
