"""Maximum credible allele frequency (Whiffin et al. 2017).

If a variant causes a disease, there is a ceiling on how common it
can be in the general population.  A variant more common than that
ceiling cannot be the cause, however damaging it looks in a
predictor.  Whiffin N, et al. "Using high-resolution variant
frequencies to empower clinical genome interpretation." *Genetics in
Medicine* 20:171-180 (2017/2018), turned that idea into a calculator
that the ExAC and gnomAD era made practical, and into the statistical
basis of the ACMG frequency criteria BA1, BS1, and PM2.

The derivation used here is the genotype-frequency form of the
Whiffin framework.  Define:

* ``prevalence``: proportion of the population with the disease.
* ``genetic_heterogeneity`` (maximum genetic contribution): the
  largest proportion of all cases attributable to one gene.
* ``allelic_heterogeneity`` (maximum allelic contribution): the
  largest proportion of pathogenic alleles in that gene attributable
  to one variant.
* ``penetrance``: probability that a person with the disease
  genotype develops the disease.

**Autosomal dominant.**  Cases attributable to the variant equal the
heterozygote frequency (about ``2q`` for rare ``q``) times
penetrance.  Setting attributable cases equal to
``prevalence * genetic_heterogeneity * allelic_heterogeneity`` and
solving for ``q``:

    q_max = prevalence * genetic_heterogeneity
            * allelic_heterogeneity / (2 * penetrance)

**Autosomal recessive.**  Cases attributable to the variant (counting
both homozygotes for the variant and, in the conservative Whiffin
treatment of a single variant's contribution, its share of disease
genotypes) scale with ``q^2`` in the simplest single-variant model:

    q_max = sqrt(prevalence * genetic_heterogeneity
                 * allelic_heterogeneity / penetrance)

These are deliberately transparent implementations of the framework,
with every input explicit.  Production use should cross-check
against the published allele frequency application and current
ClinGen SVI guidance for the specific gene and disease, including
their treatment of compound heterozygosity for recessive disease.

Threshold evaluation follows the logic of the criteria:

* **BA1**: observed allele frequency above 0.05 (5 percent) is
  stand-alone benign for most Mendelian contexts (with known gene
  specific exceptions, which this module does not silently apply).
* **BS1**: observed frequency greater than the maximum credible
  frequency for the disorder.
* **PM2**: observed frequency at or below the maximum credible
  frequency, or absent, is consistent with (never proof of)
  pathogenicity at supporting-to-moderate weight depending on the
  applicable specification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "Inheritance",
    "BA1_THRESHOLD",
    "DiseaseModel",
    "max_credible_af",
    "max_credible_allele_count",
    "FrequencyEvaluation",
    "evaluate_frequency",
    "credible_interval_excludes",
]

#: Default BA1 allele frequency threshold (Richards et al. 2015).
BA1_THRESHOLD = 0.05


class Inheritance(str, Enum):
    """Mode of inheritance for the disease model."""

    AUTOSOMAL_DOMINANT = "autosomal_dominant"
    AUTOSOMAL_RECESSIVE = "autosomal_recessive"


def _validate_probability(name: str, value: float, *, allow_zero: bool) -> None:
    if value is None or math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be a finite number")
    lower_ok = value >= 0 if allow_zero else value > 0
    if not lower_ok or value > 1:
        bound = "0 <= x <= 1" if allow_zero else "0 < x <= 1"
        raise ValueError(f"{name} must satisfy {bound}, got {value}")


@dataclass(frozen=True)
class DiseaseModel:
    """Inputs describing a disease for the Whiffin calculation.

    All proportions are fractions, not percentages: a prevalence of
    1 in 500 is ``0.002``.

    Attributes:
        prevalence: Proportion of the population affected.
        inheritance: Dominant or recessive.
        penetrance: Probability the disease genotype causes disease.
        genetic_heterogeneity: Maximum proportion of cases due to
            one gene (1.0 when a single gene explains the disease).
        allelic_heterogeneity: Maximum proportion of pathogenic
            alleles in that gene due to one variant.
        name: Optional disease label, carried into reports.
    """

    prevalence: float
    inheritance: Inheritance
    penetrance: float = 1.0
    genetic_heterogeneity: float = 1.0
    allelic_heterogeneity: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        _validate_probability("prevalence", self.prevalence, allow_zero=False)
        _validate_probability("penetrance", self.penetrance, allow_zero=False)
        _validate_probability(
            "genetic_heterogeneity", self.genetic_heterogeneity, allow_zero=False
        )
        _validate_probability(
            "allelic_heterogeneity", self.allelic_heterogeneity, allow_zero=False
        )


def max_credible_af(model: DiseaseModel) -> float:
    """Compute the maximum credible population allele frequency.

    Implements the genotype-frequency form of the Whiffin et al.
    framework described in the module docstring:

    * dominant: ``prevalence * gh * ah / (2 * penetrance)``
    * recessive: ``sqrt(prevalence * gh * ah / penetrance)``

    where ``gh`` and ``ah`` are the genetic and allelic
    heterogeneity maxima.  The result is a frequency between 0 and 1
    (capped at 1.0 for degenerate inputs, which can arise for very
    common recessive conditions under the simple model).
    """
    attributable = (
        model.prevalence
        * model.genetic_heterogeneity
        * model.allelic_heterogeneity
    )
    if model.inheritance is Inheritance.AUTOSOMAL_DOMINANT:
        value = attributable / (2.0 * model.penetrance)
    elif model.inheritance is Inheritance.AUTOSOMAL_RECESSIVE:
        value = math.sqrt(attributable / model.penetrance)
    else:  # pragma: no cover - enum is exhaustive
        raise ValueError(f"unsupported inheritance: {model.inheritance}")
    return min(value, 1.0)


def max_credible_allele_count(model: DiseaseModel, allele_number: int) -> float:
    """Maximum credible allele count in a sample of ``allele_number``.

    This is the form used when filtering a real data set: a variant
    seen more often than this count, in a sample of this size, is too
    common for the disease model.  Returns a float; callers comparing
    against integer counts should treat any observed count strictly
    greater than this value as exceeding the ceiling.
    """
    if allele_number <= 0:
        raise ValueError("allele_number must be positive")
    return max_credible_af(model) * allele_number


def credible_interval_excludes(
    observed_count: int,
    allele_number: int,
    max_af: float,
    *,
    z: float = 1.96,
) -> bool:
    """Conservative check: is the observed frequency's lower bound high?

    Uses a simple normal approximation to the binomial sampling
    interval for the observed frequency.  Returns True when even the
    lower end of the interval exceeds ``max_af``, meaning sampling
    noise alone is an unlikely explanation for the excess.  This is a
    screening helper, not a substitute for the exact Poisson/binomial
    treatment in the published allele frequency application.
    """
    if allele_number <= 0:
        raise ValueError("allele_number must be positive")
    if observed_count < 0 or observed_count > allele_number:
        raise ValueError("observed_count must be between 0 and allele_number")
    if not 0.0 <= max_af <= 1.0:
        raise ValueError("max_af must be between 0 and 1")
    p = observed_count / allele_number
    if p == 0:
        return False
    se = math.sqrt(p * (1.0 - p) / allele_number)
    return (p - z * se) > max_af


@dataclass(frozen=True)
class FrequencyEvaluation:
    """Result of evaluating an observed frequency against a model.

    Attributes:
        observed_af: Observed allele frequency (0 to 1).
        max_credible_af: Ceiling from :func:`max_credible_af`.
        ba1_met: Observed frequency exceeds the BA1 threshold.
        bs1_met: Observed frequency exceeds the credible ceiling.
        pm2_consistent: Observed frequency is at or below the
            credible ceiling (consistent with rarity; not proof).
        absent: The variant was absent (observed count zero), when
            counts were supplied.
        explanation: Plain language summary naming each outcome.
    """

    observed_af: float
    max_credible_af: float
    ba1_met: bool
    bs1_met: bool
    pm2_consistent: bool
    absent: bool = False
    explanation: str = ""


def evaluate_frequency(
    observed_af: Optional[float] = None,
    *,
    observed_count: Optional[int] = None,
    allele_number: Optional[int] = None,
    model: DiseaseModel,
    ba1_threshold: float = BA1_THRESHOLD,
) -> FrequencyEvaluation:
    """Evaluate an observed allele frequency against a disease model.

    Supply either ``observed_af`` directly, or ``observed_count``
    with ``allele_number``.  An ``observed_af`` of ``None`` with no
    counts means the variant was not observed at all, which is
    treated as an observed frequency of 0 and flagged ``absent``.

    The three booleans correspond to the ACMG frequency criteria
    BA1, BS1, and PM2 as described in the module docstring.  They are
    reported, not decided: applying a criterion remains a curator
    judgement under current gene specific guidance.
    """
    absent = False
    if observed_af is None:
        if observed_count is not None and allele_number is not None:
            if allele_number <= 0:
                raise ValueError("allele_number must be positive")
            if observed_count < 0 or observed_count > allele_number:
                raise ValueError("observed_count out of range")
            observed = observed_count / allele_number
            absent = observed_count == 0
        else:
            observed = 0.0
            absent = True
    else:
        if observed_af < 0 or observed_af > 1 or math.isnan(observed_af):
            raise ValueError("observed_af must be between 0 and 1")
        observed = float(observed_af)
        if observed_count is not None:
            absent = observed_count == 0

    ceiling = max_credible_af(model)
    ba1 = observed > ba1_threshold
    bs1 = observed > ceiling
    pm2 = observed <= ceiling

    parts: list[str] = []
    parts.append(
        f"observed allele frequency {observed:.6g}; maximum "
        f"credible frequency for this disease model {ceiling:.6g}"
    )
    if ba1:
        parts.append(
            f"observed frequency exceeds the BA1 threshold "
            f"({ba1_threshold:.3g}): stand-alone benign evidence"
        )
    if bs1:
        parts.append(
            "observed frequency exceeds the maximum credible "
            "frequency: BS1 (frequency greater than expected for "
            "the disorder) is met"
        )
    if pm2:
        if absent:
            parts.append(
                "variant absent from the population sample: "
                "consistent with PM2 rarity, subject to coverage "
                "of the site in the data set"
            )
        else:
            parts.append(
                "observed frequency is at or below the maximum "
                "credible frequency: consistent with PM2 rarity"
            )
    explanation = "; ".join(parts)
    return FrequencyEvaluation(
        observed_af=observed,
        max_credible_af=ceiling,
        ba1_met=ba1,
        bs1_met=bs1,
        pm2_consistent=pm2,
        absent=absent,
        explanation=explanation,
    )
