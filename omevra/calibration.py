"""Classifier evaluation and calibration, from scratch.

A probability is only useful if it means what it says: among cases
scored 0.8, about 80 percent should be positive.  This module measures
both halves of that promise, discrimination and calibration, using
only the standard library:

* **AUROC** by the rank (Mann-Whitney U) method, with average ranks
  for tied scores.  AUROC is the probability that a randomly chosen
  positive is scored above a randomly chosen negative.
* **Brier score**: mean squared error of the predicted probabilities.
  Lower is better; a constant forecast of the base rate scores
  ``p(1-p)``.
* **Expected Calibration Error (ECE)**: predictions are placed in
  equal-width bins over [0, 1]; ECE is the count-weighted mean of
  |observed frequency - mean predicted probability| per bin.
* **Calibration table**: the per-bin numbers behind the ECE, because
  a single summary number hides where a model is miscalibrated.
* **Log loss**: mean negative log likelihood, with probabilities
  clipped away from 0 and 1 so a single confident error is heavily
  penalised but finite.

Definitions follow the standard treatment in Niculescu-Mizil and
Caruana (2005) and the calibration literature generally.  Every
function validates its inputs and refuses mismatched lengths, empty
data, labels outside {0, 1}, and scores outside [0, 1], because
evaluation code that silently accepts bad input produces confident
nonsense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "auroc",
    "brier_score",
    "log_loss",
    "CalibrationBin",
    "CalibrationTable",
    "calibration_table",
    "expected_calibration_error",
    "maximum_calibration_error",
    "EvaluationReport",
    "evaluate",
]


def _validate(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[list[int], list[float]]:
    if len(labels) != len(scores):
        raise ValueError(
            f"labels and scores must have the same length "
            f"({len(labels)} != {len(scores)})"
        )
    if len(labels) == 0:
        raise ValueError("at least one observation is required")
    clean_labels: list[int] = []
    for label in labels:
        if label not in (0, 1):
            raise ValueError(f"labels must be 0 or 1, got {label!r}")
        clean_labels.append(int(label))
    clean_scores: list[float] = []
    for score in scores:
        try:
            value = float(score)
        except (TypeError, ValueError):
            raise ValueError(f"scores must be real numbers, got {score!r}") from None
        if math.isnan(value) or math.isinf(value):
            raise ValueError("scores must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"scores must lie in [0, 1], got {value}")
        clean_scores.append(value)
    return clean_labels, clean_scores


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve by the rank method.

    Scores are ranked ascending with average ranks for ties; AUROC is
    computed from the Mann-Whitney U statistic:

        AUROC = (sum of positive ranks - n_pos (n_pos + 1) / 2)
                / (n_pos * n_neg)

    Raises:
        ValueError: if all labels are the same class, because AUROC
            is undefined without both classes present.
    """
    clean_labels, clean_scores = _validate(labels, scores)
    n_pos = sum(clean_labels)
    n_neg = len(clean_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC is undefined when only one class is present")

    order = sorted(range(len(clean_scores)), key=lambda i: clean_scores[i])
    ranks = [0.0] * len(clean_scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and clean_scores[order[end + 1]] == clean_scores[order[index]]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2.0  # 1-based ranks
        for k in range(index, end + 1):
            ranks[order[k]] = average_rank
        index = end + 1

    rank_sum_pos = sum(
        rank for rank, label in zip(ranks, clean_labels) if label == 1
    )
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_statistic / (n_pos * n_neg)


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Mean squared error of probabilistic predictions."""
    clean_labels, clean_scores = _validate(labels, scores)
    total = sum(
        (score - label) ** 2
        for label, score in zip(clean_labels, clean_scores)
    )
    return total / len(clean_labels)


def log_loss(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    epsilon: float = 1e-15,
) -> float:
    """Mean negative log likelihood, with clipping at ``epsilon``."""
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must lie in (0, 0.5)")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    clean_labels, clean_scores = _validate(labels, scores)
    total = 0.0
    for label, score in zip(clean_labels, clean_scores):
        clipped = min(max(score, epsilon), 1.0 - epsilon)
        total += -(
            label * math.log(clipped)
            + (1 - label) * math.log(1.0 - clipped)
        )
    return total / len(clean_labels)


@dataclass(frozen=True)
class CalibrationBin:
    """One bin of a calibration table.

    Attributes:
        lower: Inclusive lower score bound of the bin.
        upper: Upper score bound (exclusive, except the last bin
            which includes 1.0).
        count: Number of observations in the bin.
        mean_predicted: Mean predicted probability in the bin.
        observed_frequency: Fraction of positives in the bin.
        gap: ``observed_frequency - mean_predicted``; positive means
            the model under-predicts in this bin.
    """

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_frequency: float
    gap: float


@dataclass(frozen=True)
class CalibrationTable:
    """A full calibration table plus its summary errors."""

    bins: tuple[CalibrationBin, ...]
    n_bins: int
    total: int
    ece: float
    mce: float

    def non_empty_bins(self) -> tuple[CalibrationBin, ...]:
        return tuple(b for b in self.bins if b.count > 0)

    def __str__(self) -> str:
        lines = [
            "bin            n   mean_pred  observed     gap",
        ]
        for b in self.bins:
            lines.append(
                f"[{b.lower:.2f},{b.upper:.2f}) {b.count:5d} "
                f"{b.mean_predicted:9.3f} {b.observed_frequency:9.3f} "
                f"{b.gap:+7.3f}"
            )
        lines.append(f"ECE = {self.ece:.4f}   MCE = {self.mce:.4f}   n = {self.total}")
        return "\n".join(lines)


def calibration_table(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> CalibrationTable:
    """Build an equal-width calibration table over [0, 1].

    A score of exactly 1.0 falls in the last bin.  Empty bins are
    reported with zero count and zero gap, and contribute nothing to
    the ECE.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int):
        raise ValueError("n_bins must be an integer")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if n_bins > 10000:
        raise ValueError("n_bins must be at most 10000 (resource cap)")
    clean_labels, clean_scores = _validate(labels, scores)
    counts = [0] * n_bins
    score_sums = [0.0] * n_bins
    pos_sums = [0] * n_bins
    for label, score in zip(clean_labels, clean_scores):
        index = int(score * n_bins)
        if index >= n_bins:
            index = n_bins - 1
        counts[index] += 1
        score_sums[index] += score
        pos_sums[index] += label

    bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    total = len(clean_labels)
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if counts[i] == 0:
            bins.append(CalibrationBin(lower, upper, 0, 0.0, 0.0, 0.0))
            continue
        mean_predicted = score_sums[i] / counts[i]
        observed = pos_sums[i] / counts[i]
        gap = observed - mean_predicted
        bins.append(
            CalibrationBin(lower, upper, counts[i], mean_predicted, observed, gap)
        )
        ece += (counts[i] / total) * abs(gap)
        mce = max(mce, abs(gap))
    return CalibrationTable(
        bins=tuple(bins), n_bins=n_bins, total=total, ece=ece, mce=mce
    )


def expected_calibration_error(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins."""
    return calibration_table(labels, scores, n_bins=n_bins).ece


def maximum_calibration_error(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Largest absolute per-bin calibration gap (MCE)."""
    return calibration_table(labels, scores, n_bins=n_bins).mce


@dataclass(frozen=True)
class EvaluationReport:
    """Discrimination and calibration metrics for one model run."""

    n: int
    n_positive: int
    prevalence: float
    auroc: float
    brier: float
    log_loss_value: float
    ece: float
    mce: float
    table: CalibrationTable


def evaluate(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> EvaluationReport:
    """Compute the full evaluation report for a set of predictions."""
    clean_labels, clean_scores = _validate(labels, scores)
    table = calibration_table(clean_labels, clean_scores, n_bins=n_bins)
    return EvaluationReport(
        n=len(clean_labels),
        n_positive=sum(clean_labels),
        prevalence=sum(clean_labels) / len(clean_labels),
        auroc=auroc(clean_labels, clean_scores),
        brier=brier_score(clean_labels, clean_scores),
        log_loss_value=log_loss(clean_labels, clean_scores),
        ece=table.ece,
        mce=table.mce,
        table=table,
    )
