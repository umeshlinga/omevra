"""Tests for omevra.acmg: Tavtigian points and Richards rules."""

import pytest

from omevra.acmg import (
    Classification,
    Criterion,
    CriterionAssessment,
    Strength,
    classify,
    classify_by_points,
    classify_by_richards_rules,
    points_for,
    total_points,
)
from omevra.acmg import Direction


def assess(*codes):
    return [CriterionAssessment(criterion=c) for c in codes]


def test_point_values_match_tavtigian():
    assert points_for(Direction.PATHOGENIC, Strength.VERY_STRONG) == 8
    assert points_for(Direction.PATHOGENIC, Strength.STRONG) == 4
    assert points_for(Direction.PATHOGENIC, Strength.MODERATE) == 2
    assert points_for(Direction.PATHOGENIC, Strength.SUPPORTING) == 1
    assert points_for(Direction.BENIGN, Strength.STAND_ALONE) == -8
    assert points_for(Direction.BENIGN, Strength.STRONG) == -4
    assert points_for(Direction.BENIGN, Strength.SUPPORTING) == -1


def test_default_strengths_from_codes():
    assert Criterion.PVS1.default_strength is Strength.VERY_STRONG
    assert Criterion.PS3.default_strength is Strength.STRONG
    assert Criterion.PM2.default_strength is Strength.MODERATE
    assert Criterion.PP3.default_strength is Strength.SUPPORTING
    assert Criterion.BA1.default_strength is Strength.STAND_ALONE
    assert Criterion.BS1.default_strength is Strength.STRONG
    assert Criterion.BP7.default_strength is Strength.SUPPORTING


@pytest.mark.parametrize("total,expected", [
    (12, Classification.PATHOGENIC),
    (10, Classification.PATHOGENIC),
    (9, Classification.LIKELY_PATHOGENIC),
    (6, Classification.LIKELY_PATHOGENIC),
    (5, Classification.VUS),
    (0, Classification.VUS),
    (-1, Classification.LIKELY_BENIGN),
    (-6, Classification.LIKELY_BENIGN),
    (-7, Classification.BENIGN),
    (-12, Classification.BENIGN),
])
def test_classify_by_points_thresholds(total, expected):
    assert classify_by_points(total) is expected


def test_pvs1_plus_ps_is_pathogenic_both_frameworks():
    report = classify(assess(Criterion.PVS1, Criterion.PS3))
    assert report.points_total == 12
    assert report.points_classification is Classification.PATHOGENIC
    assert report.richards_classification is Classification.PATHOGENIC
    assert report.agree


def test_pvs1_alone_points_vs_rules_boundary():
    # 8 points is Likely Pathogenic in the points system, but the
    # 2015 combining rules need PVS1 plus at least one more criterion
    # for Likely Pathogenic, so alone it is a VUS there.  The module
    # must surface, not hide, that boundary disagreement.
    report = classify(assess(Criterion.PVS1))
    assert report.points_total == 8
    assert report.points_classification is Classification.LIKELY_PATHOGENIC
    assert report.richards_classification is Classification.VUS
    assert not report.agree
    assert any("disagree" in n for n in report.notes)


def test_ps_plus_two_pm_is_likely_pathogenic():
    report = classify(assess(Criterion.PS1, Criterion.PM1, Criterion.PM2))
    assert report.points_total == 8
    assert report.points_classification is Classification.LIKELY_PATHOGENIC
    assert report.richards_classification is Classification.LIKELY_PATHOGENIC


def test_three_pm_is_likely_pathogenic():
    report = classify(assess(Criterion.PM1, Criterion.PM2, Criterion.PM3))
    assert report.points_total == 6
    assert report.richards_classification is Classification.LIKELY_PATHOGENIC


def test_two_ps_is_pathogenic_by_rules():
    result = classify_by_richards_rules(assess(Criterion.PS1, Criterion.PS2))
    assert result is Classification.PATHOGENIC


def test_pvs_plus_two_pp_is_pathogenic_by_rules():
    result = classify_by_richards_rules(
        assess(Criterion.PVS1, Criterion.PP1, Criterion.PP3)
    )
    assert result is Classification.PATHOGENIC


def test_supporting_only_is_vus():
    report = classify(assess(Criterion.PP1, Criterion.PP3))
    assert report.points_total == 2
    assert report.points_classification is Classification.VUS
    assert report.richards_classification is Classification.VUS


def test_ba1_alone_is_benign_both_frameworks():
    report = classify(assess(Criterion.BA1))
    assert report.points_total == -8
    assert report.points_classification is Classification.BENIGN
    assert report.richards_classification is Classification.BENIGN


def test_bs_plus_bp_is_likely_benign():
    report = classify(assess(Criterion.BS1, Criterion.BP4))
    assert report.points_total == -5
    assert report.points_classification is Classification.LIKELY_BENIGN
    assert report.richards_classification is Classification.LIKELY_BENIGN


def test_two_bp_is_likely_benign():
    report = classify(assess(Criterion.BP1, Criterion.BP4))
    assert report.points_total == -2
    assert report.richards_classification is Classification.LIKELY_BENIGN


def test_strength_modification_changes_points_and_is_noted():
    modified = CriterionAssessment(
        criterion=Criterion.PS3, strength=Strength.MODERATE,
        rationale="assay validated but at moderate strength",
    )
    assert modified.is_modified
    assert modified.points == 2
    report = classify([modified])
    assert any("modified strength" in n for n in report.notes)


def test_duplicate_criterion_is_rejected():
    with pytest.raises(ValueError):
        total_points(assess(Criterion.PM2, Criterion.PM2))


def test_conflicting_directions_are_noted():
    report = classify(assess(Criterion.PS3, Criterion.BS3))
    assert any("both directions" in n for n in report.notes)


def test_every_criterion_has_a_description():
    for criterion in Criterion:
        assert len(criterion.description) > 20
