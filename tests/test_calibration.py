"""Tests for provome.calibration: AUROC, Brier, ECE, log loss."""

import math

import pytest

from provome.calibration import (
    auroc,
    brier_score,
    calibration_table,
    evaluate,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
)


def test_auroc_perfect_separation():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auroc(labels, scores) == pytest.approx(1.0)


def test_auroc_perfectly_wrong():
    labels = [0, 0, 1, 1]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert auroc(labels, scores) == pytest.approx(0.0)


def test_auroc_all_ties_is_chance():
    assert auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_hand_computed():
    # Positives scored 0.35, 0.8; negatives 0.1, 0.4.
    # Pairs: 0.35 beats 0.1, loses to 0.4; 0.8 beats both: 3 of 4.
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert auroc(labels, scores) == pytest.approx(0.75)


def test_auroc_single_class_raises():
    with pytest.raises(ValueError):
        auroc([1, 1, 1], [0.2, 0.3, 0.4])


def test_brier_perfect_and_worst():
    assert brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert brier_score([0, 1], [1.0, 0.0]) == pytest.approx(1.0)


def test_brier_hand_computed():
    # (0.9-1)^2 and (0.1-0)^2 -> 0.01 + 0.01, mean 0.01
    assert brier_score([1, 0], [0.9, 0.1]) == pytest.approx(0.01)


def test_ece_hand_computable_example():
    # Two observations in two different bins, each off by 0.1.
    labels = [1, 0]
    scores = [0.9, 0.1]
    assert expected_calibration_error(labels, scores, n_bins=10) == pytest.approx(0.1)


def test_ece_perfect_calibration_on_bin_centres():
    # One positive scored 1.0 would sit in the last bin with observed
    # 1.0 and gap 0; one negative scored 0.0 similarly.  ECE = 0.
    labels = [1, 0]
    scores = [1.0, 0.0]
    assert expected_calibration_error(labels, scores) == pytest.approx(0.0)


def test_ece_overconfident_wrong_is_large():
    labels = [0, 0]
    scores = [0.95, 0.95]
    assert expected_calibration_error(labels, scores) == pytest.approx(0.95)


def test_calibration_table_contents():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    table = calibration_table(labels, scores, n_bins=10)
    assert table.total == 4
    top = table.bins[9]
    assert top.count == 1 and top.observed_frequency == 1.0
    bin8 = table.bins[8]
    assert bin8.count == 1 and bin8.observed_frequency == 0.0
    assert table.mce == pytest.approx(maximum_calibration_error(labels, scores))


def test_score_of_one_lands_in_last_bin():
    table = calibration_table([1], [1.0], n_bins=10)
    assert table.bins[-1].count == 1


def test_log_loss_known_values():
    # -ln(0.9) for a correct confident prediction.
    assert log_loss([1], [0.9]) == pytest.approx(-math.log(0.9))
    # Clipping keeps a confident error finite but huge.
    value = log_loss([1], [0.0])
    assert value == pytest.approx(-math.log(1e-15))


def test_evaluate_report_is_consistent():
    labels = [0, 0, 1, 1, 1, 0]
    scores = [0.05, 0.3, 0.6, 0.9, 0.7, 0.4]
    report = evaluate(labels, scores)
    assert report.n == 6
    assert report.n_positive == 3
    assert report.prevalence == pytest.approx(0.5)
    assert 0.0 <= report.auroc <= 1.0
    assert report.ece == pytest.approx(report.table.ece)


def test_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        auroc([0, 1], [0.5])
    with pytest.raises(ValueError):
        brier_score([], [])
    with pytest.raises(ValueError):
        brier_score([2], [0.5])
    with pytest.raises(ValueError):
        brier_score([1], [1.5])
