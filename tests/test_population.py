"""Tests for provome.population: Whiffin maximum credible AF."""

import math

import pytest

from provome.population import (
    DiseaseModel,
    Inheritance,
    credible_interval_excludes,
    evaluate_frequency,
    max_credible_af,
    max_credible_allele_count,
)

DOMINANT_SIMPLE = DiseaseModel(
    prevalence=0.002,  # 1 in 500
    inheritance=Inheritance.AUTOSOMAL_DOMINANT,
    penetrance=1.0,
    genetic_heterogeneity=1.0,
    allelic_heterogeneity=1.0,
)


def test_dominant_formula_arithmetic():
    # q = prevalence * gh * ah / (2 * penetrance) = 0.002 / 2
    assert max_credible_af(DOMINANT_SIMPLE) == pytest.approx(0.001)


def test_dominant_formula_with_heterogeneity_and_penetrance():
    model = DiseaseModel(
        prevalence=0.002,
        inheritance=Inheritance.AUTOSOMAL_DOMINANT,
        penetrance=0.5,
        genetic_heterogeneity=0.5,
        allelic_heterogeneity=0.1,
    )
    # 0.002 * 0.5 * 0.1 / (2 * 0.5) = 0.0001 / 1.0
    assert max_credible_af(model) == pytest.approx(0.0001)


def test_recessive_formula_arithmetic():
    # Cystic fibrosis order of magnitude: prevalence about 1 in 2500.
    model = DiseaseModel(
        prevalence=0.0004,
        inheritance=Inheritance.AUTOSOMAL_RECESSIVE,
        penetrance=1.0,
    )
    # sqrt(0.0004) = 0.02
    assert max_credible_af(model) == pytest.approx(0.02)


def test_recessive_heterogeneity_reduces_ceiling():
    model = DiseaseModel(
        prevalence=0.0004,
        inheritance=Inheritance.AUTOSOMAL_RECESSIVE,
        penetrance=1.0,
        allelic_heterogeneity=0.25,
    )
    assert max_credible_af(model) == pytest.approx(math.sqrt(0.0001))


def test_max_credible_allele_count():
    assert max_credible_allele_count(DOMINANT_SIMPLE, 100_000) == pytest.approx(100.0)


def test_model_validation():
    with pytest.raises(ValueError):
        DiseaseModel(prevalence=0.0, inheritance=Inheritance.AUTOSOMAL_DOMINANT)
    with pytest.raises(ValueError):
        DiseaseModel(prevalence=1.5, inheritance=Inheritance.AUTOSOMAL_DOMINANT)
    with pytest.raises(ValueError):
        DiseaseModel(prevalence=0.01, inheritance=Inheritance.AUTOSOMAL_DOMINANT,
                     penetrance=0.0)


def test_ba1_threshold():
    result = evaluate_frequency(0.06, model=DOMINANT_SIMPLE)
    assert result.ba1_met
    assert result.bs1_met  # also far above this disease ceiling


def test_bs1_when_above_ceiling_but_below_ba1():
    result = evaluate_frequency(0.01, model=DOMINANT_SIMPLE)
    assert not result.ba1_met
    assert result.bs1_met
    assert not result.pm2_consistent


def test_pm2_consistent_when_rare():
    result = evaluate_frequency(0.0005, model=DOMINANT_SIMPLE)
    assert result.pm2_consistent
    assert not result.bs1_met
    assert not result.absent


def test_absent_variant_is_pm2_consistent_and_flagged():
    result = evaluate_frequency(None, model=DOMINANT_SIMPLE)
    assert result.absent
    assert result.observed_af == 0.0
    assert result.pm2_consistent


def test_counts_input_matches_af_input():
    by_counts = evaluate_frequency(
        observed_count=5, allele_number=10_000, model=DOMINANT_SIMPLE
    )
    by_af = evaluate_frequency(0.0005, model=DOMINANT_SIMPLE)
    assert by_counts.observed_af == pytest.approx(by_af.observed_af)
    assert by_counts.pm2_consistent == by_af.pm2_consistent


def test_credible_interval_screening():
    # A very common variant in a huge sample: lower bound far above
    # a rare disease ceiling.
    assert credible_interval_excludes(5000, 100_000, 0.001)
    # A handful of observations: sampling noise could explain it.
    assert not credible_interval_excludes(3, 1000, 0.001)
    assert not credible_interval_excludes(0, 1000, 0.001)
