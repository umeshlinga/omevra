"""Tests for omevra.phenotype: IC, Resnik, best-match average."""

import math

import pytest

from omevra.phenotype import (
    Ontology,
    OntologyTerm,
    best_match_average,
    compare_term_sets,
    example_ontology,
    information_content,
    most_informative_common_ancestor,
    resnik_similarity,
)

ROOT = "HP:0000118"
ARACHNODACTYLY = "HP:0001166"
TAPERED_FINGER = "HP:0001182"
HAND = "HP:0001155"
HCM = "HP:0001639"
DCM = "HP:0001644"
CARDIOMYOPATHY = "HP:0001638"


@pytest.fixture(scope="module")
def ontology():
    return example_ontology()


def test_example_ontology_is_labelled_and_small(ontology):
    assert "example" in ontology.name
    assert "illustrative" in ontology.name
    assert 10 <= len(ontology) <= 40
    assert ontology.root.term_id == ROOT


def test_root_information_content_is_zero(ontology):
    assert information_content(ontology, ROOT) == pytest.approx(0.0)


def test_information_content_hand_computed(ontology):
    # Arachnodactyly count 150 of a 10000 corpus: -ln(0.015)
    assert information_content(ontology, ARACHNODACTYLY) == pytest.approx(
        -math.log(150 / 10000)
    )


def test_specific_terms_have_higher_ic_than_general(ontology):
    assert information_content(ontology, ARACHNODACTYLY) > information_content(ontology, HAND)
    assert information_content(ontology, HAND) > information_content(ontology, ROOT)


def test_mica_of_hand_terms_is_the_hand_term(ontology):
    assert most_informative_common_ancestor(
        ontology, ARACHNODACTYLY, TAPERED_FINGER
    ) == HAND


def test_resnik_similarity_is_ic_of_mica(ontology):
    expected = -math.log(1200 / 10000)  # hand term count
    assert resnik_similarity(ontology, ARACHNODACTYLY, TAPERED_FINGER) == pytest.approx(expected)


def test_resnik_self_similarity_is_own_ic(ontology):
    assert resnik_similarity(ontology, HCM, HCM) == pytest.approx(
        information_content(ontology, HCM)
    )


def test_resnik_sibling_cardiomyopathies(ontology):
    # HCM and DCM share the cardiomyopathy parent.
    assert most_informative_common_ancestor(ontology, HCM, DCM) == CARDIOMYOPATHY


def test_unrelated_branches_meet_at_general_ancestor(ontology):
    sim = resnik_similarity(ontology, ARACHNODACTYLY, HCM)
    hand_ic = information_content(ontology, HAND)
    assert sim < hand_ic  # their common ancestors are all general


def test_best_match_average_identical_sets(ontology):
    score = best_match_average(ontology, [ARACHNODACTYLY], [ARACHNODACTYLY])
    assert score == pytest.approx(information_content(ontology, ARACHNODACTYLY))


def test_best_match_average_is_symmetric(ontology):
    a = [ARACHNODACTYLY, HCM]
    b = [TAPERED_FINGER, DCM]
    assert best_match_average(ontology, a, b) == pytest.approx(
        best_match_average(ontology, b, a)
    )


def test_compare_term_sets_reports_pairs(ontology):
    result = compare_term_sets(ontology, [ARACHNODACTYLY], [TAPERED_FINGER])
    assert result.best_pairs[0][0] == ARACHNODACTYLY
    assert result.best_pairs[0][1] == TAPERED_FINGER
    assert result.similarity == pytest.approx(result.forward)


def test_ancestors_and_descendants(ontology):
    ancestors = ontology.ancestors(ARACHNODACTYLY)
    assert HAND in ancestors and ROOT in ancestors
    assert ARACHNODACTYLY in ontology.descendants(HAND)
    assert ontology.depth(ROOT) == 0
    assert ontology.depth(ARACHNODACTYLY) > ontology.depth(HAND)


def test_unknown_term_raises(ontology):
    with pytest.raises(KeyError):
        ontology.information_content("HP:9999999")


def test_ontology_validation_rejects_cycle():
    terms = (
        OntologyTerm("X:1", "root", (), 100),
        OntologyTerm("X:2", "a", ("X:3",), 50),
        OntologyTerm("X:3", "b", ("X:2",), 40),
    )
    with pytest.raises(ValueError):
        Ontology(terms)


def test_ontology_validation_rejects_missing_parent():
    terms = (
        OntologyTerm("X:1", "root", (), 100),
        OntologyTerm("X:2", "a", ("X:9",), 50),
    )
    with pytest.raises(ValueError):
        Ontology(terms)
