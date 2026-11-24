"""Phenotype ontology similarity: information content and Resnik.

Phenotype comparison is a graph problem.  Ontologies such as the
Human Phenotype Ontology (HPO; Kohler et al. 2021, *Nucleic Acids
Research* 49:D1207-D1217) organise terms in a directed acyclic graph
(DAG) from general (``Phenotypic abnormality``) to specific
(``Arachnodactyly``).  Two patients, or a patient and a disease, are
similar when their term sets share *informative* ancestors.

This module implements the standard machinery, from scratch:

* **Information content (IC)** of a term: ``-log(p(term))``, where
  ``p(term)`` is the frequency of the term *or any of its
  descendants* in an annotated corpus.  Rare, specific terms carry
  more information than common, general ones.  This is the
  corpus-based IC of Resnik (1995).
* **Most informative common ancestor (MICA)**: the common ancestor of
  two terms with the highest IC.
* **Resnik similarity**: the IC of the MICA.
* **Best-match average (BMA)**: the set-level measure used in
  practice (for example by the HPO ecosystem and Phenomizer-style
  tools): for each term in set A, take its best Resnik match in set
  B, average, do the same in reverse, and average the two directions
  so the measure is symmetric.

The bundled ontology is a **small example subset**, hand-built to be
structurally truthful (real HPO identifiers and real parent
relationships for the terms it includes, pruned to a teaching size),
and clearly labelled as an example subset wherever it is loaded.  It
is not the HPO and must not be used as if it were.  The corpus counts
attached to it are illustrative counts for demonstrating the
mathematics, not epidemiological data.

The mathematics here is ontology-agnostic: the same code scores
similarity over a veterinary or plant trait ontology DAG, which is
why :mod:`omevra.domains` lists phenotype similarity as a
species-agnostic engine with a division-specific ontology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

__all__ = [
    "OntologyTerm",
    "Ontology",
    "example_ontology",
    "information_content",
    "most_informative_common_ancestor",
    "resnik_similarity",
    "best_match_average",
    "TermSetSimilarity",
    "compare_term_sets",
]


@dataclass(frozen=True)
class OntologyTerm:
    """One term in an ontology DAG.

    Attributes:
        term_id: Stable identifier, for example ``HP:0001166``.
        name: Human readable label.
        parents: Ids of direct parent (is-a) terms.  A term with no
            parents is a root.
        annotation_count: Number of corpus annotations to this term
            or any descendant (the propagated count).  IC is computed
            from these counts, so they must be propagated counts, not
            direct-only counts, for the mathematics to be coherent.
    """

    term_id: str
    name: str
    parents: tuple[str, ...] = ()
    annotation_count: int = 0

    def __post_init__(self) -> None:
        if not self.term_id.strip():
            raise ValueError("term_id must be non-empty")
        if self.annotation_count < 0:
            raise ValueError("annotation_count must be non-negative")


class Ontology:
    """A validated ontology DAG with IC and similarity operations.

    Construction validates the graph: every parent must exist, term
    ids must be unique, and the graph must be acyclic (a cycle would
    make ancestor computation and IC meaningless).
    """

    def __init__(self, terms: Iterable[OntologyTerm], *, name: str = "ontology") -> None:
        self.name = name
        self._terms: dict[str, OntologyTerm] = {}
        for term in terms:
            if term.term_id in self._terms:
                raise ValueError(f"duplicate term id: {term.term_id}")
            self._terms[term.term_id] = term
        for term in self._terms.values():
            for parent in term.parents:
                if parent not in self._terms:
                    raise ValueError(
                        f"term {term.term_id} names missing parent {parent}"
                    )
        self._children: dict[str, list[str]] = {tid: [] for tid in self._terms}
        for term in self._terms.values():
            for parent in term.parents:
                self._children[parent].append(term.term_id)
        self._check_acyclic()
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        roots = [t for t in self._terms.values() if not t.parents]
        if len(roots) != 1:
            raise ValueError(
                f"expected exactly one root term, found {len(roots)}"
            )
        self._root = roots[0]
        if self._root.annotation_count <= 0:
            raise ValueError(
                "the root term needs a positive annotation_count: it "
                "is the corpus total IC is normalised by"
            )

    # -- structure -------------------------------------------------
    def _check_acyclic(self) -> None:
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(term_id: str) -> None:
            if term_id in done:
                return
            if term_id in visiting:
                raise ValueError(f"ontology contains a cycle at {term_id}")
            visiting.add(term_id)
            for parent in self._terms[term_id].parents:
                visit(parent)
            visiting.discard(term_id)
            done.add(term_id)

        for term_id in self._terms:
            visit(term_id)

    def __contains__(self, term_id: object) -> bool:
        return term_id in self._terms

    def __len__(self) -> int:
        return len(self._terms)

    @property
    def root(self) -> OntologyTerm:
        return self._root

    def term(self, term_id: str) -> OntologyTerm:
        """Return a term or raise ``KeyError``."""
        try:
            return self._terms[term_id]
        except KeyError as exc:
            raise KeyError(f"unknown term id: {term_id!r}") from exc

    def all_terms(self) -> tuple[OntologyTerm, ...]:
        return tuple(self._terms.values())

    def ancestors(self, term_id: str, *, include_self: bool = True) -> frozenset[str]:
        """All ancestors of a term, following is-a parents upward."""
        self.term(term_id)
        if term_id in self._ancestor_cache:
            result = self._ancestor_cache[term_id]
            return result if include_self else frozenset(result - {term_id})
        collected: set[str] = {term_id}
        for parent in self._terms[term_id].parents:
            collected |= self.ancestors(parent)
        frozen = frozenset(collected)
        self._ancestor_cache[term_id] = frozen
        return frozen if include_self else frozenset(frozen - {term_id})

    def descendants(self, term_id: str, *, include_self: bool = True) -> frozenset[str]:
        """All descendants of a term."""
        self.term(term_id)
        collected: set[str] = set()
        stack = list(self._children[term_id])
        while stack:
            current = stack.pop()
            if current in collected:
                continue
            collected.add(current)
            stack.extend(self._children[current])
        if include_self:
            collected.add(term_id)
        return frozenset(collected)

    def common_ancestors(self, a: str, b: str) -> frozenset[str]:
        return self.ancestors(a) & self.ancestors(b)

    def depth(self, term_id: str) -> int:
        """Longest path length from the root (root depth 0)."""
        term = self.term(term_id)
        if not term.parents:
            return 0
        return 1 + max(self.depth(parent) for parent in term.parents)

    # -- information content and similarity -----------------------
    def corpus_total(self) -> int:
        """Total annotations in the corpus (the root count)."""
        return self._root.annotation_count

    def information_content(self, term_id: str) -> float:
        """IC of a term: ``-ln(p)`` with ``p`` from propagated counts.

        A term with a zero count has, by convention here, the IC of
        the corpus's rarest observable event, ``-ln(1/total)``: it is
        treated as maximally informative rather than infinitely so,
        which keeps similarity scores finite and comparable.
        """
        term = self.term(term_id)
        total = self.corpus_total()
        count = term.annotation_count
        if count <= 0:
            probability = 1.0 / total
        else:
            probability = count / total
        if probability >= 1.0:
            return 0.0
        return -math.log(probability)

    def most_informative_common_ancestor(self, a: str, b: str) -> str:
        """The common ancestor of two terms with maximum IC."""
        common = self.common_ancestors(a, b)
        if not common:  # pragma: no cover - single root prevents this
            raise ValueError(f"{a} and {b} share no common ancestor")
        return max(common, key=lambda tid: self.information_content(tid))

    def resnik_similarity(self, a: str, b: str) -> float:
        """Resnik similarity: IC of the most informative common ancestor."""
        mica = self.most_informative_common_ancestor(a, b)
        return self.information_content(mica)


def information_content(ontology: Ontology, term_id: str) -> float:
    """Module level shortcut for :meth:`Ontology.information_content`."""
    return ontology.information_content(term_id)


def most_informative_common_ancestor(ontology: Ontology, a: str, b: str) -> str:
    """Module level shortcut for the MICA of two terms."""
    return ontology.most_informative_common_ancestor(a, b)


def resnik_similarity(ontology: Ontology, a: str, b: str) -> float:
    """Module level shortcut for Resnik similarity."""
    return ontology.resnik_similarity(a, b)


def _directional_best_match(
    ontology: Ontology,
    from_set: Sequence[str],
    to_set: Sequence[str],
) -> float:
    if not from_set or not to_set:
        return 0.0
    total = 0.0
    for term in from_set:
        total += max(ontology.resnik_similarity(term, other) for other in to_set)
    return total / len(from_set)


@dataclass(frozen=True)
class TermSetSimilarity:
    """Result of comparing two phenotype term sets.

    Attributes:
        similarity: Symmetric best-match average score.
        forward: Mean best match from set A to set B.
        reverse: Mean best match from set B to set A.
        best_pairs: For each term in A, its best matching term in B
            and the pair's Resnik score, so a human can audit why the
            sets scored as they did.
    """

    similarity: float
    forward: float
    reverse: float
    best_pairs: tuple[tuple[str, str, float], ...] = ()


def compare_term_sets(
    ontology: Ontology,
    set_a: Sequence[str],
    set_b: Sequence[str],
) -> TermSetSimilarity:
    """Compare two term sets with symmetric best-match averaging."""
    a = list(dict.fromkeys(set_a))
    b = list(dict.fromkeys(set_b))
    for term_id in a + b:
        ontology.term(term_id)  # raises on unknown ids
    forward = _directional_best_match(ontology, a, b)
    reverse = _directional_best_match(ontology, b, a)
    pairs: list[tuple[str, str, float]] = []
    for term in a:
        if not b:
            break
        best_other = max(b, key=lambda other: ontology.resnik_similarity(term, other))
        pairs.append((term, best_other, ontology.resnik_similarity(term, best_other)))
    return TermSetSimilarity(
        similarity=(forward + reverse) / 2.0 if (a and b) else 0.0,
        forward=forward,
        reverse=reverse,
        best_pairs=tuple(pairs),
    )


def best_match_average(
    ontology: Ontology,
    set_a: Sequence[str],
    set_b: Sequence[str],
) -> float:
    """Symmetric best-match average similarity between two term sets."""
    return compare_term_sets(ontology, set_a, set_b).similarity


# ---------------------------------------------------------------------
# Bundled example ontology subset.
#
# Identifiers and parent relationships below are real HPO structure
# for the included terms, pruned to a small connected subset.  The
# annotation counts are ILLUSTRATIVE numbers chosen to demonstrate
# the mathematics (they decrease monotonically toward the leaves, as
# propagated counts must); they are not HPO corpus frequencies and
# must not be quoted as such.
# ---------------------------------------------------------------------

_EXAMPLE_TERMS: tuple[OntologyTerm, ...] = (
    OntologyTerm("HP:0000118", "Phenotypic abnormality", (), 10000),
    OntologyTerm("HP:0000152", "Abnormality of head or neck", ("HP:0000118",), 4000),
    OntologyTerm("HP:0000234", "Abnormality of the head", ("HP:0000152",), 3000),
    OntologyTerm("HP:0000924", "Abnormality of the skeletal system", ("HP:0000118",), 5000),
    OntologyTerm("HP:0011842", "Abnormality of skeletal morphology", ("HP:0000924",), 4200),
    OntologyTerm("HP:0002973", "Abnormality of the upper limb", ("HP:0011842",), 2000),
    OntologyTerm("HP:0001155", "Abnormality of the hand", ("HP:0002973",), 1200),
    OntologyTerm("HP:0001166", "Arachnodactyly", ("HP:0001155",), 150),
    OntologyTerm("HP:0001172", "Abnormal thumb morphology", ("HP:0001155",), 500),
    OntologyTerm("HP:0001182", "Tapered finger", ("HP:0001155",), 300),
    OntologyTerm("HP:0001363", "Craniosynostosis", ("HP:0000234",), 250),
    OntologyTerm("HP:0001626", "Abnormality of the cardiovascular system", ("HP:0000118",), 3500),
    OntologyTerm("HP:0001638", "Cardiomyopathy", ("HP:0001626",), 800),
    OntologyTerm("HP:0001639", "Hypertrophic cardiomyopathy", ("HP:0001638",), 400),
    OntologyTerm("HP:0001644", "Dilated cardiomyopathy", ("HP:0001638",), 350),
    OntologyTerm("HP:0002011", "Morphological central nervous system abnormality", ("HP:0000707",), 2800),
    OntologyTerm("HP:0002072", "Chorea", ("HP:0002011",), 180),
    OntologyTerm("HP:0001250", "Seizure", ("HP:0002011",), 900),
    OntologyTerm("HP:0000707", "Abnormality of the nervous system", ("HP:0000118",), 3200),
)


def example_ontology() -> Ontology:
    """Return the bundled example ontology subset.

    This is a small, structurally truthful subset for demonstration
    and testing.  It is not the Human Phenotype Ontology, and its
    annotation counts are illustrative, not epidemiological.
    """
    return Ontology(_EXAMPLE_TERMS, name="omevra-example-hpo-subset (illustrative)")
