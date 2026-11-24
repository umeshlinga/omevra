"""Omevra: an evidence engine for biology, in three divisions.

One engine - sequence consequence, population frequency mathematics,
calibration, phenotype similarity, and a cite-or-decline evidence
gate - pointed at Human, Animal, and Plant problems.  The Human
division is live and benchmarked today; Animal and Plant are
expansion divisions on the same engine.

Modules:
    evidence:    evidence items, sources, and the cite-or-decline gate.
    copilot:     the Omevra Copilot, the single question-answering
                 surface of the engine.
    domains:     the Human / Animal / Plant division registry.
    variant:     VCF records and HGVS parsing and validation.
    consequence: molecular consequence from real sequence.
    acmg:        ACMG/AMP Richards rules and Tavtigian points
                 (Human division, clinical context only).
    population:  maximum credible allele frequency (Whiffin et al.).
    calibration: AUROC, Brier, ECE, and calibration tables.
    phenotype:   ontology information content and Resnik similarity.

Research software.  Not for clinical use.
"""

from . import (
    acmg,
    calibration,
    consequence,
    copilot,
    domains,
    evidence,
    phenotype,
    population,
    variant,
)
from .copilot import Copilot, CopilotAnswer, CopilotQuestion, OmevraCopilot
from .domains import Division, get_division
from .evidence import EvidenceItem, evaluate_gate

__version__ = "0.1.0"

__all__ = [
    "acmg",
    "calibration",
    "consequence",
    "copilot",
    "domains",
    "evidence",
    "phenotype",
    "population",
    "variant",
    "Copilot",
    "OmevraCopilot",
    "CopilotAnswer",
    "CopilotQuestion",
    "Division",
    "get_division",
    "EvidenceItem",
    "evaluate_gate",
    "__version__",
]
