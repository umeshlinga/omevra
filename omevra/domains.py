"""Divisions: one evidence engine, three biological kingdoms of use.

Omevra is not a human genetics tool with extra labels, and it is not
a cancer tool.  It is an evidence engine - variant and sequence
analysis, frequency mathematics, calibrated scoring, and a
cite-or-decline gate - and that engine is pointed at three divisions:

* **Human**: all human disease, not only cancer.  Rare disease,
  cardiac, neurological, metabolic, immunological, infectious
  disease host genetics, and more.
* **Animal**: veterinary medicine and livestock.  Companion animals,
  cattle, poultry, pigs, horses, small ruminants, and aquaculture.
* **Plant**: crops and agriculture.  Staple crops, model plants,
  plant pathology, and breeding.

This module is the registry that keeps those divisions honest.  Each
division declares its scope, its problem areas, example species, and
- critically - which engines genuinely apply to it:

* **Species-agnostic engines.**  Molecular consequence prediction
  (the standard genetic code is shared, with the caveat that some
  lineages use variant codes, for example ciliates and some
  mitochondria), population frequency mathematics (the Whiffin
  framework is algebra about prevalence and penetrance, not about
  humans), calibration metrics (AUROC, Brier, ECE are properties of
  predictions and outcomes anywhere), phenotype semantic similarity
  (the mathematics applies to any ontology DAG), and the
  cite-or-decline evidence gate (a policy about proof, not about
  species) all transfer across divisions.
* **Human-clinical specific engines.**  The ACMG/AMP criteria in
  :mod:`omevra.acmg` (Richards et al. 2015; Tavtigian et al. 2018)
  were written for human germline variant interpretation in a
  clinical setting.  They do not transfer to animals or plants, and
  this registry says so in data, so no caller can quietly apply a
  human clinical framework to a cow or a rice plant.  Veterinary and
  plant work needs its own classification frameworks, applied with
  the same rigour, and building those is division work, not a
  relabelling of ACMG.

Honest status, stated here and in the README: the Human division is
the live, benchmarked division today.  Animal and Plant are expansion
divisions built on the same species-agnostic engine.  This registry
describes the architecture truthfully; it does not claim benchmarked
coverage for Animal or Plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

__all__ = [
    "Division",
    "DivisionStatus",
    "Species",
    "EngineApplicability",
    "DivisionInfo",
    "DivisionRegistry",
    "default_registry",
    "get_division",
    "division_for_species",
    "all_divisions",
    "ENGINE_CONSEQUENCE",
    "ENGINE_POPULATION",
    "ENGINE_CALIBRATION",
    "ENGINE_EVIDENCE_GATE",
    "ENGINE_PHENOTYPE_SIMILARITY",
    "ENGINE_VARIANT_NOTATION",
    "ENGINE_ACMG",
]

ENGINE_CONSEQUENCE = "molecular_consequence"
ENGINE_POPULATION = "population_frequency"
ENGINE_CALIBRATION = "calibration"
ENGINE_EVIDENCE_GATE = "evidence_gate"
ENGINE_PHENOTYPE_SIMILARITY = "phenotype_similarity"
ENGINE_VARIANT_NOTATION = "variant_notation"
ENGINE_ACMG = "acmg_classification"


class Division(str, Enum):
    """The three Omevra divisions."""

    HUMAN = "human"
    ANIMAL = "animal"
    PLANT = "plant"


class DivisionStatus(str, Enum):
    """Honest delivery status of a division."""

    LIVE_BENCHMARKED = "live_benchmarked"
    EXPANSION = "expansion"


@dataclass(frozen=True)
class Species:
    """One example species in a division.

    Attributes:
        scientific_name: Binomial name, for example ``Bos taurus``.
        common_name: Everyday name, for example ``domestic cattle``.
        division: The division this species is an example for.
        note: Why the species matters to the division.
    """

    scientific_name: str
    common_name: str
    division: Division
    note: str = ""

    def __post_init__(self) -> None:
        if not self.scientific_name.strip():
            raise ValueError("scientific_name must be non-empty")
        if not self.common_name.strip():
            raise ValueError("common_name must be non-empty")


@dataclass(frozen=True)
class EngineApplicability:
    """Whether one engine applies to one division, and why.

    Attributes:
        engine: Engine identifier (see the ``ENGINE_`` constants).
        applies: True when the engine genuinely applies.
        note: The reasoning, including caveats.  A caveat on an
            applicable engine is as important as the yes itself.
    """

    engine: str
    applies: bool
    note: str = ""


@dataclass(frozen=True)
class DivisionInfo:
    """Full registry entry for one division.

    Attributes:
        division: The division enum value.
        name: Display name.
        scope: Plain language scope statement.
        problem_areas: The problem families the division covers.
        example_species: Species the division works with.
        engines: Applicability of every engine to this division.
        status: Live/benchmarked, or expansion.
    """

    division: Division
    name: str
    scope: str
    problem_areas: tuple[str, ...]
    example_species: tuple[Species, ...]
    engines: tuple[EngineApplicability, ...]
    status: DivisionStatus

    def engine_applies(self, engine: str) -> bool:
        """True when ``engine`` is declared applicable here."""
        for entry in self.engines:
            if entry.engine == engine:
                return entry.applies
        raise KeyError(f"engine {engine!r} not declared for {self.name}")

    def applicability(self, engine: str) -> EngineApplicability:
        """Return the full applicability record for an engine."""
        for entry in self.engines:
            if entry.engine == engine:
                return entry
        raise KeyError(f"engine {engine!r} not declared for {self.name}")

    def applicable_engines(self) -> tuple[str, ...]:
        return tuple(e.engine for e in self.engines if e.applies)

    def non_applicable_engines(self) -> tuple[str, ...]:
        return tuple(e.engine for e in self.engines if not e.applies)

    def species_names(self) -> tuple[str, ...]:
        return tuple(s.scientific_name for s in self.example_species)


def _shared_engines() -> tuple[EngineApplicability, ...]:
    """Engines that genuinely transfer across all three divisions."""
    return (
        EngineApplicability(
            engine=ENGINE_CONSEQUENCE,
            applies=True,
            note=(
                "Species-agnostic: the standard genetic code and "
                "reading-frame logic are shared. Caveat: some "
                "lineages use variant genetic codes (for example "
                "many mitochondria and ciliates); callers working "
                "in those lineages must state the code used."
            ),
        ),
        EngineApplicability(
            engine=ENGINE_POPULATION,
            applies=True,
            note=(
                "Species-agnostic mathematics: the maximum credible "
                "frequency framework is algebra over prevalence, "
                "heterogeneity, and penetrance (or selection in "
                "breeding populations). Inputs, not formulas, are "
                "division specific."
            ),
        ),
        EngineApplicability(
            engine=ENGINE_CALIBRATION,
            applies=True,
            note=(
                "Species-agnostic: AUROC, Brier score, and ECE are "
                "properties of predictions against outcomes in any "
                "organism."
            ),
        ),
        EngineApplicability(
            engine=ENGINE_EVIDENCE_GATE,
            applies=True,
            note=(
                "Species-agnostic policy: answer only with at least "
                "three verified evidence items from at least two "
                "independent sources, otherwise decline with "
                "reasons. Proof standards do not change by kingdom."
            ),
        ),
        EngineApplicability(
            engine=ENGINE_PHENOTYPE_SIMILARITY,
            applies=True,
            note=(
                "The Resnik semantic similarity mathematics applies "
                "to any ontology DAG. The ontology itself is "
                "division specific: HPO for human phenotypes, "
                "veterinary and livestock trait ontologies for "
                "animals, plant trait and phenotype ontologies for "
                "plants."
            ),
        ),
        EngineApplicability(
            engine=ENGINE_VARIANT_NOTATION,
            applies=True,
            note=(
                "Sequence variant notation and coordinate models "
                "are species-agnostic in form. Reference assemblies "
                "and transcript sets are species specific and must "
                "always be named."
            ),
        ),
    )


_ACMG_HUMAN = EngineApplicability(
    engine=ENGINE_ACMG,
    applies=True,
    note=(
        "Applies: the ACMG/AMP criteria (Richards et al. 2015) and "
        "the Tavtigian Bayesian points framework were written for "
        "human germline variant interpretation."
    ),
)

_ACMG_NOT_HUMAN = EngineApplicability(
    engine=ENGINE_ACMG,
    applies=False,
    note=(
        "Does not apply: ACMG/AMP criteria are human-clinical "
        "specific. This division needs its own classification "
        "framework; borrowing ACMG labels here would be a category "
        "error, and this registry refuses it in data."
    ),
)


def _human() -> DivisionInfo:
    return DivisionInfo(
        division=Division.HUMAN,
        name="Human",
        scope=(
            "All human disease genetics: an evidence engine for "
            "variant interpretation and disease biology across every "
            "disease area, not only cancer."
        ),
        problem_areas=(
            "rare disease",
            "cancer and somatic oncology",
            "cardiac disease",
            "neurological and neurodevelopmental disease",
            "metabolic disease",
            "immunological disease",
            "infectious disease and host genetics",
            "renal disease",
            "ophthalmic disease",
            "reproductive and prenatal genetics",
        ),
        example_species=(
            Species(
                "Homo sapiens",
                "human",
                Division.HUMAN,
                "The division's subject species.",
            ),
            Species(
                "Mus musculus",
                "house mouse",
                Division.HUMAN,
                "Principal mammalian model organism for human "
                "disease biology; listed as a model used by this "
                "division, not as a patient species.",
            ),
        ),
        engines=_shared_engines() + (_ACMG_HUMAN,),
        status=DivisionStatus.LIVE_BENCHMARKED,
    )


def _animal() -> DivisionInfo:
    return DivisionInfo(
        division=Division.ANIMAL,
        name="Animal",
        scope=(
            "Veterinary and livestock genetics: inherited disease, "
            "health, and breeding in companion animals, farm "
            "animals, and aquaculture species."
        ),
        problem_areas=(
            "companion animal inherited disease",
            "cattle health and breeding",
            "poultry disease and breeding",
            "pig health and breeding",
            "equine genetics",
            "small ruminant health",
            "aquaculture disease and breeding",
            "wildlife and conservation genetics",
        ),
        example_species=(
            Species("Canis lupus familiaris", "domestic dog", Division.ANIMAL,
                    "Companion animal with hundreds of characterised inherited diseases."),
            Species("Felis catus", "domestic cat", Division.ANIMAL,
                    "Companion animal genetics and inherited disease."),
            Species("Bos taurus", "domestic cattle", Division.ANIMAL,
                    "Dairy and beef breeding, fertility, and disease resistance."),
            Species("Sus scrofa", "pig and wild boar", Division.ANIMAL,
                    "Livestock breeding and disease resistance."),
            Species("Gallus gallus", "chicken", Division.ANIMAL,
                    "Poultry breeding and disease."),
            Species("Equus ferus caballus", "horse", Division.ANIMAL,
                    "Equine inherited disease and performance genetics."),
            Species("Ovis aries", "sheep", Division.ANIMAL,
                    "Small ruminant health and breeding."),
            Species("Capra hircus", "goat", Division.ANIMAL,
                    "Small ruminant health and breeding."),
            Species("Salmo salar", "Atlantic salmon", Division.ANIMAL,
                    "Aquaculture breeding and disease resistance."),
            Species("Oncorhynchus mykiss", "rainbow trout", Division.ANIMAL,
                    "Aquaculture breeding and disease resistance."),
        ),
        engines=_shared_engines() + (_ACMG_NOT_HUMAN,),
        status=DivisionStatus.EXPANSION,
    )


def _plant() -> DivisionInfo:
    return DivisionInfo(
        division=Division.PLANT,
        name="Plant",
        scope=(
            "Crop and plant genetics for agriculture worldwide: "
            "disease resistance, breeding, and stress tolerance in "
            "staple crops and model plants."
        ),
        problem_areas=(
            "staple crop disease resistance",
            "plant pathology and pathogen genetics",
            "crop breeding and genomic selection",
            "abiotic stress tolerance (drought, salinity, heat)",
            "yield and quality traits",
            "model plant genetics",
        ),
        example_species=(
            Species("Arabidopsis thaliana", "thale cress", Division.PLANT,
                    "The principal model plant."),
            Species("Oryza sativa", "rice", Division.PLANT,
                    "Staple crop feeding roughly half the world."),
            Species("Zea mays", "maize (corn)", Division.PLANT,
                    "Staple and feed crop with deep breeding genetics."),
            Species("Triticum aestivum", "bread wheat", Division.PLANT,
                    "Staple crop; hexaploid genome makes variant work distinctive."),
            Species("Solanum lycopersicum", "tomato", Division.PLANT,
                    "Model and commercial fruit crop."),
            Species("Solanum tuberosum", "potato", Division.PLANT,
                    "Staple tuber crop; late blight resistance is a flagship problem."),
            Species("Glycine max", "soybean", Division.PLANT,
                    "Major protein and oil crop."),
        ),
        engines=_shared_engines() + (_ACMG_NOT_HUMAN,),
        status=DivisionStatus.EXPANSION,
    )


class DivisionRegistry:
    """Registry of divisions and their species, with lookups."""

    def __init__(self, divisions: Iterable[DivisionInfo]) -> None:
        self._by_division: dict[Division, DivisionInfo] = {}
        self._by_species: dict[str, Species] = {}
        self._by_common_name: dict[str, Species] = {}
        for info in divisions:
            if info.division in self._by_division:
                raise ValueError(f"duplicate division: {info.division}")
            self._by_division[info.division] = info
            for species in info.example_species:
                if species.division is not info.division:
                    raise ValueError(
                        f"species {species.scientific_name} declares "
                        f"division {species.division}, filed under "
                        f"{info.division}"
                    )
                key = species.scientific_name.lower()
                if key in self._by_species:
                    raise ValueError(
                        f"species {species.scientific_name} appears "
                        "in more than one division entry; file it "
                        "under its primary division only"
                    )
                self._by_species[key] = species
                self._by_common_name[species.common_name.lower()] = species

    def get(self, division: Division | str) -> DivisionInfo:
        """Look up a division by enum value or string name."""
        if isinstance(division, str):
            try:
                division = Division(division.strip().lower())
            except ValueError as exc:
                raise KeyError(f"unknown division: {division!r}") from exc
        try:
            return self._by_division[division]
        except KeyError as exc:
            raise KeyError(f"unknown division: {division!r}") from exc

    def __contains__(self, division: object) -> bool:
        try:
            self.get(division)  # type: ignore[arg-type]
        except (KeyError, AttributeError):
            return False
        return True

    def __len__(self) -> int:
        return len(self._by_division)

    def all(self) -> tuple[DivisionInfo, ...]:
        """All divisions, in Human, Animal, Plant order."""
        order = (Division.HUMAN, Division.ANIMAL, Division.PLANT)
        return tuple(
            self._by_division[d] for d in order if d in self._by_division
        )

    def all_species(self) -> tuple[Species, ...]:
        """Every example species across all divisions."""
        result: list[Species] = []
        for info in self.all():
            result.extend(info.example_species)
        return tuple(result)

    def division_for_species(self, name: str) -> Optional[DivisionInfo]:
        """Find the division for a species by scientific/common name.

        Matching is case-insensitive and exact.  Returns ``None``
        when the species is not in the registry: absence means "not
        listed as an example", never "not supported in principle".
        """
        if not name:
            return None
        key = name.strip().lower()
        species = self._by_species.get(key) or self._by_common_name.get(key)
        if species is None:
            return None
        return self._by_division[species.division]

    def species(self, name: str) -> Optional[Species]:
        """Return the Species record for a name, or ``None``."""
        if not name:
            return None
        key = name.strip().lower()
        return self._by_species.get(key) or self._by_common_name.get(key)


_DEFAULT_REGISTRY: Optional[DivisionRegistry] = None


def default_registry() -> DivisionRegistry:
    """The built-in registry with the three divisions."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = DivisionRegistry([_human(), _animal(), _plant()])
    return _DEFAULT_REGISTRY


def get_division(division: Division | str) -> DivisionInfo:
    """Shortcut: look up one division in the default registry."""
    return default_registry().get(division)


def division_for_species(name: str) -> Optional[DivisionInfo]:
    """Shortcut: division lookup by species in the default registry."""
    return default_registry().division_for_species(name)


def all_divisions() -> tuple[DivisionInfo, ...]:
    """Shortcut: all divisions in the default registry."""
    return default_registry().all()
