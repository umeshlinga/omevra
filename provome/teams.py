"""Team registry: senior teams, each owning a slice of the work.

If :mod:`provome.roles` is the people layer of the Provome Copilot
organisation, this module is the org chart.  Work is divided the way
a real research organisation divides it: **separate teams for
separate problems**, each team led by a named senior role, each team
declaring exactly which division and which problem areas it owns.

Disease and species teams:

* Human division: Cancer Team, Tumor Biology Team, Rare Disease
  Team, Cardiac Team, Neuro Team, Infectious Disease Team.
* Cross-division technical team: DNA and Variant Team, the home of
  notation, sequence, and consequence work for every division.
* Animal division: Animal Health Team (companion animals, equine,
  wildlife) and Livestock Team (cattle, pigs, poultry, small
  ruminants, aquaculture).
* Plant division: Plant Crops Team and Plant Pathology Team.

Shared services, which serve every division and never own a question
on their own: the Evidence Verification Team, the Calibration and
Benchmarks Team, and the QC Team.  They are marked ``routable =
False``: the router sends questions to owning teams, and the
orchestra (:mod:`provome.orchestra`) calls the shared services in as
checkers on every question, which is a stronger guarantee than
routing some questions to them and some away.

Problem areas are taken from the division registry
(:mod:`provome.domains`) wherever a team's area is a division problem
area, so the org chart and the division registry cannot drift apart:
if a division does not declare an area, no team can quietly claim to
own it here without the mismatch being visible in the data.

Routing: :func:`route_question` maps a question - division, an
optional declared problem area, and the question text - to the
owning team by scoring problem-area ownership first and team keywords
second.  A question with no recognised area falls back to the
division's default team (DNA and Variant for Human, Animal Health
for Animal, Plant Crops for Plant), because a question must always
land on a named senior team's desk; "unassigned" is not a state this
organisation permits.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .domains import Division, get_division
from .roles import (
    ROLE_ANIMAL_DIVISION_LEAD,
    ROLE_HUMAN_DIVISION_LEAD,
    ROLE_PLANT_DIVISION_LEAD,
    ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
    ROLE_SENIOR_DATA_ANALYST,
    ROLE_SENIOR_EVIDENCE_REVIEWER,
    ROLE_SENIOR_QC_LEAD,
    ROLE_SENIOR_RESEARCHER,
    ROLE_SENIOR_SCIENTIST,
)

__all__ = [
    "Team",
    "TeamRegistry",
    "default_team_registry",
    "get_team",
    "all_teams",
    "route_question",
    "TEAM_CANCER",
    "TEAM_TUMOR_BIOLOGY",
    "TEAM_RARE_DISEASE",
    "TEAM_CARDIAC",
    "TEAM_NEURO",
    "TEAM_INFECTIOUS_DISEASE",
    "TEAM_DNA_VARIANT",
    "TEAM_ANIMAL_HEALTH",
    "TEAM_LIVESTOCK",
    "TEAM_PLANT_CROPS",
    "TEAM_PLANT_PATHOLOGY",
    "TEAM_EVIDENCE_VERIFICATION",
    "TEAM_CALIBRATION_BENCHMARKS",
    "TEAM_QC",
]

TEAM_CANCER = "cancer"
TEAM_TUMOR_BIOLOGY = "tumor-biology"
TEAM_RARE_DISEASE = "rare-disease"
TEAM_CARDIAC = "cardiac"
TEAM_NEURO = "neuro"
TEAM_INFECTIOUS_DISEASE = "infectious-disease"
TEAM_DNA_VARIANT = "dna-variant"
TEAM_ANIMAL_HEALTH = "animal-health"
TEAM_LIVESTOCK = "livestock"
TEAM_PLANT_CROPS = "plant-crops"
TEAM_PLANT_PATHOLOGY = "plant-pathology"
TEAM_EVIDENCE_VERIFICATION = "evidence-verification"
TEAM_CALIBRATION_BENCHMARKS = "calibration-benchmarks"
TEAM_QC = "qc"


@dataclass(frozen=True)
class Team:
    """One senior team in the organisation.

    Attributes:
        team_id: Stable identifier, for example ``"cancer"``.
        name: Display name, for example ``"Cancer Team"``.
        division: The division the team belongs to, or ``None`` for
            a cross-division team (the DNA and Variant Team and the
            shared services).
        lead_role_id: Id of the senior role that leads this team and
            signs off its work at senior review.
        member_role_ids: Ids of the senior roles that staff the
            team.  The lead role is listed separately and is not
            repeated here.
        problem_areas: Problem areas this team owns, worded to match
            the division registry where the area is a division area.
        keywords: Lowercase words and phrases the router matches
            against question text.  Keywords are routing hints, not
            science: ownership is decided by problem areas first.
        routable: True when the router may send a question to this
            team.  Shared services are not routable; they check every
            question instead of owning some of them.
        is_shared_service: True for the cross-division services
            (evidence verification, calibration and benchmarks, QC).
    """

    team_id: str
    name: str
    division: Optional[Division]
    lead_role_id: str
    member_role_ids: tuple[str, ...]
    problem_areas: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    routable: bool = True
    is_shared_service: bool = False

    def __post_init__(self) -> None:
        if not self.team_id or not self.team_id.strip():
            raise ValueError("team_id must be a non-empty string")
        if not self.name or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not self.lead_role_id:
            raise ValueError(f"team {self.team_id!r} must name a lead role")
        if self.lead_role_id in self.member_role_ids:
            raise ValueError(
                f"team {self.team_id!r} lists its lead role as a member; "
                "the lead is recorded separately"
            )
        if self.routable and not self.problem_areas and not self.keywords:
            raise ValueError(
                f"routable team {self.team_id!r} must own problem "
                "areas or keywords, or it can never be routed to"
            )
        if self.is_shared_service and self.routable:
            raise ValueError(
                f"shared service {self.team_id!r} must not be routable; "
                "shared services check every question instead"
            )

    def owns_problem_area(self, problem_area: str) -> bool:
        """True when this team declares ownership of a problem area."""
        wanted = problem_area.strip().lower()
        return any(area.lower() == wanted for area in self.problem_areas)

    def keyword_hits(self, text: str) -> int:
        """How many of this team's keywords appear in ``text``."""
        lowered = text.lower()
        return sum(1 for kw in self.keywords if kw in lowered)


def _human_areas() -> tuple[str, ...]:
    return get_division(Division.HUMAN).problem_areas


def _animal_areas() -> tuple[str, ...]:
    return get_division(Division.ANIMAL).problem_areas


def _plant_areas() -> tuple[str, ...]:
    return get_division(Division.PLANT).problem_areas


def _pick(areas: tuple[str, ...], *wanted: str) -> tuple[str, ...]:
    """Select declared division problem areas, failing loudly if absent.

    This is the drift guard described in the module docstring: if the
    division registry ever renames an area, building the team
    registry raises here instead of silently orphaning a team.
    """
    missing = [w for w in wanted if w not in areas]
    if missing:
        raise ValueError(
            f"division registry does not declare problem area(s): {missing}"
        )
    return tuple(w for w in wanted)


def _teams() -> tuple[Team, ...]:
    human = _human_areas()
    animal = _animal_areas()
    plant = _plant_areas()
    senior_staff = (
        ROLE_SENIOR_SCIENTIST,
        ROLE_SENIOR_RESEARCHER,
        ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
    )
    return (
        # -- Human division disease teams --------------------------
        Team(
            team_id=TEAM_CANCER,
            name="Cancer Team",
            division=Division.HUMAN,
            lead_role_id=ROLE_HUMAN_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(human, "cancer and somatic oncology"),
            keywords=(
                "cancer", "oncology", "carcinoma", "malignancy",
                "malignant", "sarcoma", "leukemia", "leukaemia",
                "lymphoma", "melanoma",
            ),
        ),
        Team(
            team_id=TEAM_TUMOR_BIOLOGY,
            name="Tumor Biology Team",
            division=Division.HUMAN,
            lead_role_id=ROLE_HUMAN_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(human, "cancer and somatic oncology"),
            keywords=(
                "tumor", "tumour", "neoplasm", "metastasis",
                "metastatic", "tumour biology", "tumor biology",
                "growth", "angiogenesis",
            ),
        ),
        Team(
            team_id=TEAM_RARE_DISEASE,
            name="Rare Disease Team",
            division=Division.HUMAN,
            lead_role_id=ROLE_HUMAN_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(
                human, "rare disease", "metabolic disease",
                "reproductive and prenatal genetics",
            ),
            keywords=(
                "rare disease", "orphan", "mendelian", "cystic fibrosis",
                "cftr", "syndrome", "inborn error", "lysosomal",
                "prenatal", "newborn screening",
            ),
        ),
        Team(
            team_id=TEAM_CARDIAC,
            name="Cardiac Team",
            division=Division.HUMAN,
            lead_role_id=ROLE_HUMAN_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(human, "cardiac disease"),
            keywords=(
                "cardiac", "cardiomyopathy", "heart", "arrhythmia",
                "long qt", "channelopathy", "aortic",
            ),
        ),
        Team(
            team_id=TEAM_NEURO,
            name="Neuro Team",
            division=Division.HUMAN,
            lead_role_id=ROLE_HUMAN_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(
                human, "neurological and neurodevelopmental disease"
            ),
            keywords=(
                "neuro", "neurological", "neurodevelopmental", "epilepsy",
                "seizure", "autism", "intellectual disability", "ataxia",
                "dementia", "parkinson", "alzheimer",
            ),
        ),
        Team(
            team_id=TEAM_INFECTIOUS_DISEASE,
            name="Infectious Disease Team",
            division=Division.HUMAN,
            lead_role_id=ROLE_HUMAN_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(
                human, "infectious disease and host genetics",
                "immunological disease",
            ),
            keywords=(
                "infectious", "infection", "pathogen", "viral", "virus",
                "bacterial", "host genetics", "immune", "immunological",
                "immunodeficiency",
            ),
        ),
        # -- Cross-division technical team --------------------------
        Team(
            team_id=TEAM_DNA_VARIANT,
            name="DNA and Variant Team",
            division=None,
            lead_role_id=ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
            member_role_ids=(
                ROLE_SENIOR_RESEARCHER,
                ROLE_SENIOR_DATA_ANALYST,
            ),
            problem_areas=(
                "variant interpretation",
                "sequence and consequence analysis",
                "genome and notation questions",
            ),
            keywords=(
                "variant", "hgvs", "vcf", "mutation", "dna",
                "sequence", "genome", "genomic", "allele", "genotype",
                "snp", "indel", "gene",
            ),
        ),
        # -- Animal division teams ----------------------------------
        Team(
            team_id=TEAM_ANIMAL_HEALTH,
            name="Animal Health Team",
            division=Division.ANIMAL,
            lead_role_id=ROLE_ANIMAL_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(
                animal,
                "companion animal inherited disease",
                "equine genetics",
                "wildlife and conservation genetics",
            ),
            keywords=(
                "dog", "canine", "cat", "feline", "companion animal",
                "pet", "horse", "equine", "veterinary", "vet",
                "wildlife", "conservation",
            ),
        ),
        Team(
            team_id=TEAM_LIVESTOCK,
            name="Livestock Team",
            division=Division.ANIMAL,
            lead_role_id=ROLE_ANIMAL_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(
                animal,
                "cattle health and breeding",
                "poultry disease and breeding",
                "pig health and breeding",
                "small ruminant health",
                "aquaculture disease and breeding",
            ),
            keywords=(
                "cattle", "cow", "bovine", "livestock", "herd",
                "dairy", "beef", "pig", "swine", "porcine", "poultry",
                "chicken", "sheep", "goat", "salmon", "trout",
                "aquaculture", "breeding",
            ),
        ),
        # -- Plant division teams -----------------------------------
        Team(
            team_id=TEAM_PLANT_CROPS,
            name="Plant Crops Team",
            division=Division.PLANT,
            lead_role_id=ROLE_PLANT_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(
                plant,
                "crop breeding and genomic selection",
                "yield and quality traits",
                "abiotic stress tolerance (drought, salinity, heat)",
                "model plant genetics",
            ),
            keywords=(
                "crop", "rice", "maize", "corn", "wheat", "soybean",
                "potato", "tomato", "yield", "breeding", "drought",
                "salinity", "heat tolerance", "arabidopsis",
            ),
        ),
        Team(
            team_id=TEAM_PLANT_PATHOLOGY,
            name="Plant Pathology Team",
            division=Division.PLANT,
            lead_role_id=ROLE_PLANT_DIVISION_LEAD,
            member_role_ids=senior_staff,
            problem_areas=_pick(
                plant,
                "staple crop disease resistance",
                "plant pathology and pathogen genetics",
            ),
            keywords=(
                "blight", "blast", "rust", "mildew", "pathogen",
                "fungal", "disease resistance", "late blight",
                "phytophthora", "wilt",
            ),
        ),
        # -- Shared services (check every question, own none) -------
        Team(
            team_id=TEAM_EVIDENCE_VERIFICATION,
            name="Evidence Verification Team",
            division=None,
            lead_role_id=ROLE_SENIOR_EVIDENCE_REVIEWER,
            member_role_ids=(
                ROLE_SENIOR_RESEARCHER,
                ROLE_SENIOR_SCIENTIST,
            ),
            routable=False,
            is_shared_service=True,
        ),
        Team(
            team_id=TEAM_CALIBRATION_BENCHMARKS,
            name="Calibration and Benchmarks Team",
            division=None,
            lead_role_id=ROLE_SENIOR_DATA_ANALYST,
            member_role_ids=(
                ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
                ROLE_SENIOR_RESEARCHER,
            ),
            routable=False,
            is_shared_service=True,
        ),
        Team(
            team_id=TEAM_QC,
            name="QC Team",
            division=None,
            lead_role_id=ROLE_SENIOR_QC_LEAD,
            member_role_ids=(
                ROLE_SENIOR_EVIDENCE_REVIEWER,
                ROLE_SENIOR_DATA_ANALYST,
            ),
            routable=False,
            is_shared_service=True,
        ),
    )


#: Default owning team per division when no area or keyword matches.
_DIVISION_DEFAULT_TEAM = {
    Division.HUMAN: TEAM_DNA_VARIANT,
    Division.ANIMAL: TEAM_ANIMAL_HEALTH,
    Division.PLANT: TEAM_PLANT_CROPS,
}


class TeamRegistry:
    """Registry of teams, with routing.

    Lookups are dictionary based.  Routing is a linear scan over the
    routable teams of one division (a small, fixed set), scoring
    declared problem-area ownership above keyword hits, so routing
    stays explainable: the registry can always say *why* a question
    went to a team, which the orchestra records in the audit trail.
    """

    def __init__(self, teams: Iterable[Team] = ()) -> None:
        self._teams: dict[str, Team] = {}
        for team in teams:
            self.register(team)

    def register(self, team: Team) -> None:
        """Add a team; raises ``ValueError`` on a duplicate id."""
        if team.team_id in self._teams:
            raise ValueError(f"duplicate team_id: {team.team_id!r}")
        self._teams[team.team_id] = team

    def get(self, team_id: str) -> Optional[Team]:
        return self._teams.get(team_id)

    def require(self, team_id: str) -> Team:
        team = self._teams.get(team_id)
        if team is None:
            raise KeyError(f"unknown team_id: {team_id!r}")
        return team

    def __contains__(self, team_id: object) -> bool:
        return team_id in self._teams

    def __len__(self) -> int:
        return len(self._teams)

    def all(self) -> tuple[Team, ...]:
        """All teams, in registration order."""
        return tuple(self._teams.values())

    def routable(self) -> tuple[Team, ...]:
        """Teams the router may assign a question to."""
        return tuple(t for t in self._teams.values() if t.routable)

    def shared_services(self) -> tuple[Team, ...]:
        """The cross-division shared service teams."""
        return tuple(t for t in self._teams.values() if t.is_shared_service)

    def for_division(self, division: Division) -> tuple[Team, ...]:
        """Routable teams serving a division, including cross-division."""
        return tuple(
            t
            for t in self.routable()
            if t.division is None or t.division is division
        )

    def route(
        self,
        division: Division,
        *,
        problem_area: Optional[str] = None,
        text: str = "",
    ) -> Team:
        """Route a question to its owning team.

        Scoring, highest first: a team that declares the given
        problem area wins outright; otherwise the team with the most
        keyword hits in the question text wins, with earlier
        registration breaking ties (disease teams are registered
        before the cross-division DNA and Variant Team, so a question
        that names a disease area goes to the disease team, and a
        purely technical question falls through to DNA and Variant).
        A question with no hits at all goes to the division default
        team.  The returned team always serves ``division``.
        """
        candidates = self.for_division(division)
        if problem_area:
            owners = [t for t in candidates if t.owns_problem_area(problem_area)]
            if owners:
                return owners[0]
        best: Optional[Team] = None
        best_hits = 0
        for team in candidates:
            hits = team.keyword_hits(text) if text else 0
            if hits > best_hits:
                best = team
                best_hits = hits
        if best is not None:
            return best
        return self.require(_DIVISION_DEFAULT_TEAM[division])


_DEFAULT_REGISTRY: Optional[TeamRegistry] = None


def default_team_registry() -> TeamRegistry:
    """The built-in registry with every team."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = TeamRegistry(_teams())
    return _DEFAULT_REGISTRY


def get_team(team_id: str) -> Team:
    """Shortcut: look up one team in the default registry."""
    return default_team_registry().require(team_id)


def all_teams() -> tuple[Team, ...]:
    """Shortcut: all teams in the default registry."""
    return default_team_registry().all()


def route_question(
    division: Division,
    *,
    problem_area: Optional[str] = None,
    text: str = "",
) -> Team:
    """Shortcut: route a question in the default registry."""
    return default_team_registry().route(
        division, problem_area=problem_area, text=text
    )
