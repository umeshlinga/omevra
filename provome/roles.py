"""Senior role registry: the people layer of the Provome organisation.

The Provome Copilot is one copilot on the outside.  Inside, it is an
organisation, and an organisation is made of roles before it is made
of code.  This module is the registry of those roles.

Every role in this registry is **senior level by design**.  There are
no junior, intern, or unsupervised roles anywhere in the Provome
Copilot organisation: the work this system does - weighing evidence
about variants, genes, and diseases - is work where an unsupervised
junior mistake becomes a wrong answer given to a scientist, so the
registry simply does not define junior roles.  The seniority ladder
here therefore starts at Senior and rises through Lead and Principal
to Chief:

* ``SENIOR`` - a senior individual contributor: a senior scientist,
  senior researcher, senior engineer.  Owns their own work end to
  end and is cross-checked by peers, like every senior in a real
  production organisation.
* ``LEAD`` - a senior who also leads a team or a division: signs off
  team output, orders re-runs, resolves disagreements escalated by
  the team.  A lead's sign-off is a QC stage in
  :mod:`provome.orchestra`, not a courtesy.
* ``PRINCIPAL`` - a senior with organisation-wide scope: production
  standards, evidence standards, QC standards.  Principals set the
  rules the gates enforce.
* ``CHIEF`` - exactly one role: the Chief Copilot, the orchestrator
  that receives every question, routes it, collects the work, and
  issues (or declines) the final answer.  There is one Chief because
  there is one Copilot.

Roles are data, not classes: a :class:`Role` is a typed record with
an id, a title, a seniority level, a tuple of responsibilities, and
a team scope.  Teams (:mod:`provome.teams`) are built by naming the
role that leads them and the roles that staff them, and the
orchestration engine (:mod:`provome.orchestra`) assigns work to
workers who hold these roles.  Keeping roles in a registry - rather
than scattering titles through the code - means the organisation
chart can be audited, tested, and extended in one place, which is
what a production organisation requires as it scales.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

__all__ = [
    "Seniority",
    "Role",
    "RoleRegistry",
    "default_role_registry",
    "get_role",
    "all_roles",
    "ROLE_CHIEF_COPILOT",
    "ROLE_SENIOR_PRODUCTION_MANAGER",
    "ROLE_SENIOR_PRODUCTION_ENGINEER",
    "ROLE_SENIOR_SCIENTIST",
    "ROLE_SENIOR_RESEARCHER",
    "ROLE_SENIOR_DATA_ANALYST",
    "ROLE_SENIOR_BIOINFORMATICS_SCIENTIST",
    "ROLE_SENIOR_EVIDENCE_REVIEWER",
    "ROLE_SENIOR_QC_LEAD",
    "ROLE_HUMAN_DIVISION_LEAD",
    "ROLE_ANIMAL_DIVISION_LEAD",
    "ROLE_PLANT_DIVISION_LEAD",
]


class Seniority(str, Enum):
    """Seniority levels in the Provome Copilot organisation.

    Every level is a senior level.  The ladder exists to express
    scope and sign-off authority, not to admit junior grades: the
    lowest rung here, ``SENIOR``, is already a fully accountable
    senior practitioner.
    """

    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    CHIEF = "chief"

    @property
    def rank(self) -> int:
        """Ordering rank: higher means wider sign-off authority."""
        return {
            Seniority.SENIOR: 1,
            Seniority.LEAD: 2,
            Seniority.PRINCIPAL: 3,
            Seniority.CHIEF: 4,
        }[self]

    @property
    def is_senior(self) -> bool:
        """True for every level: this organisation has no junior roles."""
        return True

    def outranks(self, other: "Seniority") -> bool:
        return self.rank > other.rank


@dataclass(frozen=True)
class Role:
    """One senior role in the organisation.

    Attributes:
        role_id: Stable identifier, for example ``"chief-copilot"``.
        title: Human readable title, for example ``"Chief Copilot"``.
        seniority: Seniority level.  Always a senior level; the
            registry refuses anything else by construction, because
            :class:`Seniority` has no non-senior member.
        responsibilities: What this role is accountable for, stated
            as plain sentences.  Responsibilities are part of the
            audit surface: when a worker acting in this role is
            reported for a failed result, the report is read against
            these responsibilities.
        team_scope: The teams or areas this role operates in, in
            plain language (for example ``"all teams, all
            divisions"`` for the Chief).
        division: Optional division id (``"human"``, ``"animal"``,
            ``"plant"``) for division-specific roles such as the
            division leads; ``None`` for organisation-wide roles.
    """

    role_id: str
    title: str
    seniority: Seniority
    responsibilities: tuple[str, ...]
    team_scope: str
    division: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.role_id or not self.role_id.strip():
            raise ValueError("role_id must be a non-empty string")
        if not self.title or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if not self.responsibilities:
            raise ValueError(f"role {self.role_id!r} must name responsibilities")
        if not self.team_scope or not self.team_scope.strip():
            raise ValueError(f"role {self.role_id!r} must name a team scope")
        if not self.seniority.is_senior:
            raise ValueError(
                f"role {self.role_id!r} is not senior level; the "
                "Provome Copilot organisation defines senior roles only"
            )

    @property
    def is_senior(self) -> bool:
        """True for every role in this organisation, by design."""
        return self.seniority.is_senior


# ---------------------------------------------------------------------
# Role ids.  Constants so teams and the orchestra never spell a role
# id as a bare string in more than one place.
# ---------------------------------------------------------------------

ROLE_CHIEF_COPILOT = "chief-copilot"
ROLE_SENIOR_PRODUCTION_MANAGER = "senior-production-manager"
ROLE_SENIOR_PRODUCTION_ENGINEER = "senior-production-engineer"
ROLE_SENIOR_SCIENTIST = "senior-scientist"
ROLE_SENIOR_RESEARCHER = "senior-researcher"
ROLE_SENIOR_DATA_ANALYST = "senior-data-analyst"
ROLE_SENIOR_BIOINFORMATICS_SCIENTIST = "senior-bioinformatics-scientist"
ROLE_SENIOR_EVIDENCE_REVIEWER = "senior-evidence-reviewer"
ROLE_SENIOR_QC_LEAD = "senior-qc-lead"
ROLE_HUMAN_DIVISION_LEAD = "human-division-lead"
ROLE_ANIMAL_DIVISION_LEAD = "animal-division-lead"
ROLE_PLANT_DIVISION_LEAD = "plant-division-lead"


def _roles() -> tuple[Role, ...]:
    """The built-in senior role registry content.

    Order is deliberate: the Chief first, then organisation-wide
    senior roles, then the three division leads.
    """
    return (
        Role(
            role_id=ROLE_CHIEF_COPILOT,
            title="Chief Copilot",
            seniority=Seniority.CHIEF,
            responsibilities=(
                "Receive every question put to the Provome Copilot and own the final answer or the final decline",
                "Classify each question by division and problem area and route it to the owning team",
                "Assign sub-tasks to senior workers and collect their structured results",
                "Order cross-checks, re-runs, and tiebreaks when workers disagree",
                "Issue an answer only when every QC stage and the senior review have passed",
            ),
            team_scope="all teams, all divisions",
        ),
        Role(
            role_id=ROLE_SENIOR_PRODUCTION_MANAGER,
            title="Senior Production Manager",
            seniority=Seniority.PRINCIPAL,
            responsibilities=(
                "Own the production flow of the Copilot organisation: intake, assignment, deadlines of work, and escalation paths",
                "Keep the audit trail complete: every assignment, result, check, and exclusion accounted for",
                "Escalate stalled or repeatedly failing work to the Chief Copilot with a written summary",
            ),
            team_scope="all teams, all divisions (production operations)",
        ),
        Role(
            role_id=ROLE_SENIOR_PRODUCTION_ENGINEER,
            title="Senior Production Engineer",
            seniority=Seniority.SENIOR,
            responsibilities=(
                "Build and run the deterministic engine workers: consequence, classification, frequency, calibration, and phenotype computations",
                "Guarantee reproducibility: the same input to an engine worker produces the same result, byte for byte in the numbers that matter",
                "Report engine failures as failed results, never as silent empty answers",
            ),
            team_scope="engine workers across all teams",
        ),
        Role(
            role_id=ROLE_SENIOR_SCIENTIST,
            title="Senior Scientist",
            seniority=Seniority.SENIOR,
            responsibilities=(
                "Own the scientific correctness of claims in their disease or species area",
                "Judge whether evidence genuinely supports a claim, beyond the mechanical gate",
                "Flag claims that are technically cited but scientifically misleading",
            ),
            team_scope="disease and species teams in their division",
        ),
        Role(
            role_id=ROLE_SENIOR_RESEARCHER,
            title="Senior Researcher",
            seniority=Seniority.SENIOR,
            responsibilities=(
                "Gather and structure evidence for assigned questions from registered sources",
                "Record exactly what each source says, including findings that refute the working claim",
                "Surface gaps in the evidence honestly instead of filling them with plausible text",
            ),
            team_scope="research work across all teams",
        ),
        Role(
            role_id=ROLE_SENIOR_DATA_ANALYST,
            title="Senior Data Analyst",
            seniority=Seniority.SENIOR,
            responsibilities=(
                "Own every number the organisation produces: frequencies, scores, calibration metrics",
                "Recompute numeric claims independently when assigned as a cross-checker",
                "Reject numeric claims that cannot be recomputed from the stated inputs",
            ),
            team_scope="calibration, benchmarks, and population work across all teams",
        ),
        Role(
            role_id=ROLE_SENIOR_BIOINFORMATICS_SCIENTIST,
            title="Senior Bioinformatics Scientist",
            seniority=Seniority.SENIOR,
            responsibilities=(
                "Own variant, sequence, and genome work: notation, consequence calling, and coordinate correctness",
                "Verify that a variant claim names its reference assembly and transcript context where it matters",
                "Catch notation and consequence errors in other workers' results during cross-checks",
            ),
            team_scope="DNA and variant work across all teams",
        ),
        Role(
            role_id=ROLE_SENIOR_EVIDENCE_REVIEWER,
            title="Senior Evidence Reviewer",
            seniority=Seniority.SENIOR,
            responsibilities=(
                "Verify that citations say what the claims built on them say they say",
                "Check source independence: repeated copies of one curation are one line of evidence, not several",
                "Fail any claim whose citation is missing, wrong, or weaker than the claim",
            ),
            team_scope="Evidence Verification Team, serving all divisions",
        ),
        Role(
            role_id=ROLE_SENIOR_QC_LEAD,
            title="Senior QC Lead",
            seniority=Seniority.LEAD,
            responsibilities=(
                "Own the QC pipeline: evidence gate, citation, calculation, consistency, cross-worker agreement, and senior review stages",
                "Sign off a team's collected work, or refuse sign-off with the failed stages named",
                "Report workers whose results fail checks, with the failure recorded in the audit trail",
            ),
            team_scope="QC Team, serving all divisions",
        ),
        Role(
            role_id=ROLE_HUMAN_DIVISION_LEAD,
            title="Human Division Lead",
            seniority=Seniority.LEAD,
            responsibilities=(
                "Lead the Human division: all human disease areas, not only cancer",
                "Sign off Human division team work at senior review, or escalate it to the Chief Copilot",
                "Guard the boundary of human-clinical frameworks: ACMG/AMP applies in this division and only here",
            ),
            team_scope="all Human division teams",
            division="human",
        ),
        Role(
            role_id=ROLE_ANIMAL_DIVISION_LEAD,
            title="Animal Division Lead",
            seniority=Seniority.LEAD,
            responsibilities=(
                "Lead the Animal division: veterinary, livestock, and aquaculture genetics",
                "Sign off Animal division team work at senior review, or escalate it to the Chief Copilot",
                "Refuse human-clinical frameworks for animal questions and require division-appropriate standards",
            ),
            team_scope="all Animal division teams",
            division="animal",
        ),
        Role(
            role_id=ROLE_PLANT_DIVISION_LEAD,
            title="Plant Division Lead",
            seniority=Seniority.LEAD,
            responsibilities=(
                "Lead the Plant division: crops, plant pathology, and breeding genetics",
                "Sign off Plant division team work at senior review, or escalate it to the Chief Copilot",
                "Refuse human-clinical frameworks for plant questions and require division-appropriate standards",
            ),
            team_scope="all Plant division teams",
            division="plant",
        ),
    )


class RoleRegistry:
    """Registry of senior roles, keyed by role id.

    The registry validates on construction: duplicate ids are
    refused, and every role is checked senior by the :class:`Role`
    record itself.  Lookups are dictionary based, so the registry
    stays fast when host applications register additional senior
    roles for specialist workers at scale.
    """

    def __init__(self, roles: Iterable[Role] = ()) -> None:
        self._roles: dict[str, Role] = {}
        for role in roles:
            self.register(role)

    def register(self, role: Role) -> None:
        """Add a role; raises ``ValueError`` on a duplicate id."""
        if role.role_id in self._roles:
            raise ValueError(f"duplicate role_id: {role.role_id!r}")
        self._roles[role.role_id] = role

    def get(self, role_id: str) -> Optional[Role]:
        """Return the role with this id, or ``None``."""
        return self._roles.get(role_id)

    def require(self, role_id: str) -> Role:
        """Return the role with this id or raise ``KeyError``."""
        role = self._roles.get(role_id)
        if role is None:
            raise KeyError(f"unknown role_id: {role_id!r}")
        return role

    def __contains__(self, role_id: object) -> bool:
        return role_id in self._roles

    def __len__(self) -> int:
        return len(self._roles)

    def all(self) -> tuple[Role, ...]:
        """All roles, in registration order."""
        return tuple(self._roles.values())

    def by_seniority(self, seniority: Seniority) -> tuple[Role, ...]:
        """All roles at one seniority level, in registration order."""
        return tuple(r for r in self._roles.values() if r.seniority is seniority)

    def by_division(self, division: Optional[str]) -> tuple[Role, ...]:
        """Roles scoped to a division id; ``None`` gives org-wide roles."""
        return tuple(r for r in self._roles.values() if r.division == division)

    def leads(self) -> tuple[Role, ...]:
        """All roles with sign-off authority (Lead and above)."""
        return tuple(
            r for r in self._roles.values() if r.seniority.rank >= Seniority.LEAD.rank
        )

    def chief(self) -> Role:
        """The single Chief role; raises if the registry has none."""
        chiefs = self.by_seniority(Seniority.CHIEF)
        if len(chiefs) != 1:
            raise ValueError(
                f"a role registry for the Copilot organisation must "
                f"hold exactly one Chief role, found {len(chiefs)}"
            )
        return chiefs[0]


_DEFAULT_REGISTRY: Optional[RoleRegistry] = None


def default_role_registry() -> RoleRegistry:
    """The built-in registry with every senior role."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = RoleRegistry(_roles())
    return _DEFAULT_REGISTRY


def get_role(role_id: str) -> Role:
    """Shortcut: look up one role in the default registry."""
    return default_role_registry().require(role_id)


def all_roles() -> tuple[Role, ...]:
    """Shortcut: all roles in the default registry."""
    return default_role_registry().all()
