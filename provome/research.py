"""The Research Division: 5,000 registered scientific researchers.

Stood up 2026-11-25 on the Founder's direction. This module builds
the *organisation*: five thousand researcher units in the orchestra's
worker registry, distributed across the three doors plus Drug
Discovery and a cross-division bridge group, each holding a distinct
(specialty, beat) assignment, each group of one hundred led by a
senior research lead, all indexed by capability for assignment.

Honest scope, stated plainly: a registered researcher is an
organisational unit with a standing specialty - the same basis on
which the orchestra was stress-tested at ten thousand workers in
Step 1. Deep research execution happens when the orchestra
dispatches a live engine- or model-backed worker under a unit's
specialty; a unit asked to run without a live assignment says so
(``ok=False``) rather than returning a plausible-looking empty
success, per the orchestra's own rule. Registration is deployment
of the organisation, not a claim that five thousand scientists
have each produced research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .orchestra import WorkContext, WorkResult, WorkTask, WorkerRegistry

__all__ = [
    "DIVISION_SIZES",
    "RESEARCH_DIVISION_SIZE",
    "ResearchWorker",
    "build_research_division",
    "specialties_for",
]

RESEARCH_DIVISION_SIZE = 5000

# Founder-facing distribution: the Human door is the live wedge and
# carries the largest corps; Animal and Plant carry full corps of
# their own (three doors, never an afterthought); Drug Discovery a
# dedicated group; and a bridge group owns the cross-division gaps
# (cross-species translation, the map of what is unknown,
# reclassification watch).
DIVISION_SIZES: dict = {
    "human": 2000,
    "animal": 1000,
    "plant": 1000,
    "drug": 600,
    "bridge": 400,
}

_SPECIALTIES: dict = {
    "human": (
        "variant-evidence", "literature-surveillance", "rare-disease-assembly",
        "phenotype-evidence", "population-frequency", "clinical-guideline-watch",
        "cohort-evidence", "mechanism-evidence", "biomarker-evidence",
        "reclassification-watch",
    ),
    "animal": (
        "companion-animal-genetics", "livestock-genetics", "equine-genetics",
        "aquaculture-genetics", "poultry-genetics", "wildlife-genetics",
        "veterinary-literature", "breeding-evidence", "zoonosis-watch",
        "reclassification-watch",
    ),
    "plant": (
        "crop-genetics", "plant-pathology-evidence", "trait-evidence",
        "soil-microbiome-evidence", "seed-genetics", "drought-resistance",
        "pest-resistance", "plant-literature", "germplasm-evidence",
        "reclassification-watch",
    ),
    "drug": (
        "target-evidence", "compound-evidence", "mechanism-evidence",
        "repurposing-scan", "toxicity-literature", "trial-evidence-watch",
        "biomarker-targets", "resistance-evidence", "pathway-evidence",
        "reclassification-watch",
    ),
    "bridge": (
        "cross-species-translation", "gap-mapping", "negative-results",
        "one-health-watch", "evidence-provenance", "audit-standards",
        "low-resource-labs", "benchmark-design", "vault-curation",
        "reclassification-watch",
    ),
}

LEAD_EVERY = 100  # one senior research lead per hundred researchers


def specialties_for(division: str) -> tuple:
    """The specialty taxonomy of one research division."""
    return _SPECIALTIES[division]


@dataclass
class ResearchWorker:
    """One registered scientific researcher unit.

    Carries a distinct (division, specialty, beat) assignment. Its
    ``run`` is deliberately honest: without a live research engine
    behind it, it reports that it is a standing unit awaiting a live
    assignment under its specialty, and produces no claims.
    """

    worker_id: str
    role_id: str
    team_id: str
    capabilities: tuple
    division: str = ""
    specialty: str = ""
    beat: int = 0

    def run(self, task: WorkTask, context: WorkContext) -> WorkResult:
        return WorkResult(
            task_id=task.task_id,
            worker_id=self.worker_id,
            role_id=self.role_id,
            team_id=self.team_id,
            claims=(),
            evidence_items=(),
            notes=(
                f"standing research unit ({self.division}/"
                f"{self.specialty} beat {self.beat}): registered and "
                "ready; deep research executes when a live engine or "
                "model worker is dispatched under this specialty."
            ),
            ok=False,
            error="no live research assignment dispatched to this unit",
        )


def build_research_division() -> WorkerRegistry:
    """Register the full Research Division (5,000 units).

    Deterministic: the same build always produces the same registry,
    the same ids, and the same assignments. Every unit's
    (division, specialty, beat) assignment is unique; every group of
    ``LEAD_EVERY`` units is led by a senior research lead (the first
    unit of each group).
    """
    registry = WorkerRegistry()
    for division, size in DIVISION_SIZES.items():
        specialties = _SPECIALTIES[division]
        per_specialty = size // len(specialties)
        index = 0
        for specialty in specialties:
            for beat in range(1, per_specialty + 1):
                index += 1
                is_lead = (index - 1) % LEAD_EVERY == 0
                registry.register(
                    ResearchWorker(
                        worker_id=f"research-{division}-{index:04d}",
                        role_id=(
                            "senior-research-lead"
                            if is_lead
                            else "scientific-researcher"
                        ),
                        team_id=f"research-{division}",
                        capabilities=(
                            "research",
                            f"research:{division}",
                            f"specialty:{specialty}",
                        ),
                        division=division,
                        specialty=specialty,
                        beat=beat,
                    )
                )
    return registry


def _demo() -> None:
    registry = build_research_division()
    print("Provome Research Division (registered researcher units)")
    print("=" * 72)
    for division, size in DIVISION_SIZES.items():
        workers = registry.workers_for(f"research:{division}")
        leads = sum(1 for w in workers if w.role_id == "senior-research-lead")
        print(f"  {division:<8} {len(workers):>5} researchers, {leads} senior leads")
    print(f"  TOTAL    {len(registry):>5} researchers")
    print(
        "Note: registered organisational units with standing\n"
        "specialties; deep research executes on live dispatch.\n"
        "Registration is not a claim of completed research."
    )


if __name__ == "__main__":
    _demo()
