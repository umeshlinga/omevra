"""Disease corps: 5,000 dedicated researchers per disease area.

Stood up 2026-11-25 on the Founder's direction: every major disease
area gets its OWN corps of five thousand scientific researchers,
each with a distinct (focus, function, beat) assignment - never
5,000 units staring at one problem. Cancer, oncology, and tumour
are one field and form ONE corps (Oncology) with tumour types as
focus areas inside it; duplicating the same field under three
names would break the company's own no-duplicate law.

Each corps lives in its own worker registry (a corps is a unit of
organisation, like a department), which keeps registration and
assignment fast at this scale. The corps builder is generic: any
disease area in any division can be given a corps, and corps are
activated through the company's stage gates - the first wave below
is deployed; later waves stand up as the work reaches them.

Honest scope, as in :mod:`provome.research`: a registered
researcher is an organisational unit with a standing assignment.
Registration is the organisation standing up, not a claim of
completed research; a unit with no live dispatched assignment
says so and produces nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .orchestra import WorkerRegistry
from .research import LEAD_EVERY, ResearchWorker

__all__ = [
    "CORPS_SIZE",
    "FIRST_WAVE",
    "DiseaseCorps",
    "build_disease_corps",
    "deploy_first_wave",
]

CORPS_SIZE = 5000

# The first wave of disease corps: (corps name, division).
# Ten corps x 5,000 = 50,000 disease researchers.
FIRST_WAVE: tuple = (
    ("oncology", "human"),
    ("cardiovascular", "human"),
    ("infectious-diseases", "human"),
    ("neurological", "human"),
    ("rare-diseases", "human"),
    ("metabolic", "human"),
    ("animal-infectious", "animal"),
    ("animal-genetic", "animal"),
    ("plant-pathology", "plant"),
    ("crop-resistance", "plant"),
)

# Focus areas per corps: the distinct subjects its researchers own.
_FOCUS: dict = {
    "oncology": (
        "lung", "breast", "colorectal", "blood", "skin", "prostate",
        "pancreatic", "liver", "brain-tumours", "paediatric-tumours",
        "rare-tumours", "tumour-immunology",
    ),
    "cardiovascular": (
        "heart-failure", "arrhythmia", "stroke", "hypertension",
        "lipid-disorders", "congenital-heart", "valve-disease",
        "peripheral-vascular",
    ),
    "infectious-diseases": (
        "covid-19", "influenza", "tuberculosis", "hiv", "malaria",
        "antimicrobial-resistance", "emerging-viruses", "sepsis",
    ),
    "neurological": (
        "alzheimer", "parkinson", "epilepsy", "multiple-sclerosis",
        "als", "migraine", "stroke-recovery", "rare-neurological",
    ),
    "rare-diseases": (
        "metabolic-rare", "neuromuscular-rare", "immune-rare",
        "skeletal-rare", "renal-rare", "undiagnosed", "newborn-screening",
        "rare-evidence-assembly",
    ),
    "metabolic": (
        "type2-diabetes", "type1-diabetes", "obesity", "thyroid",
        "lipid-metabolic", "liver-metabolic", "rare-metabolic",
        "metabolic-syndrome",
    ),
    "animal-infectious": (
        "canine-infectious", "feline-infectious", "cattle-infectious",
        "swine-infectious", "poultry-infectious", "equine-infectious",
        "aquaculture-infectious", "zoonosis-bridge",
    ),
    "animal-genetic": (
        "canine-genetic", "feline-genetic", "equine-genetic",
        "cattle-genetic", "breeding-health", "companion-oncology",
        "rare-animal-genetic", "wildlife-genetic",
    ),
    "plant-pathology": (
        "fungal-disease", "bacterial-disease", "viral-disease",
        "nematodes", "wheat-disease", "rice-disease", "maize-disease",
        "emerging-pathogens",
    ),
    "crop-resistance": (
        "drought", "salinity", "heat", "pest-resistance",
        "disease-resistance", "nutrient-use", "yield-stability",
        "climate-adaptation",
    ),
}

# Evidence functions: the distinct kind of work done on each focus.
_FUNCTIONS: tuple = (
    "evidence-assembly", "literature-surveillance", "variant-evidence",
    "mechanism-evidence", "gap-mapping", "cohort-evidence",
    "guideline-watch", "reclassification-watch",
)


@dataclass
class DiseaseCorps:
    """One disease area's researcher corps, in its own registry."""

    name: str
    division: str
    registry: WorkerRegistry

    def __len__(self) -> int:
        return len(self.registry)


def build_disease_corps(
    name: str, division: str, size: int = CORPS_SIZE
) -> DiseaseCorps:
    """Register one disease corps of ``size`` researchers.

    Assignments are the cross product of the corps' focus areas and
    the shared evidence functions, filled with numbered beats so
    every unit's (focus, function, beat) assignment is unique. The
    first unit of every hundred is a senior research lead.
    """
    focuses = _FOCUS[name]
    assignments = [
        (focus, function) for focus in focuses for function in _FUNCTIONS
    ]
    registry = WorkerRegistry()
    for index in range(1, size + 1):
        focus, function = assignments[(index - 1) % len(assignments)]
        beat = (index - 1) // len(assignments) + 1
        is_lead = (index - 1) % LEAD_EVERY == 0
        registry.register(
            ResearchWorker(
                worker_id=f"disease-{name}-{index:05d}",
                role_id=(
                    "senior-research-lead"
                    if is_lead
                    else "scientific-researcher"
                ),
                team_id=f"disease-{name}",
                capabilities=(
                    f"disease:{name}",
                    f"focus:{name}:{focus}",
                    f"function:{function}",
                ),
                division=division,
                specialty=f"{focus}/{function}",
                beat=beat,
            )
        )
    return DiseaseCorps(name=name, division=division, registry=registry)


def deploy_first_wave() -> dict:
    """Build every corps in the first wave; returns name -> corps."""
    return {
        name: build_disease_corps(name, division)
        for name, division in FIRST_WAVE
    }


def _demo() -> None:
    print("Provome disease corps - first wave (registered researchers)")
    print("=" * 72)
    total = 0
    for name, division in FIRST_WAVE:
        corps = build_disease_corps(name, division)
        total += len(corps)
        print(f"  {name:<22} ({division:<6}) {len(corps):>6} researchers")
    print(f"  {'TOTAL':<22}          {total:>6} researchers")
    print(
        "Note: registered organisational units with distinct\n"
        "assignments; research executes on live dispatch."
    )


if __name__ == "__main__":
    _demo()
