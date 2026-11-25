"""Three doors, one engine: the division product split (Item 31).

The architecture decision is that Provome is ONE shared engine -
the cite-or-decline gate, the source registry, the audit trail -
with THREE separate division products on top of it:

* **Provome Human** (console script ``provome-human``)
* **Provome Animal** (console script ``provome-animal``)
* **Provome Plant** (console script ``provome-plant``)

Each door wires ONLY its own division's copilot and provider, its
own boundary text, and its own example questions.  A user of one
door never sees a mixed product: the Animal door cannot issue
human evidence, the Human door cannot issue animal evidence, and
so on - not because this module filters anything, but because each
door's provider is division-scoped (see
:attr:`~provome.copilot.Copilot.ask`'s scope guard) and each door
asks only through its own provider.

This module is wiring only.  It contains no gate logic, no
provider logic, and no evidence handling of its own: the gate
stays in :mod:`provome.evidence`, the Copilot in
:mod:`provome.copilot`, and each division's provider in its own
module (:mod:`provome.live`, :mod:`provome.animal`,
:mod:`provome.plant`).  If a behaviour needs changing, it changes
there, once, for every door.

Usage:

    provome-human "What is known about CFTR p.Phe508del?"
    provome-animal --demo
    provome-plant "What is known about FLC in Arabidopsis?"

The Human and Animal/Plant defaults differ in one honest way:
the Human door's default provider is the live one (ClinVar,
gnomAD, Europe PMC - the same provider as ``python -m
provome.live``), guided by the conversational layer in
:mod:`provome.converse`; the Animal and Plant doors use their
live species providers.  All three therefore make network calls
by default, exactly like the module demos they wrap.  Tests and
hosts inject fixture providers instead.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .animal import BOUNDARY as ANIMAL_BOUNDARY
from .animal import AnimalCopilot
from .converse import Guide
from .copilot import Copilot
from .domains import Division, get_division
from .live import LiveEvidenceProvider
from .plant import BOUNDARY as PLANT_BOUNDARY
from .plant import PlantCopilot

__all__ = [
    "HUMAN_BOUNDARY",
    "Door",
    "HumanDoor",
    "AnimalDoor",
    "PlantDoor",
    "human_door",
    "animal_door",
    "plant_door",
    "get_door",
    "all_doors",
    "human_main",
    "animal_main",
    "plant_main",
]

#: The Human door's boundary text.  The Animal and Plant doors use
#: the BOUNDARY constants from their own modules, unchanged; the
#: Human division's standing wording is the one the Guide already
#: prints in its limits section ("Research software - not for
#: clinical use, and not a diagnosis").
HUMAN_BOUNDARY = (
    "Research software — not for clinical use, and not a "
    "diagnosis. This door answers only from evidence that passes "
    "the cite-or-decline gate; when the evidence is not there, "
    "it declines and names exactly what is missing."
)


class Door:
    """One division product: a copilot, a boundary, examples, a CLI.

    Subclasses wire exactly one division.  Everything behavioural
    (gating, parsing, fetching) lives in the wrapped copilot and
    its provider; a Door only frames the output with the product
    name and division label, appends the door's boundary where the
    wrapped surface does not already print it, and provides the
    shared command-line shape.
    """

    division: Division
    command: str  # console script name, e.g. "provome-human"
    boundary: str
    examples: tuple[str, ...] = ()

    @property
    def division_name(self) -> str:
        return get_division(self.division).name

    @property
    def product_name(self) -> str:
        return f"Provome {self.division_name}"

    @property
    def header(self) -> str:
        """The product + division label printed above every output."""
        return f"{self.product_name} — division: {self.division_name}"

    def ask(self, question: str) -> str:
        """Ask this door's copilot; return the framed output text."""
        raise NotImplementedError

    def _frame(self, body: str, *, boundary_included: bool) -> str:
        parts = [self.header, "", body]
        if not boundary_included:
            parts += ["", "Boundary: " + self.boundary]
        return "\n".join(parts)

    # -- command line -------------------------------------------------
    def main(self, argv: Optional[Sequence[str]] = None) -> int:
        """Console entry point: one question, or ``--demo``."""
        parser = argparse.ArgumentParser(
            prog=self.command,
            description=(
                f"{self.product_name}: the {self.division_name} "
                "division product of the Provome evidence engine. "
                "Answers only through the cite-or-decline gate."
            ),
        )
        parser.add_argument(
            "question",
            nargs="*",
            help="the question to ask this door",
        )
        parser.add_argument(
            "--demo",
            action="store_true",
            help="run this door's example questions and exit",
        )
        args = parser.parse_args(argv)
        if args.demo:
            for question in self.examples:
                print("=" * 72)
                print(self.ask(question))
                print()
            return 0
        if not args.question:
            parser.error("give a question, or pass --demo")
        print(self.ask(" ".join(args.question)))
        return 0


class HumanDoor(Door):
    """Provome Human: the Guide over the live human provider."""

    division = Division.HUMAN
    command = "provome-human"
    boundary = HUMAN_BOUNDARY
    examples = (
        "What is known about CFTR p.Phe508del?",
        "What is known about HBB p.Glu6Val?",
        "What can you tell me about the BRCA1 c.68_69delAG founder variant?",
    )

    def __init__(self, guide: Optional[Guide] = None) -> None:
        # Default: the conversational Guide over a Copilot running
        # the live human provider (ClinVar / gnomAD / Europe PMC).
        self.guide = (
            guide
            if guide is not None
            else Guide(Copilot(provider=LiveEvidenceProvider()))
        )

    @property
    def copilot(self) -> Copilot:
        return self.guide.copilot

    def ask(self, question: str) -> str:
        guided = self.guide.ask(question)
        return self._frame(guided.render(), boundary_included=False)


class AnimalDoor(Door):
    """Provome Animal: the Animal division copilot and provider."""

    division = Division.ANIMAL
    command = "provome-animal"
    boundary = ANIMAL_BOUNDARY
    examples = (
        "What is known about ABCB1 in dogs?",
        "What is known about PRKDC in horses?",
        "What is known about MYBPC3 in cats?",
    )

    def __init__(self, copilot: Optional[AnimalCopilot] = None) -> None:
        self.surface = copilot if copilot is not None else AnimalCopilot()

    @property
    def copilot(self) -> Copilot:
        return self.surface.copilot

    @property
    def provider(self):
        return self.surface.provider

    def ask(self, question: str) -> str:
        # AnimalCopilot.ask already appends this door's boundary.
        return self._frame(self.surface.ask(question), boundary_included=True)


class PlantDoor(Door):
    """Provome Plant: the Plant division copilot and provider."""

    division = Division.PLANT
    command = "provome-plant"
    boundary = PLANT_BOUNDARY
    examples = (
        "What is known about FLC in Arabidopsis?",
        "What is known about XA21 in rice?",
        "What is known about SP in tomato?",
    )

    def __init__(self, copilot: Optional[PlantCopilot] = None) -> None:
        self.surface = copilot if copilot is not None else PlantCopilot()

    @property
    def copilot(self) -> Copilot:
        return self.surface.copilot

    @property
    def provider(self):
        return self.surface.provider

    def ask(self, question: str) -> str:
        # PlantCopilot.ask already appends this door's boundary.
        return self._frame(self.surface.ask(question), boundary_included=True)


# -- factories ---------------------------------------------------------


def human_door(guide: Optional[Guide] = None) -> HumanDoor:
    return HumanDoor(guide=guide)


def animal_door(copilot: Optional[AnimalCopilot] = None) -> AnimalDoor:
    return AnimalDoor(copilot=copilot)


def plant_door(copilot: Optional[PlantCopilot] = None) -> PlantDoor:
    return PlantDoor(copilot=copilot)


_DOOR_CLASSES = {
    Division.HUMAN: HumanDoor,
    Division.ANIMAL: AnimalDoor,
    Division.PLANT: PlantDoor,
}


def get_door(division: Division | str) -> Door:
    """Build the (default, live) door for one division."""
    if isinstance(division, str):
        division = Division(division.strip().lower())
    return _DOOR_CLASSES[division]()


def all_doors() -> tuple[Door, ...]:
    """All three doors, in Human, Animal, Plant order."""
    return tuple(
        _DOOR_CLASSES[d]()
        for d in (Division.HUMAN, Division.ANIMAL, Division.PLANT)
    )


# -- console script entry points ---------------------------------------


def human_main(argv: Optional[Sequence[str]] = None) -> int:
    return human_door().main(argv)


def animal_main(argv: Optional[Sequence[str]] = None) -> int:
    return animal_door().main(argv)


def plant_main(argv: Optional[Sequence[str]] = None) -> int:
    return plant_door().main(argv)


if __name__ == "__main__":
    raise SystemExit(human_main())
