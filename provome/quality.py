"""The Quality Division: 2,000 checkers over EVERY department.

Stood up 2026-11-25 on the Founder's direction: quality is not one
team watching one product - it covers the entire company, every
department, every corps. The Division is independent: its chief
reports to the CEO, not to the builders, and it holds a
stop-the-line veto over any release (roadmap Items 35-36).

Two laws are built into the assignments:

* No identical checks: every checker owns a distinct
  (department, method, beat) assignment, and every department is
  covered by ALL of the Division's distinct checking methods, so
  an important claim can always be verified by a pair of checkers
  using two DIFFERENT methods that must agree.
* Honesty: a registered checker is an organisational unit with a
  standing assignment. Asked to run without a live dispatched
  check, it says so and passes nothing - a checker that waves
  work through unchecked would defeat the Division's purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

from .diseases import FIRST_WAVE
from .orchestra import WorkContext, WorkResult, WorkTask, WorkerRegistry

__all__ = [
    "DEPARTMENTS",
    "METHODS",
    "QUALITY_DIVISION_SIZE",
    "QualityWorker",
    "build_quality_division",
]

QUALITY_DIVISION_SIZE = 2000

# Every department of the company, each with its own quality team:
# the orchestra, the research division, the three doors, drug
# discovery, the bridge group, and every disease corps.
DEPARTMENTS: tuple = (
    ("orchestra", "research-division", "human", "animal", "plant",
     "drug-discovery", "bridge")
    + tuple(name for name, _ in FIRST_WAVE)
)

# The Division's distinct checking methods. No two checkers on the
# same beat use the same method; important claims are checked by a
# pair using two different methods (Item 36).
METHODS: tuple = (
    "source-verification",
    "cross-source-triangulation",
    "citation-audit",
    "provenance-trace",
    "contradiction-scan",
    "statistical-recheck",
    "logic-consistency",
    "benchmark-replay",
    "reproducibility-replay",
    "coverage-audit",
    "freshness-check",
    "adversarial-red-team",
)


@dataclass
class QualityWorker:
    """One registered quality checker unit.

    Carries a distinct (department, method, beat) assignment. Its
    ``run`` is deliberately honest: without a live dispatched check
    behind it, it passes nothing and says so.
    """

    worker_id: str
    role_id: str
    team_id: str
    capabilities: tuple
    department: str = ""
    method: str = ""
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
                f"standing quality unit ({self.department}/"
                f"{self.method} beat {self.beat}): registered and "
                "ready; checks execute on live dispatch under the "
                "Quality Division's independent authority."
            ),
            ok=False,
            error="no live quality check dispatched to this unit",
        )


def department_sizes() -> dict:
    """The Division's 2,000 checkers, spread over every department."""
    base, remainder = divmod(QUALITY_DIVISION_SIZE, len(DEPARTMENTS))
    return {
        dept: base + (1 if i < remainder else 0)
        for i, dept in enumerate(DEPARTMENTS)
    }


def build_quality_division() -> WorkerRegistry:
    """Register the full Quality Division (2,000 units).

    Deterministic. The first unit registered is the Chief Quality
    Copilot; the first unit of each department is its senior QA
    lead; every unit's (department, method, beat) assignment is
    unique, and every department is covered by every method.
    """
    registry = WorkerRegistry()
    serial = 0
    for dept, size in department_sizes().items():
        for index in range(1, size + 1):
            serial += 1
            method = METHODS[(index - 1) % len(METHODS)]
            beat = (index - 1) // len(METHODS) + 1
            if serial == 1:
                role = "chief-quality-copilot"
            elif index == 1:
                role = "senior-qa-lead"
            else:
                role = "quality-checker"
            registry.register(
                QualityWorker(
                    worker_id=f"quality-{serial:04d}",
                    role_id=role,
                    team_id=f"quality-{dept}",
                    capabilities=(
                        "quality",
                        f"quality:{dept}",
                        f"method:{method}",
                    ),
                    department=dept,
                    method=method,
                    beat=beat,
                )
            )
    return registry


def _demo() -> None:
    registry = build_quality_division()
    sizes = department_sizes()
    print("Provome Quality Division (registered checkers)")
    print("=" * 72)
    for dept in DEPARTMENTS:
        print(f"  {dept:<22} {sizes[dept]:>6} checkers")
    print(f"  {'TOTAL':<22} {len(registry):>6} checkers")
    print(f"  departments covered: {len(DEPARTMENTS)}")
    print(f"  distinct methods:    {len(METHODS)}")
    print(
        "Note: registered organisational units; the Division is\n"
        "independent of the builders and holds a stop-the-line veto."
    )


if __name__ == "__main__":
    _demo()
